import math

import torch
import torch.optim as optim

from kfac_utils import (ComputeCovA, ComputeCovG)
from kfac_utils import update_running_stat
from kfac_utils import try_contiguous
from kfac_utils import cycle

import horovod.torch as hvd

class KFAC(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.1,
                 stat_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 TCov=10,
                 TInv=100,
                 batch_averaged=True,
                 inv_block_count=1):
        
        defaults = dict(lr=lr, damping=damping)
        super(KFAC, self).__init__(model.parameters(), defaults)
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged

        self.known_modules = {'Linear', 'Conv2d'}

        self.modules = []
        self.known_params = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        self.m_aa_inv, self.m_gg_inv = {}, {}


        self.stat_decay = stat_decay
        self.kl_clip = kl_clip
        self.TCov = TCov
        self.TInv = TInv
        self.inv_block_count = inv_block_count

        self.acc_stats = True
        self.rank_iter = cycle(list(range(hvd.size())))

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            aa = self.CovAHandler(input[0].data, module)
            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))
            update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.acc_stats and self.steps % self.TCov == 0:
            gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))
            update_running_stat(gg, self.m_gg[module], self.stat_decay)

    def _prepare_model(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)

    def _update_a_inv(self, m, ranks, damping):
        self.m_aa_inv[m] = try_contiguous(torch.zeros_like(self.m_aa[m]))
        if hvd.rank() not in ranks:
            return
        self.m_aa_inv[m].add_(
                self._approx_inverse_block(self.m_aa[m], ranks, damping))

    def _update_g_inv(self, m, ranks, damping):
        self.m_gg_inv[m] = try_contiguous(torch.zeros_like(self.m_gg[m]))
        if hvd.rank() not in ranks:
            return 
        self.m_gg_inv[m] += self._approx_inverse_block(self.m_gg[m], ranks, damping)

    def _approx_inverse_block(self, a, ranks, damping):
        i = ranks.index(hvd.rank())
        n = len(ranks)
        x_len, y_len = a.shape[0] // n, a.shape[1] // n
        if n >= min(x_len, y_len):
            n = min(x_len, y_len)
        xs, ys = i*x_len, i*y_len
        xe = (i+1)*x_len if i + 1 < n else a.shape[0]
        ye = (i+1)*y_len if i + 1 < n else a.shape[1]
        
        inverse = torch.zeros_like(a)

        if i < n:
            diag_block = a[xs:xe, ys:ye].clone()
            diag_damping = diag_block.new(diag_block.shape[0]).fill_(damping)
            diag_block += torch.diag(diag_damping)
            # Use cholesky inverse because torch.inverse() was segfaulting
            # on longhorn
            u = torch.cholesky(diag_block)
            inverse[xs:xe, ys:ye] += torch.cholesky_inverse(u)

        return inverse

    @staticmethod
    def _get_matrix_form_grad(m, classname):
        """
        :param m: the layer
        :param classname: the class name of the layer
        :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
        """
        if classname == 'Conv2d':
            p_grad_mat = m.weight.grad.data.view(m.weight.grad.data.size(0), -1)  # n_filters * (in_c * kw * kh)
        else:
            p_grad_mat = m.weight.grad.data
        if m.bias is not None:
            p_grad_mat = torch.cat([p_grad_mat, m.bias.grad.data.view(-1, 1)], 1)
        return p_grad_mat

    def _get_natural_grad(self, m, p_grad_mat):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        v = self.m_gg_inv[m] @ p_grad_mat @ self.m_aa_inv[m]

        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]
        return v

    def _kl_clip_and_update_grad(self, updates, lr):
        # do kl clip
        vg_sum = 0
        for m in self.modules:
            v = updates[m]
            vg_sum += (v[0] * m.weight.grad.data * lr ** 2).sum().item()
            if m.bias is not None:
                vg_sum += (v[1] * m.bias.grad.data * lr ** 2).sum().item()
        nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))

        for m in self.modules:
            v = updates[m]
            m.weight.grad.data.copy_(v[0])
            m.weight.grad.data.mul_(nu)
            if m.bias is not None:
                m.bias.grad.data.copy_(v[1])
                m.bias.grad.data.mul_(nu)

    def kfac_step(self):
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = {}
        handles = []

        if hvd.size() > 1 and self.steps % self.TCov == 0:
            self.sync_handles()

        # Compute and gather inverses across workers
        if self.steps % self.TInv == 0:
            for m in self.modules:
                a_ranks = self.rank_iter.next(self.inv_block_count)
                g_ranks = self.rank_iter.next(self.inv_block_count)

                self._update_a_inv(m, a_ranks, damping)
                self._update_g_inv(m, g_ranks, damping)
                
                if hvd.size() > 1:
                    handles.append(hvd.allreduce_async_(self.m_aa_inv[m], op=hvd.Sum))
                    handles.append(hvd.allreduce_async_(self.m_gg_inv[m], op=hvd.Sum))

            for handle in handles:
                hvd.synchronize(handle)

        for m in self.modules:
            classname = m.__class__.__name__
            p_grad_mat = self._get_matrix_form_grad(m, classname)
            v = self._get_natural_grad(m, p_grad_mat)
            updates[m] = v

        self._kl_clip_and_update_grad(updates, lr)

    def step(self, closure=None):
        if hvd.size() > 1:
            self.allreduce_grads()
        self.kfac_step()
        self.steps += 1

    def allreduce_grads(self):
        handles = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                handles.append(hvd.allreduce_async_(p.grad, op=hvd.Average))
        for handle in handles:
            hvd.synchronize(handle)

    def sync_handles(self):
        handles = []

        for m in self.modules:
            handles.append(hvd.allreduce_async_(self.m_aa[m].data, op=hvd.Average))
            handles.append(hvd.allreduce_async_(self.m_gg[m].data, op=hvd.Average))
            #if m.weight.grad is not None:
            #    handles.append(hvd.allreduce_async_(m.weight.grad, op=hvd.Average))
            #    if m.bias is not None:
            #        handles.append(hvd.allreduce_async_(m.bias.grad, op=hvd.Average))

        for handle in handles:
            hvd.synchronize(handle)

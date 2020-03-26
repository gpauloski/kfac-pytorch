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
       
        if inv_block_count > 1:
            raise ValueError("Inv block count > 1 not implemented yet")
 
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

        self.m_a, self.m_g = {}, {}
        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}

        self.stat_decay = stat_decay
        self.kl_clip = kl_clip
        self.TCov = TCov
        self.TInv = TInv
        self.inv_block_count = inv_block_count

        self.rank_iter = cycle(list(range(hvd.size())))

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            #aa = self.CovAHandler(input[0].data, module)
            # Initialize buffers
            #if self.steps == 0:
            #    self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))
            #update_running_stat(aa, self.m_aa[module], self.stat_decay)
            self.m_a[module] = input[0].data

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.steps % self.TCov == 0:
            #gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
            # Initialize buffers
            #if self.steps == 0:
            #    self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))
            #update_running_stat(gg, self.m_gg[module], self.stat_decay)
            self.m_g[module] = grad_output[0].data

    def _update_cov_a(self, module): 
        aa = self.CovAHandler(self.m_a[module], module)
        # Initialize buffers
        if self.steps == 0:
            self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))
        update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _update_cov_g(self, module):
        gg = self.CovGHandler(self.m_g[module], module, self.batch_averaged)
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

    def _update_inv(self, m):
        eps = 1e-10  # for numerical stability

        self.d_a[m], self.Q_a[m] = torch.symeig(
            self.m_aa[m], eigenvectors=True)
        self.d_g[m], self.Q_g[m] = torch.symeig(
            self.m_gg[m], eigenvectors=True)

        self.d_g[m].mul_((self.d_g[m] > eps).float())
        self.d_a[m].mul_((self.d_a[m] > eps).float())

    @staticmethod
    def _get_matrix_form_grad(m, classname):
        """
        :param m: the layer
        :param classname: the class name of the layer
        :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
        """
        if classname == 'Conv2d':
            # n_filters * (in_c * kw * kh)
            p_grad_mat = m.weight.grad.data.view(m.weight.grad.data.size(0), -1)  
        else:
            p_grad_mat = m.weight.grad.data
        if m.bias is not None:
            p_grad_mat = torch.cat([p_grad_mat, m.bias.grad.data.view(-1, 1)], 1)
        return p_grad_mat

    def _get_natural_grad(self, m, p_grad_mat, damping):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
        v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping)
        v = self.Q_g[m] @ v2 @ self.Q_a[m].t()

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

    def step(self, closure=None):
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = {}
        handles = []

        if hvd.size() > 1:
            self.allreduce_grads()

        if self.steps % self.TCov == 0:
            for module in self.modules:
                self._update_cov(module)
            if hvd.size() > 1:
                self.sync_handles()

        for i, m in enumerate(self.modules):
            #rank = self.rank_iter.next(1)
            #if hvd.rank() in rank:
            if i % hvd.size() == hvd.rank():
                classname = m.__class__.__name__
                if self.steps % self.TInv == 0:
                    self._update_inv(m)
                p_grad_mat = self._get_matrix_form_grad(m, classname)
                v = self._get_natural_grad(m, p_grad_mat, damping)
                v = [try_contiguous(x) for x in v]
            else:
                v = self.get_zeros_like(m)

            updates[m] = v

            if hvd.size() > 1:
                for x in updates[m]:
                    handles.append(hvd.allreduce_async_(x, op=hvd.Sum))
    
        for handle in handles:
            hvd.synchronize(handle)

        self._kl_clip_and_update_grad(updates, lr)

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

    def get_zeros_like(self, m):
        if m.bias is not None:
            v = [torch.zeros_like(m.weight.grad),
                 torch.zeros_like(m.bias.grad)]
        else:
            v = [torch.zeros_like(m.weight.grad)]
        return v

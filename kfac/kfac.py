import math

import torch
import torch.optim as optim

from kfac_utils import (ComputeCovA, ComputeCovG)
from kfac_utils import update_running_stat
from kfac_utils import try_contiguous
from kfac_utils import cycle
from kfac_utils import get_block_boundary

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
                 diag_blocks=1):
       
        defaults = dict(lr=lr, damping=damping)
        super(KFAC, self).__init__(model.parameters(), defaults)
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged

        self.known_modules = {'Linear', 'Conv2d'}

        self.modules = []

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
        self.diag_blocks = diag_blocks

        self.eps = 1e-10  # for numerical stability
        self.distribute_eigen_factors = False
        self.rank_iter = cycle(list(range(hvd.size())))

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            self.m_a[module] = input[0].data

    def _save_grad_output(self, module, grad_input, grad_output):
        if self.steps % self.TCov == 0:
            self.m_g[module] = grad_output[0].data

    def _prepare_model(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
        if hvd.size() > len(self.modules):
            self.distribute_eigen_factors = True

    def _init_a_buffers(self, factor, module):
        self.m_aa[module] = torch.diag(factor.new(factor.shape[0]).fill_(1))
        self.d_a[module] = factor.new_zeros(factor.shape[0])
        self.Q_a[module] = factor.new_zeros(factor.shape)

    def _init_g_buffers(self, factor, module):
        self.m_gg[module] = torch.diag(factor.new(factor.shape[0]).fill_(1))
        self.d_g[module] = factor.new_zeros(factor.shape[0])
        self.Q_g[module] = factor.new_zeros(factor.shape)

    def _update_cov_a(self):
        for module in self.modules: 
            aa = self.CovAHandler(self.m_a[module], module)
            if self.steps == 0:
                self._init_a_buffers(aa, module)
            update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _update_cov_g(self):
        for module in self.modules:
            gg = self.CovGHandler(self.m_g[module], module, self.batch_averaged)
            if self.steps == 0:
                self._init_g_buffers(gg, module)
            update_running_stat(gg, self.m_gg[module], self.stat_decay)

    def _update_eigen_a(self, m, ranks):
        i = ranks.index(hvd.rank())
        n = len(ranks)
        a = self.m_aa[m]
        if n > min(a.shape):
            n = min(a.shape)

        if i < n:
            start, end = get_block_boundary(i, n, a.shape)
            block = a[start[0]:end[0], start[1]:end[1]]
            d, Q = torch.symeig(block, eigenvectors=True)
            d = torch.mul(d, (d > self.eps).float())
            self.d_a[m].data[start[0]:end[0]].copy_(d)
            self.Q_a[m].data[start[0]:end[0], start[1]:end[1]].copy_(Q)

    def _update_eigen_g(self, m, ranks):
        i = ranks.index(hvd.rank())
        n = len(ranks)
        g = self.m_gg[m]
        if n > min(g.shape):
            n = min(g.shape)

        if i < n:
            start, end = get_block_boundary(i, n, g.shape)
            block = g[start[0]:end[0], start[1]:end[1]]
            d, Q = torch.symeig(block, eigenvectors=True)
            d = torch.mul(d, (d > self.eps).float())
            self.d_g[m].data[start[0]:end[0]].copy_(d)
            self.Q_g[m].data[start[0]:end[0], start[1]:end[1]].copy_(Q)

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
        if hvd.rank() == 0:
            print("m", m)
            print("m weight", m.weight.data.size())
            print("m weight grad", m.weight.grad.size())
            print("m aa", self.m_aa[m].size())
            print("m gg", self.m_gg[m].size())
            print("pgradmat", p_grad_mat.size())
            print("q_g", self.Q_g[m].size())
            print("q_a", self.Q_a[m].size())
            print("d_g", self.d_g[m].size())
            print("d_a", self.d_a[m].size())
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
        nu = min(1.0, math.sqrt(self.kl_clip / abs(vg_sum)))

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

        if self.steps % self.TCov == 0:
            self._update_cov_a()
            self._update_cov_g()
            if hvd.size() > 1:
                self.allreduce_covs()

        if self.steps % self.TInv == 0:
            # reset rank iter so device get the same layers
            # to compute to take advantage of caching
            self.rank_iter.reset() 
            for m in self.modules:
                ranks_a = self.rank_iter.next(self.diag_blocks)
                if self.distribute_eigen_factors:
                    ranks_g = self.rank_iter.next(self.diag_blocks)
                else:
                    ranks_g = ranks_a

                if hvd.rank() in ranks_a:
                    self._update_eigen_a(m, ranks_a)
                else:
                    self.Q_a[m].fill_(0)
                    self.d_a[m].fill_(0)

                if hvd.rank() in ranks_g:
                    self._update_eigen_g(m, ranks_g)
                else:
                    self.Q_g[m].fill_(0)
                    self.d_g[m].fill_(0)

            if hvd.size() > 1:
                self.allreduce_eigens()

        for m in self.modules:
            classname = m.__class__.__name__
            p_grad_mat = self._get_matrix_form_grad(m, classname)
            v = self._get_natural_grad(m, p_grad_mat, damping)
            updates[m] = v

        self._kl_clip_and_update_grad(updates, lr)

        self.steps += 1

    def allreduce_covs(self):
        handles = []

        for m in self.modules:
            handles.append(hvd.allreduce_async_(self.m_aa[m].data, op=hvd.Average))
            handles.append(hvd.allreduce_async_(self.m_gg[m].data, op=hvd.Average))

        for handle in handles:
            hvd.synchronize(handle)

    def allreduce_eigens(self):
        handles = []

        for m in self.modules:
            handles.append(hvd.allreduce_async_(self.Q_a[m].data, op=hvd.Sum))
            handles.append(hvd.allreduce_async_(self.Q_g[m].data, op=hvd.Sum))
            handles.append(hvd.allreduce_async_(self.d_a[m].data, op=hvd.Sum))
            handles.append(hvd.allreduce_async_(self.d_g[m].data, op=hvd.Sum))
    
        for handle in handles:
            hvd.synchronize(handle)

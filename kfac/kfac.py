import math

import torch
import torch.optim as optim

from kfac_utils import (ComputeA, ComputeG)
from kfac_utils import update_running_stat
from kfac_utils import try_contiguous
from kfac_utils import cycle
from kfac_utils import get_block_boundary

import horovod.torch as hvd

class KFAC(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.1,
                 factor_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 fac_update_freq=10,
                 inv_update_freq=100,
                 batch_averaged=True,
                 diag_blocks=1,
                 diag_warmup=0,
                 distribute_layer_factors=None):

        defaults = dict(lr=lr,
                        damping=damping,
                        fac_update_freq=fac_update_freq,
                        inv_update_freq=inv_update_freq) 

        super(KFAC, self).__init__(model.parameters(), defaults)

        self.computeA = ComputeA()
        self.computeG = ComputeG()
        self.known_modules = {'Linear', 'Conv2d'}
        self.modules = []
        self._register_modules(model)

        self.steps = 0

        self.m_a, self.m_g = {}, {}
        self.m_A, self.m_G = {}, {}
        self.m_QA, self.m_QG = {}, {}
        self.m_dA, self.m_dG = {}, {}

        self.factor_decay = factor_decay
        self.kl_clip = kl_clip
        self.fac_update_freq = fac_update_freq
        self.inv_update_freq = inv_update_freq
        self.diag_blocks = diag_blocks
        self.diag_warmup = diag_warmup
        self.batch_averaged = batch_averaged
        
        if distribute_layer_factors is None:
            self.distribute_layer_factors = True \
                    if hvd.size() > len(self.modules) else False
        else:
            self.distribute_layer_factors = distribute_layer_factors

        self.have_cleared_Q = True if self.diag_warmup == 0 else False
        self.eps = 1e-10  # for numerical stability
        self.rank_iter = cycle(list(range(hvd.size())))

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.fac_update_freq == 0:
            self.m_a[module] = input[0].data

    def _save_grad_output(self, module, grad_input, grad_output):
        if self.steps % self.fac_update_freq == 0:
            self.m_g[module] = grad_output[0].data

    def _register_modules(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)

    def _init_A(self, factor, module):
        self.m_A[module] = torch.diag(factor.new(factor.shape[0]).fill_(1))
        self.m_dA[module] = factor.new_zeros(factor.shape[0])
        self.m_QA[module] = factor.new_zeros(factor.shape)

    def _init_G(self, factor, module):
        self.m_G[module] = torch.diag(factor.new(factor.shape[0]).fill_(1))
        self.m_dG[module] = factor.new_zeros(factor.shape[0])
        self.m_QG[module] = factor.new_zeros(factor.shape)

    def _update_A(self):
        for module in self.modules: 
            a = self.computeA(self.m_a[module], module)
            if self.steps == 0:
                self._init_A(a, module)
            update_running_stat(a, self.m_A[module], self.factor_decay)

    def _update_G(self):
        for module in self.modules:
            g = self.computeG(self.m_g[module], module, self.batch_averaged)
            if self.steps == 0:
                self._init_G(g, module)
            update_running_stat(g, self.m_G[module], self.factor_decay)

    def _update_eigen_A(self, module, ranks):
        i = ranks.index(hvd.rank())
        n = len(ranks)
        a = self.m_A[module]
        if n > min(a.shape):
            n = min(a.shape)

        if i < n:
            start, end = get_block_boundary(i, n, a.shape)
            block = a[start[0]:end[0], start[1]:end[1]]
            d, Q = torch.symeig(block, eigenvectors=True)
            d = torch.mul(d, (d > self.eps).float())
            self.m_dA[module].data[start[0]:end[0]].copy_(d)
            self.m_QA[module].data[start[0]:end[0], start[1]:end[1]].copy_(Q)

    def _update_eigen_G(self, module, ranks):
        # Only compute g on first rank to enter this function
        i = ranks.index(hvd.rank())
        n = len(ranks)
        g = self.m_G[module]
        if n > min(g.shape):
            n = min(g.shape)

        if i < n:
            start, end = get_block_boundary(i, n, g.shape)
            block = g[start[0]:end[0], start[1]:end[1]]
            d, Q = torch.symeig(block, eigenvectors=True)
            d = torch.mul(d, (d > self.eps).float())
            self.m_dG[module].data[start[0]:end[0]].copy_(d)
            self.m_QG[module].data[start[0]:end[0], start[1]:end[1]].copy_(Q)

    def _get_grad(self, module, classname):
        """
        :param m: the layer
        :param classname: the class name of the layer
        :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
        """
        if classname == 'Conv2d':
            # n_filters * (in_c * kw * kh)
            grad = module.weight.grad.data.view(module.weight.grad.data.size(0), -1)  
        else:
            grad = module.weight.grad.data
        if m.bias is not None:
            grad = torch.cat([grad, module.bias.grad.data.view(-1, 1)], 1)
        return grad

    def _get_preconditioned_grad(self, module, grad):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        v1 = self.m_QG[module].t() @ grad @ self.m_QA[module]
        v2 = v1 / (self.m_dG[module].unsqueeze(1) * self.m_dA[module].unsqueeze(0) + 
                   self.damping)
        v = self.m_QG[module] @ v2 @ self.m_QA[module].t()

        if module.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(module.weight.grad.data.size())
            v[1] = v[1].view(module.bias.grad.data.size())
        else:
            v = [v.view(module.weight.grad.data.size())]
        return v

    def _update_scale_grad(self, updates):
        # do kl clip
        vg_sum = 0
        for module in self.modules:
            v = updates[module]
            vg_sum += (v[0] * module.weight.grad.data * self.lr ** 2).sum().item()
            if module.bias is not None:
                vg_sum += (v[1] * module.bias.grad.data * self.lr ** 2).sum().item()
        nu = min(1.0, math.sqrt(self.kl_clip / abs(vg_sum)))

        for module in self.modules:
            v = updates[module]
            module.weight.grad.data.copy_(v[0])
            module.weight.grad.data.mul_(nu)
            if module.bias is not None:
                module.bias.grad.data.copy_(v[1])
                module.bias.grad.data.mul_(nu)

    def step(self, closure=None, epoch=None):
        group = self.param_groups[0]
        self.lr = group['lr']
        self.damping = group['damping']
        self.fac_update_freq = group['fac_update_freq']
        self.inv_update_freq = group['inv_update_freq']

        updates = {}
        handles = []

        if epoch is None:
            if self.diag_warmup > 0:
                print("WARNING: diag_warmup > 0 but epoch was not passed to "
                      "KFAC.step(). Defaulting to no diag_warmup")
            diag_blocks = self.diag_blocks
        else:
            diag_blocks = self.diag_blocks if epoch >= self.diag_warmup else 1

        if self.steps % self.fac_update_freq == 0:
            self._update_A()
            self._update_G()
            if hvd.size() > 1:
                self._allreduce_factors()

        # if we are switching from no approx to approx, we need to clear
        # off-block-diagonal elements
        if not self.have_cleared_Q and \
                epoch == self.diag_warmup and \
                self.steps % self.inv_update_freq == 0:
            for m in self.modules:
                self.m_QA[m].fill_(0)
                self.m_QG[m].fill_(0)
            self.have_cleared_Q = True

        if self.steps % self.inv_update_freq == 0:
            # reset rank iter so device get the same layers
            # to compute to take advantage of caching
            self.rank_iter.reset() 
            for module in self.modules:
                n = diag_blocks if module.__class__.__name__ == 'Conv2d' else 1
                ranks_a = self.rank_iter.next(n)
                if self.distribute_layer_factors:
                    ranks_g = self.rank_iter.next(n)
                else:
                    ranks_g = ranks_a

                if hvd.rank() in ranks_a:
                    self._update_eigen_a(module, ranks_a)
                else:
                    self.m_QA[module].fill_(0)
                    self.m_dA[module].fill_(0)

                if hvd.rank() in ranks_g:
                    self._update_eigen_g(module, ranks_g)
                else:
                    self.m_QG[module].fill_(0)
                    self.m_dG[module].fill_(0)

            if hvd.size() > 1:
                self._allreduce_eigendecomp()

        for module in self.modules:
            classname = module.__class__.__name__
            grad = self._get_grad(module, classname)
            precon_grad = self._get_preconditioned_grad(module, grad)
            updates[module] = precon_grad

        self._update_scale_grad(updates)

        self.steps += 1

    def _allreduce_factors(self):
        handles = []

        for m in self.modules:
            handles.append(hvd.allreduce_async_(self.m_A[m].data, op=hvd.Average))
            handles.append(hvd.allreduce_async_(self.m_G[m].data, op=hvd.Average))

        for handle in handles:
            hvd.synchronize(handle)

    def _allreduce_eigendecomp(self):
        handles = []

        for m in self.modules:
            handles.append(hvd.allreduce_async_(self.m_QA[m].data, op=hvd.Sum))
            handles.append(hvd.allreduce_async_(self.m_QG[m].data, op=hvd.Sum))
            handles.append(hvd.allreduce_async_(self.m_dA[m].data, op=hvd.Sum))
            handles.append(hvd.allreduce_async_(self.m_dG[m].data, op=hvd.Sum))
    
        for handle in handles:
            hvd.synchronize(handle)


class KFACParamScheduler():
    def __init__(self,
                 kfac,
                 damping_alpha=1,
                 damping_schedule=None,
                 update_frem_QAlpha=1,
                 update_freq_schedule=None,
                 start_epoch=0):

        self.kfac = kfac
        params = self.kfac.param_groups[0]

        self.damping_base = params['damping']
        self.damping_alpha = damping_alpha
        self.damping_schedule = damping_schedule
        self.damping_factor_func = \
                self.get_factor_func(self.damping_schedule,
                                     self.damping_alpha)

        self.fac_update_freq_base = params['fac_update_freq']
        self.inv_update_freq_base = params['inv_update_freq']
        self.update_frem_QAlpha = update_frem_QAlpha
        self.update_freq_schedule = update_freq_schedule
        self.update_freq_factor_func = \
                self.get_factor_func(self.update_freq_schedule,
                                     self.update_frem_QAlpha)

        self.epoch = start_epoch

    def get_factor_func(self, schedule, alpha):
        if schedule is not None:
            schedule.sort(reverse=True)
        else:
            schedule = []

        def factor_func(epoch):
            factor = 1.
            for e in schedule:
                if epoch >= e:
                    factor *= alpha
            return factor

        return factor_func

    def step(self, epoch=None):
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch += 1

        params = self.kfac.param_groups[0]

        params['damping'] = self.damping_base * self.damping_factor_func(self.epoch)

        factor = self.update_freq_factor_func(self.epoch)
        params['fac_update_freq'] = int(self.fac_update_freq_base * factor)
        params['inv_update_freq'] = int(self.inv_update_freq_base * factor)

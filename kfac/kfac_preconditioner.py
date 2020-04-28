import math
import torch
import torch.optim as optim
import horovod.torch as hvd

from kfac.utils import (ComputeA, ComputeG)
from kfac.utils import update_running_avg
from kfac.utils import try_contiguous
from kfac.utils import cycle
from kfac.utils import get_block_boundary

class KFAC(optim.Optimizer):
    """KFAC Distributed Gradient Preconditioner

    Computes the natural gradient of a model in place with a layer-wise
    FIM approximation. Layer computations are distributed across workers
    using Horovod.

    Usage:
      optimizer = optim.SGD(model.parameters(), ...)
      optimizer = hvd.DistributedOptimizer(optimizer, ...)
      preconditioner = KFAC(model, ...)
      ... 
      for i, (data, target) in enumerate(train_loader):
          optimizer.zero_grad()
          output = model(data)
          loss = criterion(output, target)
          loss.backward()
          optimizer.synchronize()
          preconditioner.step()
          with optimizer.skip_synchronize():
              optimizer.step()

    Args:
      model (nn): Torch model to precondition
      lr (float, optional): learning rate (default: 0.1)
      factor_decay (float, optional): running average coefficient for Kronecker
          factors (default: 0.95)
      damping (float, optional): Tikhonov damping parameter (default: 0.001)
      kl_clip (float, optional): clipping parameter for gradient scaling
          (default: 0.001)
      fac_update_freq (int, optional): iterations between calculating and
          updating the running average of the Kronecker factors (default: 10)
      kfac_update_freq (int, optional): iterations between applying gradient
          preconditioning (default: 100)
      batch_averaged (bool, optional): boolean representing if the gradient
          is alrady averaged across the batches (default: True)
      diag_blocks (int, optional): Experimental: number of diagonal blocks to
          approximate the Kronecker factor eigendecomposition with. 
          `diag_blocks=1` computes the eigendecomposition of the entire factor
          (default: 1)
      diag_warmup (int, optional): number of epochs to wait before starting
          the block diagonal factor approximation (default: 0)
      distribute_layer_factors (bool, optional): if `True`, computes factors A
          and G on different workers else computes A and G for a single layer
          on the same worker. If `None`, determines best value based on layer
          count (default: None)
    """
    def __init__(self,
                 model,
                 lr=0.1,
                 factor_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 fac_update_freq=10,
                 kfac_update_freq=100,
                 batch_averaged=True,
                 diag_blocks=1,
                 diag_warmup=0,
                 distribute_layer_factors=None):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < factor_decay <= 1:
            raise ValueError("Invalid factor decay rate: {}".format(factor_decay))
        if not 0.0 < damping:
            raise ValueError("Invalid damping: {}".format(damping))
        if not 0.0 < kl_clip:
            raise ValueError("Invalid clipping value: {}".format(kl_clip))
        if not 0 < fac_update_freq:
            raise ValueError("Invalid factor update frequency: {}".format(fac_update_freq))
        if not 0 < kfac_update_freq:
            raise ValueError("Invalid K-FAC update frequency: {}".format(kfac_update_freq))
        if not 0 == kfac_update_freq % fac_update_freq:
            print("WARNING: it is suggested that kfac_update_freq be a multiple of fac_update_freq")
        if not 0 < diag_blocks:
            raise ValueError("Invalid diagonal block approx count: {}".format(diag_blocks))
        if not 0 <= diag_blocks:
            raise ValueError("Invalid diagonal block approx count: {}".format(diag_blocks))
        if not 1 == diag_blocks:
            print("WARNING: diag_blocks > 1 is experimental and may give poor results.")

        # For compatibility with `KFACParamScheduler`
        defaults = dict(lr=lr,
                        damping=damping,
                        fac_update_freq=fac_update_freq,
                        kfac_update_freq=kfac_update_freq) 

        super(KFAC, self).__init__(model.parameters(), defaults)

        self.computeA = ComputeA()
        self.computeG = ComputeG()
        self.known_modules = {'Linear', 'Conv2d'}
        self.modules = []
        self._register_modules(model)

        self.steps = 0

        # Dictionaries keyed by `module` to storing the factors and
        # eigendecompositions
        self.m_a, self.m_g = {}, {}
        self.m_A, self.m_G = {}, {}
        self.m_QA, self.m_QG = {}, {}
        self.m_dA, self.m_dG = {}, {}

        self.factor_decay = factor_decay
        self.kl_clip = kl_clip
        self.fac_update_freq = fac_update_freq
        self.kfac_update_freq = kfac_update_freq
        self.diag_blocks = diag_blocks
        self.diag_warmup = diag_warmup
        self.batch_averaged = batch_averaged
        
        # Compute ideal value for `distribute_layer_factors` based on
        # registered module count
        if distribute_layer_factors is None:
            self.distribute_layer_factors = True \
                    if hvd.size() > len(self.modules) else False
        else:
            self.distribute_layer_factors = distribute_layer_factors

        self.have_cleared_Q = True if self.diag_warmup == 0 else False
        self.eps = 1e-10  # for numerical stability
        self.rank_iter = cycle(list(range(hvd.size())))

    def _save_input(self, module, input):
        """Hook for saving layer input"""
        if torch.is_grad_enabled() and self.steps % self.fac_update_freq == 0:
            self.m_a[module] = input[0].data

    def _save_grad_output(self, module, grad_input, grad_output):
        """Hook for saving gradient w.r.t output"""
        if self.steps % self.fac_update_freq == 0:
            self.m_g[module] = grad_output[0].data

    def _register_modules(self, model):
        """Register hooks to all supported layers in the model"""
        for module in model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)

    def _init_A(self, factor, module):
        """Initialize memory for factor A and its eigendecomp"""
        self.m_A[module] = torch.diag(factor.new(factor.shape[0]).fill_(1))
        self.m_dA[module] = factor.new_zeros(factor.shape[0])
        self.m_QA[module] = factor.new_zeros(factor.shape)

    def _init_G(self, factor, module):
        """Initialize memory for factor G and its eigendecomp"""
        self.m_G[module] = torch.diag(factor.new(factor.shape[0]).fill_(1))
        self.m_dG[module] = factor.new_zeros(factor.shape[0])
        self.m_QG[module] = factor.new_zeros(factor.shape)

    def _clear_eigen(self):
        """Clear eigendecompositions

        Useful for when switching between `diag_blocks=1` and `diag-blocks>1`
        because eigendecompositions saved in place and the off-diagonals must
        be cleared.
        """
        for module in self.modules:
            self.m_QA[module].fill_(0)
            self.m_QG[module].fill_(0)
            self.m_dA[module].fill_(0)
            self.m_dG[module].fill_(0)

    def _update_A(self):
        """Compute and update factor A for all modules"""
        for module in self.modules: 
            a = self.computeA(self.m_a[module], module)
            if self.steps == 0:
                self._init_A(a, module)
            update_running_avg(a, self.m_A[module], self.factor_decay)

    def _update_G(self):
        """Compute and update factor G for all modules"""
        for module in self.modules:
            g = self.computeG(self.m_g[module], module, self.batch_averaged)
            if self.steps == 0:
                self._init_G(g, module)
            update_running_avg(g, self.m_G[module], self.factor_decay)

    def _update_eigen_A(self, module, ranks):
        """Compute eigendecomposition of A for module on specified workers

        Note: all ranks will enter this function but only the ranks specified
        in `ranks` will continue to actually compute the eigendecomposition.
        All other ranks will simply zero out their buffer for the 
        eigendecomposition for the current module. This is done so we can sum
        the eigendecompositions across all ranks to communicate the results
        of locally computed eigendecompositions.

        Args:
          module: module to compute eigendecomposition of A on
          ranks: list of horovod ranks (i.e. workers) to use when computing
              the eigendecomposition.
        """
        if hvd.rank() in ranks:
            self._distributed_compute_eigen(self.m_A[module], 
                    self.m_QA[module], self.m_dA[module], ranks)
        else:
            self.m_QA[module].fill_(0)
            self.m_dA[module].fill_(0)

    def _update_eigen_G(self, module, ranks):
        """Compute eigendecomposition of A for module on specified workers

        See `_update_eigen_A` for more info`
        """
        if hvd.rank() in ranks:
            self._distributed_compute_eigen(self.m_G[module], 
                    self.m_QG[module], self.m_dG[module], ranks)
        else:
            self.m_QG[module].fill_(0)
            self.m_dG[module].fill_(0)

    def _distributed_compute_eigen(self, factor, evectors, evalues, ranks):
        """Computes the eigendecomposition of a factor across ranks
        
        Assigns each rank in `ranks` to enter this function to compute a
        diagonal block of `factor`. Results are written to `evectors` and
        `evalues`. If `len(ranks)==1`, then that rank computes the
        eigendecomposition of the entire `factor`.

        Args:
            factor (tensor): tensor to eigendecompose
            evectors (tensor): tensor to save eigenvectors of `factor` to
            evalues (tensor): tensor to save eigenvalues of `factor` to
            ranks (list): list of ranks that will enter this function
        """
        i = ranks.index(hvd.rank())
        n = len(ranks)
        if n > min(factor.shape):
            n = min(factor.shape)

        if i < n:
            start, end = get_block_boundary(i, n, factor.shape)
            block = factor[start[0]:end[0], start[1]:end[1]]
            d, Q = torch.symeig(block, eigenvectors=True)
            d = torch.mul(d, (d > self.eps).float())
            evalues.data[start[0]:end[0]].copy_(d)
            evectors.data[start[0]:end[0], start[1]:end[1]].copy_(Q)

    def _get_diag_blocks(self, module, diag_blocks):
        """Helper method for determining number of diag_blocks to use

        Overrides `diag_blocks` if the `module` does not support
        `diag_blocks>1`. I.e. for a Linear layer, we do not want to
        use a `diag_blocks>1`.

        Args:
          module: module
          diag_blocks (int): default number of diag blocks to use
        """
        return diag_blocks if module.__class__.__name__ == 'Conv2d' else 1

    def _get_grad(self, module):
        """Get formated gradient of module

        Args:
          module: module/layer to get gradient of

        Returns:
          Formatted gradient with shape [output_dim, input_dim] for module
        """
        if module.__class__.__name__ == 'Conv2d':
            # n_filters * (in_c * kw * kh)
            grad = module.weight.grad.data.view(module.weight.grad.data.size(0), -1)  
        else:
            grad = module.weight.grad.data
        if module.bias is not None:
            grad = torch.cat([grad, module.bias.grad.data.view(-1, 1)], 1)
        return grad

    def _get_preconditioned_grad(self, module, grad):
        """Precondition gradient of module
        
        Args:
          module: module to compute preconditioned gradient for
          grad: formatted gradient from `_get_grad()`

        Returns:
          preconditioned gradient with same shape as `grad`
        """
        v1 = self.m_QG[module].t() @ grad @ self.m_QA[module]
        v2 = v1 / (self.m_dG[module].unsqueeze(1) * self.m_dA[module].unsqueeze(0) + 
                   self.damping)
        v = self.m_QG[module] @ v2 @ self.m_QA[module].t()

        if module.bias is not None:
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(module.weight.grad.data.size()) # weight
            v[1] = v[1].view(module.bias.grad.data.size())   # bias
        else:
            v = [v.view(module.weight.grad.data.size())]
        return v

    def _update_scale_grad(self, updates):
        """Update the gradients in place and scale

        Updates the gradients in-place for all modules using the preconditioned
        gradients and scales the gradients.

        Args:
          updates (dict): dict of {module: precon_grad}
        """
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
        """Perform one K-FAC step

        Note:
        - this function should always be called before `optimizer.step()`
        - gradients must be averaged across ranks before calling `step()`

        Args:
          closure: for compatibility with the base optimizer class.
              `closure` is ignored by KFAC
          epoch (int, optional): epoch to use for determining when to end
              the `diag_warmup` period. `epoch` is not necessary if not using
              `diag_warmup`
        """

        # Update params, used for compatibilty with `KFACParamScheduler`
        group = self.param_groups[0]
        self.lr = group['lr']
        self.damping = group['damping']
        self.fac_update_freq = group['fac_update_freq']
        self.kfac_update_freq = group['kfac_update_freq']

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

        # if we are switching from no diag approx to approx, we need to clear
        # off-block-diagonal elements
        if not self.have_cleared_Q and \
                epoch == self.diag_warmup and \
                self.steps % self.kfac_update_freq == 0:
            self._clear_eigen()
            self.have_cleared_Q = True

        if self.steps % self.kfac_update_freq == 0:
            # reset rank iter so device get the same layers
            # to compute to take advantage of caching
            self.rank_iter.reset() 

            for module in self.modules:
                # Get ranks to compute this layer on
                n = self._get_diag_blocks(module, diag_blocks)
                ranks_a = self.rank_iter.next(n)
                ranks_g = self.rank_iter.next(n) if self.distribute_layer_factors \
                                                 else ranks_a

                self._update_eigen_A(module, ranks_a)
                self._update_eigen_G(module, ranks_g)

            if hvd.size() > 1:
                self._allreduce_eigendecomp()

        for module in self.modules:
            grad = self._get_grad(module)
            precon_grad = self._get_preconditioned_grad(module, grad)
            updates[module] = precon_grad

        self._update_scale_grad(updates)

        self.steps += 1

    def _allreduce_factors(self):
        """Allreduce the factors for all layers"""
        handles = []

        for m in self.modules:
            handles.append(hvd.allreduce_async_(self.m_A[m].data, op=hvd.Average))
            handles.append(hvd.allreduce_async_(self.m_G[m].data, op=hvd.Average))

        for handle in handles:
            hvd.synchronize(handle)

    def _allreduce_eigendecomp(self):
        """Allreduce the eigendecompositions for all layers

        Note: we use `op=hvd.Sum` to simulate an allgather`. Each rank will
        either compute the eigendecomposition for a factor or just return
        zeros so we sum instead of averaging.
        """
        handles = []

        for m in self.modules:
            handles.append(hvd.allreduce_async_(self.m_QA[m].data, op=hvd.Sum))
            handles.append(hvd.allreduce_async_(self.m_QG[m].data, op=hvd.Sum))
            handles.append(hvd.allreduce_async_(self.m_dA[m].data, op=hvd.Sum))
            handles.append(hvd.allreduce_async_(self.m_dG[m].data, op=hvd.Sum))
    
        for handle in handles:
            hvd.synchronize(handle)


class KFACParamScheduler():
    """Updates KFAC parameters according to the epoch

    Similar to `torch.optim.lr_scheduler.StepLR()`

    Usage:
      Call KFACParamScheduler.step() each epoch to compute new parameter
      values.

    Args:
      kfac (KFAC): wrapped KFAC preconditioner
      damping_alpha (float, optional): multiplicative factor of the damping 
          (default: 1)
      damping_schedule (list, optional): list of epochs to update the damping
          by `damping_alpha` (default: None)
      update_freq_alpha (float, optional): multiplicative factor of the KFAC
          update freq (default: 1)
      update_freq_schedule (list, optional): list of epochs to update the KFAC
          update freq by `update_freq_alpha` (default: None)
      start_epoch (int, optional): starting epoch, for use if resuming training
          from checkpoint (default: 0)
    """
    def __init__(self,
                 kfac,
                 damping_alpha=1,
                 damping_schedule=None,
                 update_freq_alpha=1,
                 update_freq_schedule=None,
                 start_epoch=0):

        self.kfac = kfac
        params = self.kfac.param_groups[0]

        self.damping_base = params['damping']
        self.damping_alpha = damping_alpha
        self.damping_schedule = damping_schedule
        self.damping_factor_func = \
                self._get_factor_func(self.damping_schedule,
                                     self.damping_alpha)

        self.fac_update_freq_base = params['fac_update_freq']
        self.kfac_update_freq_base = params['kfac_update_freq']
        self.update_freq_alpha = update_freq_alpha
        self.update_freq_schedule = update_freq_schedule
        self.update_freq_factor_func = \
                self._get_factor_func(self.update_freq_schedule,
                                     self.update_freq_alpha)

        self.epoch = start_epoch

    def _get_factor_func(self, schedule, alpha):
        """Returns a function to compute an update factor using the epoch"""
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
        """Update KFAC parameters"""
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch += 1

        params = self.kfac.param_groups[0]

        params['damping'] = self.damping_base * self.damping_factor_func(self.epoch)

        factor = self.update_freq_factor_func(self.epoch)
        params['fac_update_freq'] = int(self.fac_update_freq_base * factor)
        params['kfac_update_freq'] = int(self.kfac_update_freq_base * factor)

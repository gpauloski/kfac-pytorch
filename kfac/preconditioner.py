import math
import torch
import torch.optim as optim
import horovod.torch as hvd

from kfac.layers import get_kfac_layer
from kfac.layers import KNOWN_MODULES
from kfac.utils import cycle

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
      use_eigen_decomp (bool, optional): use the eigendecomposition method for
          the KFAC update, otherwise use normal inv method (default: True)
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
                 use_eigen_decomp=True,
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

        self.steps = 0

        self.lr = lr
        self.damping = damping
        self.fac_update_freq = fac_update_freq
        self.kfac_update_freq = kfac_update_freq
        self.factor_decay = factor_decay
        self.kl_clip = kl_clip
        self.fac_update_freq = fac_update_freq
        self.kfac_update_freq = kfac_update_freq
        self.use_eigen_decomp = use_eigen_decomp
        self.diag_blocks = diag_blocks
        self.diag_warmup = diag_warmup
        self.batch_averaged = batch_averaged
        
        self.layers = []
        self._register_layers(model)

        # Compute ideal value for `distribute_layer_factors` based on
        # registered module count
        if distribute_layer_factors is None:
            self.distribute_layer_factors = True \
                    if hvd.size() > len(self.modules) else False
        else:
            self.distribute_layer_factors = distribute_layer_factors

        self.have_cleared_Q = True if self.diag_warmup == 0 else False
        self.rank_iter = cycle(list(range(hvd.size())))

    def _get_input_hook(self, kfac_layer):
        def _save_input(module, input):
            """Hook for saving layer input"""
            if (torch.is_grad_enabled() and 
                self.steps % self.fac_update_freq == 0):
                kfac_layer.save_input(input)
        return _save_input

    def _get_output_hook(self, kfac_layer):
        def _save_grad_output(module, grad_input, grad_output):
            """Hook for saving gradient w.r.t output"""
            if self.steps % self.fac_update_freq == 0:
                kfac_layer.save_grad_output(grad_output)
        return _save_grad_output

    def _register_layers(self, model):
        """Register hooks to all supported layers in the model"""
        for module in model.modules():
            if module.__class__.__name__ not in KNOWN_MODULES:
                continue
            kfac_layer = get_kfac_layer(module, self.use_eigen_decomp,
                    self.damping, self.factor_decay, self.batch_averaged)
            self.layers.append(kfac_layer)
            module.register_forward_pre_hook(self._get_input_hook(kfac_layer))
            module.register_backward_hook(self._get_output_hook(kfac_layer))

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

    def _update_scale_grad(self, updates):
        """Update the gradients in place and scale

        Updates the gradients in-place for all modules using the preconditioned
        gradients and scales the gradients.

        Args:
          updates (dict): dict of {module: precon_grad}
        """
        vg_sum = 0
        for layer in self.layers:
            module = layer.module
            v = updates[module]
            vg_sum += (v[0] * module.weight.grad.data * self.lr ** 2).sum().item()
            if module.bias is not None:
                vg_sum += (v[1] * module.bias.grad.data * self.lr ** 2).sum().item()
        nu = min(1.0, math.sqrt(self.kl_clip / abs(vg_sum)))

        for layer in self.layers:
            module = layer.module
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
            for layer in self.layers:
                layer.update_A()
                layer.update_G()
            if hvd.size() > 1:
                self._allreduce_factors()

        # if we are switching from no diag approx to approx, we need to clear
        # off-block-diagonal elements
        if not self.have_cleared_Q and \
                epoch == self.diag_warmup and \
                self.steps % self.kfac_update_freq == 0:
            for layer in self.layers:
                layer.clear_inverse()
            self.have_cleared_Q = True

        if self.steps % self.kfac_update_freq == 0:
            # reset rank iter so device get the same layers
            # to compute to take advantage of caching
            self.rank_iter.reset() 

            for layer in self.layers:
                # Get ranks to compute this layer on
                n = self._get_diag_blocks(layer.module, diag_blocks)
                ranks_a = self.rank_iter.next(n)
                ranks_g = self.rank_iter.next(n) if self.distribute_layer_factors \
                                                 else ranks_a

                layer.compute_A_inv(ranks_a)
                layer.compute_G_inv(ranks_g)

            if hvd.size() > 1:
                self._allreduce_inverses()

        for layer in self.layers:
            # TODO: make layer hashable so we don't need to use module as key
            updates[layer.module] = layer.get_preconditioned_gradient()

        self._update_scale_grad(updates)

        self.steps += 1

    def _allreduce_factors(self):
        """Allreduce the factors for all layers"""
        handles = []

        for layer in self.layers:
            handles.extend(layer.get_factor_handles())

        for handle in handles:
            hvd.synchronize(handle)

    def _allreduce_inverses(self):
        """Allreduce the eigendecomp/invs for all layers"""
        handles = []

        for layer in self.layers:
            handles.extend(layer.get_inverse_handles())
   
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

import math
import torch
import torch.optim as optim
import horovod.torch as hvd

from .layers import *
from .utils import load_balance

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
        
        self.layers = {}  # key: nn.Module, value: KFACLayer
        self._register_layers(model)

        # Compute ideal value for `distribute_layer_factors` based on
        # registered module count
        if distribute_layer_factors is None:
            self.distribute_layer_factors = True \
                    if hvd.size() > len(self.layers) else False
        else:
            self.distribute_layer_factors = distribute_layer_factors

        self.have_cleared_Q = True if self.diag_warmup == 0 else False

    def _save_input(self, module, input):
        """Hook for saving layer input"""
        if (torch.is_grad_enabled() and 
                self.steps % self.fac_update_freq == 0):
            self.layers[module].save_inputs(input)

    def _save_grad_output(self, module, grad_input, grad_output):
        """Hook for saving gradient w.r.t output"""
        if self.steps % self.fac_update_freq == 0:
            self.layers[module].save_grad_outputs(grad_output)

    def _register_layers(self, model):
        """Register hooks to all supported layers in the model"""
        for module in model.modules():
            if module.__class__.__name__ not in KNOWN_MODULES:
                continue
            if not module_requires_grad(module):
                continue
            kfac_layer = get_kfac_layer(module, self.use_eigen_decomp,
                    self.damping, self.factor_decay, self.batch_averaged)
            if hvd.rank() == 0:
                print('Registered layer {}'.format(module.__class__.__name__))
            self.layers[module] = kfac_layer
            module.register_forward_pre_hook(self._save_input)
            module.register_backward_hook(self._save_grad_output)

    def _compute_grad_scale(self):
        """Computes scale factor for preconditioned gradients

        Returns:
          sum_{layers} (sum_{gradients} precon_grad * grad * lr^2) 
        """
        vg_sum = 0.
        for module, layer in self.layers.items():
            for i, v in enumerate(layer.preconditioned_gradients):
                vg_sum += (v[0] * layer._get_weight(i).grad.data *
                           self.lr ** 2).sum().item()
                if layer.has_bias:
                    vg_sum += (v[1] * layer._get_bias(i).grad.data * 
                               self.lr ** 2).sum().item()
        return min(1.0, math.sqrt(self.kl_clip / abs(vg_sum)))

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

        # if we are switching from no diag approx to approx, we need to clear
        # off-block-diagonal elements
        if not self.have_cleared_Q and \
                epoch == self.diag_warmup and \
                self.steps % self.kfac_update_freq == 0:
            for layer in self.layers.values():
                layer.clear_inverses()
            self.have_cleared_Q = True

        if self.steps % self.fac_update_freq == 0:
            for layer in self.layers.values():
                layer.update_A_factors()
                layer.update_G_factors()
            if hvd.size() > 1:
                self._allreduce_factors()

        # We do this after layer.update_*_factors because the inverse buffers
        # are not instantiate until this point and we use the size of the
        # buffers to approximate the time each layer will take to compute.
        if self.steps == 0:
            self._assign_layers_to_workers()

        if self.steps % self.kfac_update_freq == 0:
            for layer in self.layers.values():
                layer.compute_A_invs()
                layer.compute_G_invs()
            if hvd.size() > 1:
                self._allreduce_inverses()

        for layer in self.layers.values():
            layer.compute_preconditioned_gradients()

        nu = self._compute_grad_scale()

        for layer in self.layers.values():
            layer.update_gradients(nu)

        self.steps += 1

    def _allreduce_factors(self):
        """Allreduce the factors for all layers"""
        handles = []

        for layer in self.layers.values():
            handles.extend(layer.get_factor_handles())

        for handle in handles:
            hvd.synchronize(handle)

    def _allreduce_inverses(self):
        """Allreduce the eigendecomp/invs for all layers"""
        handles = []

        for layer in self.layers.values():
            handles.extend(layer.get_inverse_handles())
   
        for handle in handles:
            hvd.synchronize(handle)

    def _assign_layers_to_workers(self):
        """Assigns layers to workers to minimize max load on any worker"""
        func = lambda n: n**2.4  # approx inverse complexity
        a_sizes = [[x.shape[0] for x in l.A_invs] for l in self.layers.values()]
        g_sizes = [[x.shape[0] for x in l.G_invs] for l in self.layers.values()]
        a_times = [sum(map(func, sizes)) for sizes in a_sizes]
        g_times = [sum(map(func, sizes)) for sizes in a_sizes]
            
        if self.distribute_layer_factors:
            times = a_times + g_times
            locs = load_balance(hvd.size(), times)
            a_locs, g_locs = locs[0:len(a_times)], locs[len(a_times):]
        else:
            times = [a + g for a, g in zip(a_times, g_times)]
            locs = load_balance(hvd.size(), times)
            a_locs, g_locs = locs, locs

        for i, layer in enumerate(self.layers.values()):
            layer.A_ranks = [a_locs[i]]
            layer.G_ranks = [g_locs[i]]

        # TODO(gpauloski): remove before merging to master
        # used for testing the load balancing reduces max worker assignments
        #load = [0] * hvd.size()
        #for a, g, layer in zip(a_locs, g_locs, self.layers.values()):
        #    load[a] += layer.A_factors[0].nelement() 
        #    load[g] += layer.G_factors[0].nelement() 
        #if hvd.rank() == 0:
        #    for i, load in enumerate(load):
        #        print(i, load)

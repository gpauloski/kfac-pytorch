import math
import torch
import torch.optim as optim

from . import layers as kfac_layers
from . import utils

class KFAC(optim.Optimizer):
    """KFAC Distributed Gradient Preconditioner

    Computes the natural gradient of a model in place with a layer-wise
    FIM approximation. Layer computations are distributed across workers
    using Horovod or torch.Distributed.

    Horovod usage example:
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
      kl_clip (float, optional): clipping parameter for gradient scaling. If
          None, no scaling/clipping will be applied. (default: 0.001)
      fac_update_freq (int, optional): iterations between calculating and
          updating the running average of the Kronecker factors (default: 10)
      kfac_update_freq (int, optional): iterations between applying gradient
          preconditioning (default: 100)
      use_eigen_decomp (bool, optional): use the eigendecomposition method for
          the KFAC update, otherwise use normal inv method (default: True)
      batch_averaged (bool, optional): boolean representing if the gradient
          is alrady averaged across the batches (default: True)
      distribute_layer_factors (bool, optional): if `True`, computes factors A
          and G on different workers else computes A and G for a single layer
          on the same worker. For small worker counts, computing per layer
          factors on the same device can yeild improvements. (default: True)
      skip_layers (str or list, optional): name or list of names of modules to
          ignore when registering layers (default: None)
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
                 distribute_layer_factors=True,
                 skip_layers=None):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < factor_decay <= 1:
            raise ValueError("Invalid factor decay rate: {}".format(factor_decay))
        if not 0.0 < damping:
            raise ValueError("Invalid damping: {}".format(damping))
        if kl_clip is not None and not 0.0 < kl_clip:
            raise ValueError("Invalid clipping value: {}".format(kl_clip))
        if not 0 < fac_update_freq:
            raise ValueError("Invalid factor update frequency: {}".format(fac_update_freq))
        if not 0 < kfac_update_freq:
            raise ValueError("Invalid K-FAC update frequency: {}".format(kfac_update_freq))
        if not 0 == kfac_update_freq % fac_update_freq:
            print("WARNING: it is suggested that kfac_update_freq be a multiple of fac_update_freq")

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
        self.batch_averaged = batch_averaged
        self.distribute_layer_factors = distribute_layer_factors

        self.backend = utils.get_comm_backend()

        self.known_modules = {m.lower() for m in kfac_layers.KNOWN_MODULES}
        if isinstance(skip_layers, str):
            self.known_modules.discard(skip_layers.lower())
        elif isinstance(skip_layers, list):
            for layer in skip_layers: 
                self.known_modules.discard(layer.lower())

        self.layers = {}  # key: nn.Module, value: KFACLayer
        self._register_modules(model)

    def _save_input(self, module, input):
        """Hook for saving layer input"""
        if (torch.is_grad_enabled() and 
                self.steps % self.fac_update_freq == 0):
            self.layers[module].save_inputs(input)

    def _save_grad_output(self, module, grad_input, grad_output):
        """Hook for saving gradient w.r.t output"""
        if self.steps % self.fac_update_freq == 0:
            self.layers[module].save_grad_outputs(grad_output)

    def _register_module(self, module):
        """Create and register a KFAC layer for a module.

        Note: For a single module, there may be multiple KFAC layers
          registered. E.g. kfac.modules.LSTMCell is made up of two 
          torch.nn.Linear so both Linear modules will have a registered KFAC
          Layer.
        """
        layer_list = kfac_layers.get_kfac_layers(
            module,
            use_eigen_decomp = self.use_eigen_decomp, 
            damping = self.damping,
            factor_decay = self.factor_decay,
            batch_averaged = self.batch_averaged
        )
        for module, kfac_layer in layer_list:
            if self.backend.rank() == 0:
                print('Registered layer {} ({})'.format(kfac_layer.__class__.__name__,
                        module.__class__.__name__))
            self.layers[module] = kfac_layer
            module.register_forward_pre_hook(self._save_input)
            module.register_backward_hook(self._save_grad_output)

    def _register_modules(self, model):
        """Iterate over and register modules that KFAC supports."""
        for module in model.children():
            if module.__class__.__name__.lower() not in self.known_modules:
                self._register_modules(module)
            else:
                if kfac_layers.module_requires_grad(module):
                    self._register_module(module)

    def _compute_grad_scale(self):
        """Computes scale factor for preconditioned gradients

        Returns:
          sum_{layers} (sum_{gradients} precon_grad * grad * lr^2) 
        """
        vg_sum = 0.
        for module, layer in self.layers.items():
            v = layer.preconditioned_gradient
            vg_sum += (v[0] * layer._get_weight_grad().data *
                       self.lr ** 2).sum().item()
            if layer.has_bias:
                vg_sum += (v[1] * layer._get_bias_grad().data * 
                           self.lr ** 2).sum().item()
        return min(1.0, math.sqrt(self.kl_clip / abs(vg_sum)))

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one K-FAC step

        Note:
        - this function should always be called before `optimizer.step()`
        - gradients must be averaged across ranks before calling `step()`

        Args:
          closure: for compatibility with the base optimizer class.
              `closure` is ignored by KFAC
        """

        # Update params, used for compatibilty with `KFACParamScheduler`
        group = self.param_groups[0]
        self.lr = group['lr']
        self.damping = group['damping']
        self.fac_update_freq = group['fac_update_freq']
        self.kfac_update_freq = group['kfac_update_freq']

        updates = {}
        handles = []

        if self.steps % self.fac_update_freq == 0:
            for layer in self.layers.values():
                layer.update_A_factor()
                layer.update_G_factor()

            # We do this after layer.update_*_factors because the inverse buffers
            # are not instantiate until this point and we use the size of the
            # buffers to approximate the time each layer will take to compute.
            if self.steps == 0:
                self._assign_layers_to_workers()

            if self.backend.size() > 1:
                self._allreduce_factors()

        if self.steps % self.kfac_update_freq == 0:
            rank = self.backend.rank()
            for i, layer in enumerate(self.layers.values()):
                layer.compute_A_inv(rank)
                layer.compute_G_inv(rank)
            if self.backend.size() > 1:
                self._broadcast_inverses()

        for layer in self.layers.values():
            layer.compute_preconditioned_gradient()

        nu = None if self.kl_clip is None else self._compute_grad_scale()

        for layer in self.layers.values():
            layer.update_gradient(nu)

        self.steps += 1

    def _allreduce_factors(self):
        """Allreduce the factors for all layers"""
        tensors = []

        for layer in self.layers.values():
            tensors.extend(layer.get_factors())

        self.backend.allreduce(tensors, op=self.backend.Average)

    def _broadcast_inverses(self):
        """Broadcast the eigendecomp/invs for all layers"""
        tensors = []
        ranks = []

        for layer in self.layers.values():
            tensor_list, rank_list = layer.get_inverses(return_ranks=True)
            tensors.extend(tensor_list)
            ranks.extend(rank_list)

        assert len(tensors) == len(ranks) 
        self.backend.broadcast(tensors, ranks)

    def _assign_layers_to_workers(self):
        """Assigns layers to workers to minimize max load on any worker.

        Approximates load by estimating inverse computation time as O(n^3)
        for each n x n factor.
        """
        if len(self.layers) == 0:
            return

        func = lambda n: n**3  # approx inverse complexity
        a_sizes = [l.A_inv.shape[0] for l in self.layers.values()]
        g_sizes = [l.G_inv.shape[0] for l in self.layers.values()]
        a_times = list(map(func, a_sizes))
        g_times = list(map(func, g_sizes))
            
        if self.distribute_layer_factors:
            times = a_times + g_times
            locs = utils.load_balance(self.backend.size(), times)
            a_locs, g_locs = locs[0:len(a_times)], locs[len(a_times):]
        else:
            times = [sum(x) for x in zip(a_times, g_times)]
            locs = utils.load_balance(self.backend.size(), times)
            a_locs, g_locs = locs, locs

        for i, layer in enumerate(self.layers.values()):
            layer.A_rank = a_locs[i]
            layer.G_rank = g_locs[i]

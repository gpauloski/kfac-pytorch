import math
import warnings
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
      damping (float, optional): Tikhonov damping parameter (default: 0.001)
      factor_decay (float, optional): running average coefficient for Kronecker
          factors (default: 0.95)
      fac_update_freq (int, optional): iterations between calculating and
          updating the running average of the Kronecker factors (default: 10)
      kfac_update_freq (int, optional): iterations between applying gradient
          preconditioning (default: 100)
      kl_clip (float, optional): clipping parameter for gradient scaling. If
          None, no scaling/clipping will be applied. (default: 0.001)
      lr (float, optional): learning rate (default: 0.1)
      accumulate_data (bool, optional): if `True`, accumulates the input/output
          data for each KFAC registered module. This is useful if you have a
          module that is called multiple times per optimization step (e.g.
          LSTMCells) or if you are accumulating gradients over multiple batches
          and you want KFAC to use the input/output for all batches when
          computing the factors. Note: if accumulating the data, memory usage
          can increase substantially. (default: True)
      batch_first (bool, optional): True if the batch dimension is dim 0
          (default: True)
      compute_factor_in_hook (bool, optional): If `True`, compute the factors
          during the module forward/backward pass hooks and add to the running
          average. Recommended if using gradient accumulation and 
          `accumulate_data=False`, however it is usually slower. If `False`,
          factors are computed during `KFAC.step()`. (default: False)
      distribute_layer_factors (bool, optional): if `True`, computes factors A
          and G on different workers else computes A and G for a single layer
          on the same worker. For small worker counts, computing per layer
          factors on the same device can yeild improvements. (default: True)
      use_eigen_decomp (bool, optional): use the eigendecomposition method for
          the KFAC update, otherwise use normal inv method (default: True)
      skip_layers (str or list, optional): name or list of names of modules to
          ignore when registering layers. Note: this prevents recursively
          registering within an ignored module. I.e. if you have a module named
          `my_module` and skip it, then any sub module of `my_module` will also
          be skipped even if it is not explicitly passed to `skip_layers`. 
          (default: None)
      verbose (bool, optional): print information about registered layers
    """
    def __init__(self,
                 model,
                 damping=0.001,
                 factor_decay=0.95,
                 fac_update_freq=10,
                 kfac_update_freq=100,
                 kl_clip=0.001,
                 lr=0.1,
                 accumulate_data=True,
                 batch_first=True,
                 compute_factor_in_hook=False,
                 distribute_layer_factors=True,
                 use_eigen_decomp=True,
                 skip_layers=None,
                 verbose=True):

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

        self.damping = damping
        self.factor_decay = factor_decay
        self.fac_update_freq = fac_update_freq
        self.kfac_update_freq = kfac_update_freq
        self.kl_clip = kl_clip
        self.lr = lr
        self.accumulate_data = accumulate_data
        self.batch_first = batch_first
        self.compute_factor_in_hook = compute_factor_in_hook
        self.distribute_layer_factors = distribute_layer_factors
        self.use_eigen_decomp = use_eigen_decomp
        self.verbose = verbose

        self.backend = utils.get_comm_backend()

        if isinstance(skip_layers, str):
            self.skip_layers = [skip_layers.lower()]
        elif isinstance(skip_layers, list):
            self.skip_layers = [s.lower() for s in skip_layers]
        self.known_modules = {m.lower() for m in kfac_layers.KNOWN_MODULES}
        for layer in self.skip_layers: 
            self.known_modules.discard(layer)

        self.layers = []
        self.hook_layers = {}  # key: nn.Module, value: KFACLayer
        self.register_modules(model)

    def register_module(self, module):
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
            batch_first = self.batch_first,
            accumulate_data = self.accumulate_data
        )
        for module, kfac_layer in layer_list:
            if self.backend.rank() == 0 and self.verbose:
                print('Registered: {}'.format(kfac_layer))
            self.hook_layers[module] = kfac_layer
            self.layers.append(kfac_layer)
            module.register_forward_pre_hook(self._save_input)
            module.register_backward_hook(self._save_grad_output)

    def register_modules(self, model):
        """Iterate over and register modules that KFAC supports."""
        for module in model.children():
            # Do not recurse in children if we are skipping the module
            if module.__class__.__name__.lower() in self.skip_layers:
                pass
            # Recurse into module if we are not skipping it but it is also not
            # a known module to KFAC
            elif module.__class__.__name__.lower() not in self.known_modules:
                self.register_modules(module)
            # This is a known module to KFAC so register it if it is trainable
            else:
                if kfac_layers.module_requires_grad(module):
                    self.register_module(module)

    def register_shared_module(self, main_module, second_module, reverse_hooks=False):
        """Create and register a KFAC layer for modules that share a weight

        Useful for the case where two modules share a weight matrix and you want to
        incorporate the input and grad_output for both modules. E.g. in a language
        model it is common to tie the embedding and decoding (a linear module) weights
        but if only the embedding module is registered with KFAC, the forward and
        backward pass information will be lost for the linear module.

        Args:
          main_module (nn.Module): main module to register, a pointer to this module
              will be saved with the KFACLayer instance.
          second_module (nn.Module): the secondary module that shares its weight matrix
              with `main_module`. Only the forward/backward hooks will be registered
              for this module.
          reverse_hooks (bool, optional): if True, reverse the hooks for the
              `second_module`. Useful in cases such as tied embeddings where the input
              to the embedding is related to the output of the decoding.
        """
        warnings.warn('Registering shared weight modules with KFAC is '
                      'experimental and may produce poor results')

        if not isinstance(main_module, torch.nn.Module):
            raise ValueError('main_module must be of type torch.nn.Module')
        if not isinstance(second_module, torch.nn.Module):
            raise ValueError('second_module must be of type torch.nn.Module')
        # Note: this is because the second module hook that gets called will
        # overwrite the saved data from the first module hook call so we need
        # the hook calls to accumulate the data and not just save the most recent
        if not self.accumulate_data:
            raise ValueError('shared weight module registration will not work '
                             'is self.accumulate_data=False')
        layer_list = kfac_layers.get_kfac_layers(
            main_module,
            use_eigen_decomp = self.use_eigen_decomp, 
            damping = self.damping,
            factor_decay = self.factor_decay,
            batch_first = self.batch_first
        )
        
        if len(layer_list) > 1:
            raise ValueError('KFAC registering for shared weight modules does not work '
                             'for modules with multiple KFACLayers (e.g. LSTMCells)')
        else:
            _, kfac_layer = layer_list[0]

        if self.backend.rank() == 0 and self.verbose:
            print('Registered: {} (shared weight)'.format(kfac_layer))
        self.hook_layers[main_module] = kfac_layer
        self.hook_layers[second_module] = kfac_layer
        self.layers.append(kfac_layer)
        main_module.register_forward_pre_hook(self._save_input)
        main_module.register_backward_hook(self._save_grad_output)
        # TODO(gpauloski): this will not work with compute_factor_in_hook=True
        # because the factors may be computed before _save_*_as_*() is called.
        if reverse_hooks:
            second_module.register_forward_pre_hook(self._save_input_as_grad_output)
            second_module.register_backward_hook(self._save_grad_output_as_input)
        else:
            second_module.register_forward_pre_hook(self._save_input)
            second_module.register_backward_hook(self._save_grad_output)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one K-FAC step

        Note:
        - This function should always be called before `optimizer.step()` as
          it modifies the gradients in-place and does not modify the weights.
        - Gradients must be averaged across ranks before calling `step()`.
          This condition is guarenteed to be true if using `torch.distributed`
          as gradients are communicated during `loss.backward()`.

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
            if not self.compute_factor_in_hook:
                for layer in self.layers:
                    layer.update_A_factor()
                    layer.update_G_factor()

            # We do this after layer.update_*_factor because the buffers
            # are not instantiate until this point and we use the size of the
            # buffers to approximate the time each layer will take to compute.
            if self.steps == 0:
                self._assign_layers_to_workers()

            if self.backend.size() > 1:
                self._allreduce_factors()

        if self.steps % self.kfac_update_freq == 0:
            rank = self.backend.rank()
            for i, layer in enumerate(self.layers):
                layer.compute_A_inv(rank)
                layer.compute_G_inv(rank)
            if self.backend.size() > 1:
                self._broadcast_inverses()

        for layer in self.layers:
            layer.compute_preconditioned_gradient()

        nu = None if self.kl_clip is None else self._compute_grad_scale()

        for layer in self.layers:
            layer.update_gradient(nu)

        self.steps += 1

    def memory_usage(self):
        """Returns approximate memory usage for KFAC

        Note: this does not take into account:
          - intermediate memory requirements of computations
          - input/output accumulation depending on when the function is called
        """
        b = 0

        def sizeof_tensor(tensor):
            return tensor.nelement() * tensor.element_size() if tensor is not None else 0

        for layer in self.layers:
            b += sizeof_tensor(layer.A_factor)
            b += sizeof_tensor(layer.G_factor)
            b += sizeof_tensor(layer.A_inv)
            b += sizeof_tensor(layer.G_inv)
            b += sum(map(sizeof_tensor, layer.a_inputs))
            b += sum(map(sizeof_tensor, layer.g_outputs))
        return b

    def _allreduce_factors(self):
        """Allreduce the factors for all layers"""
        tensors = []

        for layer in self.layers:
            tensors.extend(layer.get_factors())

        self.backend.allreduce(tensors, op=self.backend.Average)

    def _assign_layers_to_workers(self):
        """Assigns layers to workers to minimize max load on any worker.

        Approximates load by estimating inverse computation time as O(n^3)
        for each n x n factor.
        """
        if len(self.layers) == 0:
            return

        func = lambda n: n**3  # approx inverse complexity
        a_sizes = [l.A_inv.shape[0] for l in self.layers]
        g_sizes = [l.G_inv.shape[0] for l in self.layers]
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

        for i, layer in enumerate(self.layers):
            layer.A_rank = a_locs[i]
            layer.G_rank = g_locs[i]

    def _broadcast_inverses(self):
        """Broadcast the eigendecomp/invs for all layers"""
        tensors = []
        ranks = []

        for layer in self.layers:
            tensor_list, rank_list = layer.get_inverses(return_ranks=True)
            tensors.extend(tensor_list)
            ranks.extend(rank_list)

        assert len(tensors) == len(ranks) 
        self.backend.broadcast(tensors, ranks)

    def _compute_grad_scale(self):
        """Computes scale factor for preconditioned gradients

        Returns:
          sum_{layers} (sum_{gradients} precon_grad * grad * lr^2) 
        """
        vg_sum = 0.
        for layer in self.layers:
            v = layer.preconditioned_gradient
            vg_sum += (v[0] * layer._get_weight_grad().data *
                       self.lr ** 2).sum().item()
            if layer.has_bias:
                vg_sum += (v[1] * layer._get_bias_grad().data * 
                           self.lr ** 2).sum().item()
        if vg_sum == 0.0:
            return None
        return min(1.0, math.sqrt(self.kl_clip / abs(vg_sum)))

    def _periodic_hook(grad_enabled=True):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                if grad_enabled:
                    if (torch.is_grad_enabled() and 
                        self.steps % self.fac_update_freq == 0):
                        func(self, *args, **kwargs)
                else:
                    if self.steps % self.fac_update_freq == 0:
                        func(self, *args, **kwargs)
            return wrapper
        return decorator

    @_periodic_hook(grad_enabled=True)
    def _save_input(self, module, input):
        self.hook_layers[module].save_inputs(input)
        if self.compute_factor_in_hook:
            self.hook_layers[module].update_A_factor()

    @_periodic_hook(grad_enabled=False)
    def _save_grad_output(self, module, grad_input, grad_output):
        self.hook_layers[module].save_grad_outputs(grad_output)
        if self.compute_factor_in_hook:
            self.hook_layers[module].update_G_factor()

    @_periodic_hook(grad_enabled=True)
    def _save_input_as_grad_output(self, module, input):
        self.hook_layers[module].save_grad_outputs(input)

    @_periodic_hook(grad_enabled=False)
    def _save_grad_output_as_input(self, module, grad_input, grad_output):
        self.hook_layers[module].save_inputs(grad_output)

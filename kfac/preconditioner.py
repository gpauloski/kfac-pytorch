import enum
import math
import warnings

import torch
import torch.optim as optim

from . import layers as kfac_layers
from . import utils
from . import comm

try:
    from torch.cuda.amp import autocast
    TORCH_FP16 = True
except ImportError:
    TORCH_FP16 = False


class CommMethod(enum.Enum):
    """KFAC Communication Method

    - COMM_OPT: Optimize KFAC to reduce communication by decoupling gradient
        preconditioning from inverse calculations. This method is referred
        to as 'KFAC_opt' in https://arxiv.org/abs/2007.00784.
    - MEM_OPT: Optimize KFAC to reduce memory usage by computing the inverse
        calculations and preconditioned gradient for a single layer on one
        worker and broadcasting the gradient to all workers. This method is 
        referred to as 'KFAC_lw' in https://arxiv.org/abs/2007.00784.
    - HYBRID_OPT: A hybrid between COMM_OPT and MEM_OPT. A user defined
        fraction of the workers will receive the inverses and precondition
        the gradients, while the remaining workers will wait for the 
        preconditioned gradient to be sent for them.
    """
    COMM_OPT = 1
    MEM_OPT = 2
    HYBRID_OPT = 3


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
      factor_update_freq (int, optional): iterations between calculating and
          updating the running average of the Kronecker factors (default: 10)
      inv_update_freq (int, optional): iterations between applying gradient
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
          can increase substantially. (default: False)
      assignment_strategy (str, optional): One of ['compute', 'memory'].
          Controls if the eigen decomposition placement strategy should 
          optimize for balancing the compute time or memory usage across
          workers. (default: compute)
      batch_first (bool, optional): True if the batch dimension is dim 0
          (default: True)
      comm_method (CommMethod, optional): Communication optimization
          to use. See `CommMethod` docstring for more info. (default: COMM_OPT)
      compute_factor_in_hook (bool, optional): If `True`, compute the factors
          during the module forward/backward pass hooks and add to the running
          average. Recommended if using gradient accumulation and 
          `accumulate_data=False`, however it is usually slower. If `False`,
          factors are computed during `KFAC.step()`. (default: False)
      distribute_layer_factors (bool, optional): if `True`, computes factors A
          and G on different workers else computes A and G for a single layer
          on the same worker. For small worker counts, computing per layer
          factors on the same device can yeild improvements. When using 
          `MEM_OPT`, distribute_layer_factors is forced to False. 
          (default: True)
      inv_dtype (torch.dtype, optional): Data type to store inverses in. FP32
          is recommended. (default: torch.float32)
      grad_scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler used
          if using torch.cuda.amp for fp16 training. KFAC will use the same
          data type for storing the factors as the data in the forward/backward
          pass. E.g. with torch.cuda.amp, the forward/backward pass data is
          autocast to the lowest precision possible so KFAC will store the
          factors in that precision, saving memory. The grad scaler
          is needed by KFAC to correctly unscale the gradients from the
          backward pass. (default: None)
      factor_dtype (torch.dtype, optional): Data type to store factors in.
          None keeps factors in their exisiting datatype (default: None)
      grad_worker_fraction (float): fraction of workers to compute the
          preconditioned gradient for a layer. Remaining workers will wait for
          the gradient to be sent to them. In COMM_OPT, this fraction is 1.0,
          and in MEM_OPT, this fraction is 1/world_size such that this
          parameter is only used in HYBRID_OPT where the user defines the
          fraction. Note: this fraction should produce evenly sized groups.
          E.g. world_size=8 and fraction=0.33 would produce groups of size
          3, 3, and 2 which would not be valid. (default: 0.25).
      precompute_outer_eigen (bool, optional): If True, 
          1 / (dG * dG.T + damping) will be computed on the inverse worker and
          broadcast to remaining gradient workers. This value is redundantly
          computed every non-KFAC update step on every worker so precomputing
          on a single worker can reduce computation at the cost of slightly
          more memory (default: False).
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
                 factor_update_freq=10,
                 inv_update_freq=100,
                 kl_clip=0.001,
                 lr=0.1,
                 accumulate_data=False,
                 assignment_strategy='compute',
                 batch_first=True,
                 comm_method=CommMethod.COMM_OPT,
                 compute_factor_in_hook=False,
                 distribute_layer_factors=True,
                 inv_dtype=torch.float32,
                 grad_scaler=None,
                 grad_worker_fraction=0.25,
                 factor_dtype=None,
                 precompute_outer_eigen=True,
                 use_eigen_decomp=True,
                 skip_layers=[],
                 verbose=False):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < factor_decay <= 1:
            raise ValueError("Invalid factor decay rate: {}".format(
                             factor_decay))
        if not 0.0 < damping:
            raise ValueError("Invalid damping: {}".format(damping))
        if kl_clip is not None and not 0.0 < kl_clip:
            raise ValueError("Invalid clipping value: {}".format(kl_clip))
        if not 0 < factor_update_freq:
            raise ValueError("Invalid factor update frequency: {}".format(
                             factor_update_freq))
        if not 0 < inv_update_freq:
            raise ValueError("Invalid K-FAC update frequency: {}".format(
                             inv_update_freq))
        if not 0 == inv_update_freq % factor_update_freq:
            warnings.warn('It is suggested that inv_update_freq be a multiple '
                          'of factor_update_freq')
        if not assignment_strategy in ['compute', 'memory']:
            raise ValueError(
                    'assignment_strategy must be "compute" or "memory"')
        # TODO(gpauloski): asserting distribute_layer_factors if HYBRID_OPT is
        # not actually necessary because we will have to communicate the
        # inverses anyways, but we would need to constrain the device placement
        # for the two inverses of a layer to be in the same inverse
        # communication group
        if comm_method in [CommMethod.MEM_OPT, CommMethod.HYBRID_OPT] \
                and distribute_layer_factors is True:
            warnings.warn('MEM_OPT or HYBRID_OPT and distribute_layer_factors='
                          'True cannot be used at the same time. Defaulting '
                          'to distribute_layer_factors=False')
            distribute_layer_factors = False

        known_modules = {m.lower() for m in kfac_layers.KNOWN_MODULES}
        if skip_layers is not None:
            if isinstance(skip_layers, str):
                skip_layers = [skip_layers.lower()]
            elif isinstance(skip_layers, list):
                skip_layers = [s.lower() for s in skip_layers]
            for layer in skip_layers: 
                known_modules.discard(layer)
        else:
            skip_layers = []

        # For compatibility with `KFACParamScheduler`
        defaults = dict(
            damping=damping,
            factor_decay=factor_decay,
            factor_update_freq=factor_update_freq,
            inv_update_freq=inv_update_freq,
            kl_clip=kl_clip,
            lr=lr,
            step=0
        ) 

        # KFAC does not register parameters so we pass fake tensor
        super(KFAC, self).__init__([torch.tensor(0.0)], defaults)

        # We do not need to save the params to the default group because
        # they are not dependent on the current state of training
        self.accumulate_data = accumulate_data
        self.assignment_strategy = assignment_strategy
        self.batch_first = batch_first
        self.comm_method = comm_method
        self.compute_factor_in_hook = compute_factor_in_hook
        self.distribute_layer_factors = distribute_layer_factors
        self.inv_dtype = inv_dtype
        self.grad_scaler = grad_scaler
        self.factor_dtype = factor_dtype
        self.precompute_outer_eigen = precompute_outer_eigen
        self.use_eigen_decomp = use_eigen_decomp
        self.skip_layers = skip_layers
        self.known_modules = known_modules
        self.verbose = verbose
        self.workers_assigned = False

        comm.init_comm_backend()
        if self.comm_method == CommMethod.COMM_OPT:
            self.grad_worker_fraction = 1
        elif self.comm_method == CommMethod.MEM_OPT:
            self.grad_worker_fraction = 0
        elif self.comm_method == CommMethod.HYBRID_OPT:
            size = comm.backend.size()
            if 'Horovod' in self.comm_method.__class__.__name__:
                raise ValueError('HYBRID_OPT does not support Horovod backend')
            if 0 > grad_worker_fraction or 1 < grad_worker_fraction:
                raise ValueError('grad_worker_fraction must in (0, 1) when '
                                 'using HYBRID_OPT')
            if size % min(1, round(size * grad_worker_fraction)) != 0:
                raise ValueError('grad_worker_fraction must produce groups of '
                                 'equal size')
            if 1 / size >= grad_worker_fraction:
                warnings.warn('grad_worker_fraction <= 1/world_size, for best '
                              'performance, use COMM_OPT')
            elif 1 - (1 / size) <= grad_worker_fraction:
                warnings.warn('grad_worker_fraction >= 1-1/world_size, for '
                              'best performance, use COMM_OPT')
            elif 0.5 < grad_worker_fraction:
                warnings.warn('grad_worker_fraction={}, for best performance '
                              'use a value in [0, 0.5] when using '
                              'HYBRID_OPT.'.format(grad_worker_fraction))
            self.grad_worker_fraction = grad_worker_fraction

        self.layers = []
        self.hook_layers = {}  # key: nn.Module, value: KFACLayer
        self.register_model(model)

    def __repr__(self):
        extra_params = {
            'accumulate_data': self.accumulate_data,
            'assignment_strategy': self.assignment_strategy,
            'batch_first': self.batch_first,
            'comm_method': self.comm_method,
            'compute_factor_in_hook': self.compute_factor_in_hook,
            'distribute_layer_factors': self.distribute_layer_factors,
            'inv_dtype': self.inv_dtype,
            'grad_scaler': True if self.grad_scaler is not None else False,
            'grad_worker_fraction': self.grad_worker_fraction,
            'factor_dtype': self.factor_dtype,
            'known_modules': self.known_modules,
            'precompute_outer_eigen': self.precompute_outer_eigen,
            'use_eigen_decomp': self.use_eigen_decomp,
            'skip_layers': self.skip_layers,
            'verbose': self.verbose,
            'registered_layers': len(self.layers),
        }
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups + [extra_params]):
            format_string += '\n'
            format_string += 'Parameter Group {0}\n'.format(i)
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, group[key]) 
        format_string += ')'
        return format_string

    def state_dict(self, include_layer_factors=True, 
                   include_layer_inverses=False):
        """Returns KFAC state dict.

        Args:
          include_layer_factors (optional, bool): include tensors with factors
              for all registered KFACLayers as a part of the state_dict. Note: 
              can make the state_dict fairly large. (default: True)
          include_layer_inverses (optional, bool): include tensors with inverse
              for all registered KFACLayers as a part of the state_dict. Note: 
              can make the state_dict fairly large. If False, the inverses can
              be recomputed from the factors to save storage space 
              (default: False).
        """
        state_dict = super(KFAC, self).state_dict()
        layers = None
        if include_layer_factors:
            if (self.comm_method is CommMethod.MEM_OPT 
                    and include_layer_inverses):
                warnings.warn('Layer inverses cannot be saved to the state '
                              'dict when using CommMethod.MEM_OPT. Skipping '
                              'saving inverses.')
                include_layer_inverses = False
            layers = [layer.state_dict(include_layer_inverses)
                      for layer in self.layers]
        state_dict['layers'] = layers
        return state_dict

    def load_state_dict(self, state_dict, compute_inverses=True):
        """Loads the KFAC state.

        Args:
          state_dict (dict): KFAC state. Should be an object returned from a
              call to `state_dict`.
          compute_inverse (bool, optional): if True, compute the inverses
              from the loaded factors. This is useful if the loaded state dict
              was produced from with a call to state_dict() with 
              `include_layer_inverses=False`. (default: True)
        """
        if state_dict['layers'] is not None:
            if len(state_dict['layers']) != len(self.layers):
                raise ValueError('loaded state dict contains a different '
                                 'number of layers')
            for layer, layer_state in zip(self.layers, state_dict['layers']):
                layer.load_state_dict(layer_state)
            state_dict = {key: state_dict[key] for key in state_dict
                          if key != 'layers'}
        else:
            warnings.warn('Layer factors are not included in the state_dict so '
                          'inverses cannot be computed. Skipping inverse '
                          'computation.')
            compute_inverses = False  # Cannot be computed if no layers
        super(KFAC, self).load_state_dict(state_dict)
        if compute_inverses:
            self._assign_workers()
            self.workers_assigned = True
            self.compute_inverses(damping=self.param_groups[0]['damping'])
            if self.comm_method in [CommMethod.COMM_OPT, 
                                    CommMethod.HYBRID_OPT]:
                self.broadcast_inverses()

    def register_module(self, module, name=None):
        """Create and register a KFAC layer for a module.

        Note: For a single module, there may be multiple KFAC layers
          registered. E.g. kfac.modules.LSTMCell is made up of two 
          torch.nn.Linear so both Linear modules will have a registered KFAC
          Layer.
        """
        layer_list = kfac_layers.get_kfac_layers(
            module,
            accumulate_data = self.accumulate_data,
            batch_first = self.batch_first,
            inv_dtype = self.inv_dtype,
            grad_scaler = self.grad_scaler,
            factor_dtype = self.factor_dtype,
            prediv_eigenvalues = self.precompute_outer_eigen,
            use_eigen_decomp = self.use_eigen_decomp,
        )
        for module, kfac_layer in layer_list:
            if comm.backend.rank() == 0 and self.verbose:
                print('Registered {}: {}'.format(
                        name if name is not None else '', kfac_layer))
            self.hook_layers[module] = kfac_layer
            self.layers.append(kfac_layer)
            module.register_forward_pre_hook(self._save_input)
            module.register_backward_hook(self._save_grad_output)

    def register_submodules(self, parent_module, prefix=''):
        """Iterate over and register submodules that KFAC supports."""
        for name, module in parent_module.named_children():
            name = prefix + ('.' if prefix != '' else '') + name
            module_name = module.__class__.__name__.lower()
            if module_name in self.skip_layers:
                pass
            elif module_name not in self.known_modules:
                self.register_submodules(module, prefix=name)
            elif (kfac_layers.module_requires_grad(module) and
                    module not in self.hook_layers):
                self.register_module(module, name)

    def register_model(self, model):
        """Registers a model to KFAC."""
        if len(list(model.children())) == 0:  # Handle if model is a module
            if (model.__class__.__name__.lower() in self.known_modules and
                model.__class__.__name__.lower() not in self.skip_layers):
                self.register_module(model)
        else:
            self.register_submodules(model)

    def register_shared_module(self, main_module, second_module,
                               reverse_hooks=False):
        """Create and register a KFAC layer for modules that share a weight

        Useful for the case where two modules share a weight matrix and you 
        want to incorporate the input and grad_output for both modules. E.g. in
        a language model it is common to tie the embedding and decoding (a 
        linear module) weights but if only the embedding module is registered 
        with KFAC, the forward and backward pass information will be lost for 
        the linear module.

        Args:
          main_module (nn.Module): main module to register, a pointer to this 
             module  will be saved with the KFACLayer instance.
          second_module (nn.Module): the secondary module that shares its 
              weight matrix with `main_module`. Only the forward/backward hooks
              will be registered for this module.
          reverse_hooks (bool, optional): if True, reverse the hooks for the
              `second_module`. Useful in cases such as tied embeddings where 
              the input to the embedding is related to the output of the 
              decoding.
        """
        warnings.warn('Registering shared weight modules with KFAC is '
                      'experimental and may produce poor results')

        if not isinstance(main_module, torch.nn.Module):
            raise ValueError('main_module must be of type torch.nn.Module')
        if not isinstance(second_module, torch.nn.Module):
            raise ValueError('second_module must be of type torch.nn.Module')
        # Note: this is because the second module hook that gets called will
        # overwrite the saved data from the first module hook call so we need
        # the hook calls to accumulate the data and not just save the most 
        # recent
        if not self.accumulate_data:
            raise ValueError('shared weight module registration will not work '
                             'is self.accumulate_data=False')
        layer_list = kfac_layers.get_kfac_layers(
            main_module,
            use_eigen_decomp = self.use_eigen_decomp, 
            batch_first = self.batch_first,
            accumulate_data = self.accumulate_data
        )
        
        if len(layer_list) > 1:
            raise ValueError('KFAC registering for shared weight modules does '
                             'not work for modules with multiple KFACLayers '
                             '(e.g. LSTMCells)')
        else:
            _, kfac_layer = layer_list[0]

        if comm.backend.rank() == 0 and self.verbose:
            print('Registered: {} (shared weight)'.format(kfac_layer))
        self.hook_layers[main_module] = kfac_layer
        self.hook_layers[second_module] = kfac_layer
        self.layers.append(kfac_layer)
        main_module.register_forward_pre_hook(self._save_input)
        main_module.register_backward_hook(self._save_grad_output)
        # TODO(gpauloski): this will not work with compute_factor_in_hook=True
        # because the factors may be computed before _save_*_as_*() is called.
        if reverse_hooks:
            second_module.register_forward_pre_hook(
                    self._save_input_as_grad_output)
            second_module.register_backward_hook(
                    self._save_grad_output_as_input)
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
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        params = self.param_groups[0]

        if params['step'] % params['factor_update_freq'] == 0:
            if not self.compute_factor_in_hook:
                self.compute_factors(alpha=params['factor_decay'])
            self.allreduce_factors()

        # We do this after compute_factors() because the buffers
        # are not instantiated until this point and we use the size of the
        # buffers to approximate the time each layer will take to compute.
        if not self.workers_assigned:
            self._assign_workers()
            self.workers_assigned = True

        if params['step'] % params['inv_update_freq'] == 0:
            self.compute_inverses(damping=params['damping'])
            if self.comm_method in \
                    [CommMethod.COMM_OPT, CommMethod.HYBRID_OPT]:
                self.broadcast_inverses()

        self.compute_preconditioned_gradients(damping=params['damping'])
        if self.comm_method in [CommMethod.MEM_OPT, CommMethod.HYBRID_OPT]:
            self.broadcast_gradients()

        scale = None if params['kl_clip'] is None \
                     else self._compute_grad_scale()

        self.update_gradients(scale)

        params['step'] += 1

        return loss

    def allreduce_factors(self):
        """Allreduce the factors for all layers"""
        if comm.backend.size() == 1:
            return

        handles = []
        for layer in self.layers:
            handles.extend(layer.allreduce_factors())
        comm.backend.sync(handles)

    def broadcast_inverses(self):
        """Broadcast the eigendecomp/invs for all layers"""
        if comm.backend.size() == 1:
            return
        
        handles = []
        for layer in self.layers:
            handles.extend(layer.broadcast_inverses())
        comm.backend.sync(handles)

    def broadcast_gradients(self):
        """Broadcast the preconditioned gradients for all layers"""
        if comm.backend.size() == 1:
            return
        
        handles = []
        for layer in self.layers:
            handles.extend(layer.broadcast_gradient())
        comm.backend.sync(handles)

    @torch.no_grad()
    def compute_inverses(self, damping=0.001):
        """Compute inverses of all factors.

        Args:
          damping (float, optional): inverse damping value (default: 0.001)
        """
        for layer in self.layers:
            layer.compute_A_inv(damping=damping)
            layer.compute_G_inv(damping=damping)
    
    @torch.no_grad()
    def compute_factors(self, alpha=0.95):
        """Compute all factors.

        Args:
          alpha (float, optional): running average parameter (default: 0.95)
        """
        for layer in self.layers:
            layer.update_A_factor(alpha=alpha)
            layer.update_G_factor(alpha=alpha)

    @torch.no_grad()
    def compute_preconditioned_gradients(self, damping=0.001):
        """Compute the preconditioned gradients for all layers.

        Args:
          damping (float, optional): damping value (default: 0.001)
        """
        for layer in self.layers:
            layer.compute_preconditioned_gradient(damping=damping)

    @torch.no_grad()
    def update_gradients(self, scale=None):
        for layer in self.layers:
            layer.update_gradient(scale=scale)

    def memory_usage(self):
        """Returns current approximate memory usage for KFAC

        Note: this does not take into account:
          - intermediate memory requirements of computations
          - input/output accumulation depending on when the function is called
          - differences in memory usage between workers when using 
            CommMethod.MEM_OPT
        """
        b = 0

        def sizeof_tensor(tensor):
            if isinstance(tensor, list) or isinstance(tensor, tuple):
                return sum([sizeof_tensor(t) for t in tensor])
            return (tensor.nelement() * tensor.element_size() 
                    if tensor is not None else 0)

        for layer in self.layers:
            b += sum(sizeof_tensor(value) for key, value in layer.state.items())
            b += sum(map(sizeof_tensor, layer.a_inputs))
            b += sum(map(sizeof_tensor, layer.g_outputs))
            b += sizeof_tensor(layer.preconditioned_gradient)
        return b

    def _assign_workers(self):
        """Assigns workers to minimize max load on any worker.

        Approximates load by estimating inverse computation time as O(n^3)
        for each n x n factor.
        """
        if len(self.layers) == 0:
            return

        if self.assignment_strategy == 'compute':
            func = lambda n: n**3  # approx inverse complexity
        elif self.assignment_strategy == 'memory':
            func = lambda n: n**2
        else:
            raise ValueError(
                    'assignment_strategy must be "compute" or "memory"')

        a_sizes = [l.state['A'].shape[0] for l in self.layers]
        g_sizes = [l.state['G'].shape[0] for l in self.layers]
        a_times = list(map(func, a_sizes))
        g_times = list(map(func, g_sizes))
            
        if self.distribute_layer_factors:
            times = a_times + g_times
            locs = utils.load_balance(comm.backend.size(), times)
            a_locs, g_locs = locs[0:len(a_times)], locs[len(a_times):]
        else:
            times = [sum(x) for x in zip(a_times, g_times)]
            locs = utils.load_balance(comm.backend.size(), times)
            a_locs, g_locs = locs, locs

        allocator = utils.WorkerAllocator(comm.backend.size(),
                self.grad_worker_fraction)

        for i, layer in enumerate(self.layers):
            layer.assign_inverse_workers(
                a_locs[i],
                g_locs[i],
                allocator.get_inv_group(a_locs[i]),
                allocator.get_inv_group(g_locs[i])
            )
            src_ranks = allocator.get_inv_ranks(a_locs[i])
            layer.assign_gradient_workers(src_ranks, 
                    allocator.get_grad_groups(src_ranks))

    def _compute_grad_scale(self):
        """Computes scale factor for preconditioned gradients

        Returns:
          sum_{layers} (sum_{gradients} precon_grad * grad * lr^2) 
        """
        vg_sum = 0.
        group = self.param_groups[0]
        lr = group['lr']
        kl_clip = group['kl_clip']
        for layer in self.layers:
            v = layer.preconditioned_gradient
            vg_sum += (
                v[0] * layer._get_weight_grad().data * lr ** 2
            ).sum().item()
            if layer.has_bias:
                vg_sum += (
                    v[1] * layer._get_bias_grad().data * lr ** 2
                ).sum().item()
        if vg_sum == 0.0:
            return None
        return min(1.0, math.sqrt(kl_clip / abs(vg_sum)))

    def _periodic_hook(grad_enabled=True):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                group = self.param_groups[0]
                step = group['step']
                update_freq = group['factor_update_freq']
                if grad_enabled:
                    if torch.is_grad_enabled() and step % update_freq == 0:
                        with torch.no_grad():
                            func(self, *args, **kwargs)
                else:
                    if step % update_freq == 0:
                        with torch.no_grad():
                            func(self, *args, **kwargs)
            return wrapper
        return decorator

    @_periodic_hook(grad_enabled=True)
    def _save_input(self, module, input):
        self.hook_layers[module].save_inputs(input)
        if self.compute_factor_in_hook:
            # We are inside the forward pass which could be in an autocast
            # region so temporarily disable autocast
            if TORCH_FP16:
                with autocast(enabled=False):
                    self.hook_layers[module].update_A_factor(
                            alpha=self.param_groups[0]['factor_decay'])
            else:
                self.hook_layers[module].update_A_factor(
                        alpha=self.param_groups[0]['factor_decay'])

    @_periodic_hook(grad_enabled=False)
    def _save_grad_output(self, module, grad_input, grad_output):
        self.hook_layers[module].save_grad_outputs(grad_output)
        if self.compute_factor_in_hook:
            # We are inside the backward pass which should never be in an
            # autocast region but just in case temporarily disable autocast
            if TORCH_FP16:
                with autocast(enabled=False):
                    self.hook_layers[module].update_G_factor(
                            alpha=self.param_groups[0]['factor_decay'])
            else:
                self.hook_layers[module].update_G_factor(
                        alpha=self.param_groups[0]['factor_decay'])

    @_periodic_hook(grad_enabled=True)
    def _save_input_as_grad_output(self, module, input):
        self.hook_layers[module].save_grad_outputs(input)

    @_periodic_hook(grad_enabled=False)
    def _save_grad_output_as_input(self, module, grad_input, grad_output):
        self.hook_layers[module].save_inputs(grad_output)


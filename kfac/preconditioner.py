from __future__ import annotations

import enum
import logging
import math
import warnings
from collections import defaultdict
from typing import Any
from typing import Callable

import torch
import torch.distributed as dist
import torch.optim as optim

from kfac.distributed import TorchDistributedCommunicator
from kfac.enums import AllreduceMethod
from kfac.enums import DistributedStrategy
from kfac.layers import get_kfac_layers
from kfac.layers import KNOWN_MODULES
from kfac.layers import module_requires_grad
from kfac.layers.base import KFACBaseLayer

logger = logging.getLogger(__name__)


class AssignmentStrategy(enum.Enum):
    """KFAC Factor Distribution Method

    KFAC assigns factors for second-order computation using a heuristic-based
    longest-processing time greedy algorithm. AssignmentStrategy.COMPUTE
    uses an estimation of the second-order computation time as the heuristic
    and AssignmentStrategy.MEMORY uses the memory requirements of storing
    the second-order results as the heuristic.
    """

    COMPUTE = 1
    MEMORY = 2


class ComputeMethod(enum.Enum):
    """KFAC Second Order Computation Method

    Controls if eigen decompositions or inverse of the factors will be used
    to precondition the gradients.
    """

    EIGEN = 1
    INVERSE = 2


class KFACPreconditioner(optim.Optimizer):
    """KFAC Distributed Gradient Preconditioner

    Preconditions the gradients of a model with a layer-wise FIM approximation.
    Layer computations are distributed across workers using torch.distributed.

    Example:
    >>> model = torch.nn.parallel.DistributedDataParallel(model, ...)
    >>> optimizer = optim.SGD(model.parameters(), ...)
    >>> preconditioner = KFAC(model, ...)
    >>>
    >>> for i, (data, target) in enumerate(train_loader):
    >>>     optimizer.zero_grad()
    >>>     output = model(data)
    >>>     loss = criterion(output, target)
    >>>     loss.backward()
    >>>     preconditioner.step()
    >>>     optimizer.step()

    Args:
      model (torch.nn.Module): model to register and perform KFAC updates on.
      factor_update_steps (int): steps between computing and updating the
          running average of the Kronecker factors.
      inv_update_steps (int): steps between recomputing and communicating
          the second-order information.
      damping (Callable, float): Tikhonov damping parameter or a callable
          that will return the damping parameter as a float (default: 0.001).
      factor_decay (Callable, float): running average coefficient for Kronecker
          factors or callable that will return the factor_decay
          (default: 0.95).
      kl_clip (Callable, float): clipping parameter for gradient scaling or
          a callable that returns a float. If None, no scaling/clipping
          will be applied (default: 0.001).
      lr (Callable, float): learning rate or callable that will return learning
          rate (default: 0.1).
      accumulation_steps (int): number of forward/backward passes
          between optimization steps (default: 1).
      allreduce_bucket_cap_mb (float): maximum size in megabytes for allreduce
          bucketing. If zero, bucketing is not used (default: 25).
      assignment_strategy (AssignmentStrategy, str): See `AssignmentStrategy`
          for more details (default: AssignmentStrategy.COMPUTE).
      colocate_factors (bool): assign both factors for a single layer to the
          same worker. Reccomended when num_layers < world_size
          (default: True).
      compute_method (ComputeMethod, str): See `ComputeMethod` for more
          details (default: ComputeMethod.EIGEN).
      compute_eigenvalue_outer_product (bool): when using the eigen compute
          method, precompute the element-wise inverse of the outer product of
          eigenvectors on the eigen decomposition worker rather to reduce
          computation in the gradient preconditioning stage.
          `colocate_factors` must be True (default: True).
      grad_worker_fraction (DistributedStrategy, float): controls the fraction
          of workers assigned as gradient workers for each layer. Optionally,
          predefined configurations can be passed using the
          DistributedStrategy enum (default: DistributedStrategy.COMM_OPT).
      symmetry_aware (bool): communicate only the upper triangle of symmetric
          matrices. Can reduce communication time when factors are large
          (default: False).
      grad_scaler (torch.cuda.amp.GradScaler or callable): Gradient scaler
          used for Torch AMP training. Used to unscale the G factors as they
          are accumulated during the backward pass. Alternatively can be
          a callable which will return the current scale (default: None).
      factor_dtype (torch.dtype): force data type for storing factors. If None,
          defaults to data type of intermediate values in forward/backward pass
          (default: None).
      inv_dtype (torch.dtype): force data type for storing second-order data
          (e.g., inverses or eigen decompositions) (default: torch.float32).
      skip_layers (list): list of module names to ignore when registering
          layers. Passing the name of parent modules will prevent recursively
          registering child modules of the parent. Case-insensitive
          (default: []).
      update_factors_in_hook (bool): If True, running average of factors is
          updated in the module hook and the async commmunication is started.
          Otherwise, this will be performed at the start of step() (default:
          True).
      loglevel (int): logging level (default: logging.DEBUG).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        factor_update_steps: int,
        inv_update_steps: int,
        # KFAC hyperparameters
        damping: Callable[[], float] | float = 0.001,
        factor_decay: Callable[[], float] | float = 0.95,
        kl_clip: Callable[[], float] | float = 0.001,
        lr: Callable[[], float] | float = 0.1,
        # Distribution strategy
        accumulation_steps: int = 1,
        allreduce_bucket_cap_mb: float = 25.0,
        assignment_strategy: (
            AssignmentStrategy | str
        ) = AssignmentStrategy.COMPUTE,
        colocate_factors: bool = True,
        compute_method: ComputeMethod | str = ComputeMethod.EIGEN,
        compute_eigenvalue_outer_product: bool = True,
        grad_worker_fraction: (
            DistributedStrategy | float
        ) = DistributedStrategy.COMM_OPT,
        symmetry_aware: bool = False,
        # Optional other parameters
        grad_scaler: (
            torch.cuda.amp.GradScaler | Callable[[], float] | None
        ) = None,
        factor_dtype: torch.dtype | None = None,
        inv_dtype: torch.dtype = torch.float32,
        skip_layers: list[str] | None = None,
        update_factors_in_hook: bool = True,
        loglevel: int = logging.DEBUG,
    ) -> None:
        if not 0 < factor_update_steps:
            raise ValueError('factor_update_steps must be > 0')
        if not 0 < inv_update_steps:
            raise ValueError('inv_update_steps must be > 0')
        if not callable(damping) and not 0.0 < damping:
            raise ValueError('damping must be > 0')
        if not callable(factor_decay) and not 0.0 < factor_decay <= 1:
            raise ValueError('factor_decay must be in (0, 1]')
        if not callable(kl_clip) and not 0.0 < kl_clip:
            raise ValueError('kl_clip must be > 0')
        if not callable(lr) and not 0.0 <= lr:
            raise ValueError('lr be > 0')
        if not 0 < accumulation_steps:
            raise ValueError('accumulation_steps must be > 0')
        if not 0 <= allreduce_bucket_cap_mb:
            raise ValueError('allreduce_bucket_cap_mb must be >= 0')
        if not 0 == inv_update_steps % factor_update_steps:
            warnings.warn(
                'It is suggested that inv_update_steps be an integer multiple '
                'of factor_update_steps',
            )
        if compute_eigenvalue_outer_product and not colocate_factors:
            raise ValueError(
                'colocate_factors must be True to use '
                'compute_eigenvalue_outer_product',
            )
        if isinstance(assignment_strategy, str):
            assignment_strategy = AssignmentStrategy[
                assignment_strategy.upper()
            ]
        if isinstance(compute_method, str):
            compute_method = ComputeMethod[compute_method.upper()]

        size = dist.get_world_size()
        if isinstance(grad_worker_fraction, DistributedStrategy):
            distributed_strategy = grad_worker_fraction
            if distributed_strategy == DistributedStrategy.COMM_OPT:
                grad_worker_fraction = 1.0
            elif distributed_strategy == DistributedStrategy.HYBRID_OPT:
                grad_worker_fraction = 0.5
            elif distributed_strategy == DistributedStrategy.MEM_OPT:
                grad_worker_fraction = 1 / size
            else:
                raise ValueError(f'Unknown enum {grad_worker_fraction}')
        else:
            if not 0 <= grad_worker_fraction or not 1 >= grad_worker_fraction:
                raise ValueError('grad_worker_fraction must in [0, 1]')
            if grad_worker_fraction == 0:
                grad_worker_fraction = 1 / size
            if size % min(1, round(size * grad_worker_fraction)) != 0:
                raise ValueError(
                    'grad_worker_fraction must produce groups of '
                    'equal size',
                )
            if grad_worker_fraction == 1:
                distributed_strategy = DistributedStrategy.COMM_OPT
            elif grad_worker_fraction <= 1 / size:
                distributed_strategy = DistributedStrategy.MEM_OPT
            else:
                distributed_strategy = DistributedStrategy.HYBRID_OPT
        assert isinstance(grad_worker_fraction, float)

        if (
            not colocate_factors
            and distributed_strategy is DistributedStrategy.MEM_OPT
        ):
            warnings.warn(
                'grad_worker_frac=1/world_size (MEM_OPT) requires '
                'colocate_factors=True. Enabling colocate_factors.',
            )
            colocate_factors = True

        known_modules = {m.lower() for m in KNOWN_MODULES}
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
            factor_update_steps=factor_update_steps,
            inv_update_steps=inv_update_steps,
            kl_clip=kl_clip,
            lr=lr,
            steps=0,
        )

        # We count the mini_steps (forward/backward passes between optimization
        # steps on a per-layer basis so the key for the dict is the module
        # registered with KFAC
        self.mini_steps: dict[
            torch._C.Module | torch.nn.Module,
            int,
        ] = defaultdict(int)

        # KFAC does not register parameters so we pass fake tensor
        # to satisfy torch.optim.Optimizer parameter checks.
        super().__init__([torch.tensor(0.0)], defaults)

        self.accumulation_steps = accumulation_steps
        self.allreduce_bucket_cap_mb = allreduce_bucket_cap_mb
        self.assignment_strategy = assignment_strategy
        self.colocate_factors = colocate_factors
        self.compute_method = compute_method
        self.compute_eigenvalue_outer_product = (
            compute_eigenvalue_outer_product
        )
        self.grad_worker_fraction = grad_worker_fraction
        self.distributed_strategy = distributed_strategy
        self.symmetry_aware = symmetry_aware
        self.grad_scaler = grad_scaler
        self.factor_dtype = factor_dtype
        self.inv_dtype = inv_dtype
        self.skip_layers = skip_layers
        self.update_factors_in_hook = update_factors_in_hook
        self.known_modules = known_modules
        self.loglevel = loglevel

        self.tdc = TorchDistributedCommunicator(
            bucket_cap_mb=self.allreduce_bucket_cap_mb,
        )
        if self.allreduce_bucket_cap_mb > 0:
            self.allreduce_method = AllreduceMethod.ALLREDUCE_BUCKETED
        else:
            self.allreduce_method = AllreduceMethod.ALLREDUCE

        # key: nn.Module, value: KFACLayer
        self.layers: dict[
            torch._C.Module | torch.nn.Module,
            KFACBaseLayer,
        ] = {}
        self.register_model(model)

    def __repr__(self) -> str:
        extra_params = {
            'accumulation_steps': self.accumulation_steps,
            'allreduce_bucket_cap_mb': self.allreduce_bucket_cap_mb,
            'allreduce_method': self.allreduce_method,
            'assignment_strategy': self.assignment_strategy,
            'colocate_factors': self.colocate_factors,
            'compute_method': self.compute_method,
            'compute_eigenvalue_outer_product': self.compute_eigenvalue_outer_product,  # noqa: E501
            'distributed_strategy': self.distributed_strategy,
            'grad_worker_fraction': self.grad_worker_fraction,
            'symmetry_aware': self.symmetry_aware,
            'grad_scaler': True if self.grad_scaler is not None else False,
            'factor_dtype': self.factor_dtype,
            'inv_dtype': self.inv_dtype,
            'known_modules': self.known_modules,
            'skip_layers': self.skip_layers,
            'update_factors_in_hook': self.update_factors_in_hook,
            'loglevel': self.loglevel,
            'registered_layers': len(self.layers),
        }
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups + [extra_params]):
            format_string += '\n'
            format_string += f'Parameter Group {i}\n'
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += f'    {key}: {group[key]}\n'
        format_string += ')'
        return format_string

    @property
    def damping(self) -> float:
        damping = self.param_groups[0]['damping']
        return damping() if callable(damping) else damping

    @property
    def factor_decay(self) -> float:
        factor_decay = self.param_groups[0]['factor_decay']
        return factor_decay() if callable(factor_decay) else factor_decay

    @property
    def kl_clip(self) -> float:
        kl_clip = self.param_groups[0]['kl_clip']
        return kl_clip() if callable(kl_clip) else kl_clip

    @property
    def lr(self) -> float:
        lr = self.param_groups[0]['lr']
        return lr() if callable(lr) else lr

    @property
    def factor_update_steps(self) -> int:
        factor_update_steps = self.param_groups[0]['factor_update_steps']
        return (
            factor_update_steps()
            if callable(factor_update_steps)
            else factor_update_steps
        )

    @property
    def inv_update_steps(self) -> int:
        inv_update_steps = self.param_groups[0]['inv_update_steps']
        return (
            inv_update_steps()
            if callable(inv_update_steps)
            else inv_update_steps
        )

    @property
    def steps(self) -> int:
        return self.param_groups[0]['steps']

    @steps.setter
    def steps(self, value: int) -> None:
        self.param_groups[0]['steps'] = value

    def state_dict(self, include_factors: bool = True) -> dict[str, Any]:
        """Returns KFAC state dict.

        Args:
          include_factors (optional, bool): include tensors with factors
              for all registered KFACLayers as a part of the state_dict. Note:
              can make the state_dict fairly large, but not saving the factors
              can cause issues if the first iteration when resuming from a
              checkpoint is not a KFAC update step (default: True).
        """
        state_dict = super().state_dict()
        # Remove parameters that are callables because pickling could fail
        # and mini_step because training should not resume in middle of
        # gradient accumulation but may end in middle of gradient accumulation
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not callable(value)
        }
        if include_factors:
            state_dict['layers'] = [
                layer.state_dict() for layer in self.layers.values()
            ]
        return state_dict

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        compute_inverses: bool = True,
    ) -> None:
        """Loads the KFAC state.

        Args:
          state_dict (dict): KFAC state. Should be an object returned from a
              call to `state_dict`.
          compute_inverses (bool, optional): if True, compute the inverses
              from the loaded factors. Note that not computing the inverses
              when loading from a checkpoint will only work if the first
              iteration is a KFAC update step (default: True).
        """
        if 'layers' in state_dict:
            if len(state_dict['layers']) != len(self.layers):
                raise ValueError(
                    'loaded state dict contains a different '
                    'number of layers',
                )
            for layer, layer_state in zip(
                self.layers.values(),
                state_dict['layers'],
            ):
                layer.load_state_dict(layer_state)
            del state_dict['layers']
        else:
            warnings.warn(
                'Layer factors are not included in the state_dict so '
                'inverses cannot be computed. Skipping inverse '
                'computation.',
            )
            compute_inverses = False  # Cannot be computed if no layers
        super().load_state_dict(state_dict)
        if compute_inverses:
            self._assign_workers()
            self.workers_assigned = True
            for layer in self.layers.values():
                layer.compute_a_inv(damping=self.damping)
                layer.compute_g_inv(damping=self.damping)
                if self.distributed_strategy in [
                    DistributedStrategy.COMM_OPT,
                    DistributedStrategy.HYBRID_OPT,
                ]:
                    layer.broadcast_a_inv(src=0)
                    layer.broadcast_g_inv(src=0)

    def register_module(
        self,
        module: torch.nn.Module,
        name: str | None = None,
    ) -> None:
        """Create and register a KFAC layer for a module.

        Note: For a single module, there may be multiple KFAC layers
          registered. E.g. kfac.modules.LSTMCell is made up of two
          torch.nn.Linear so both Linear modules will have a registered KFAC
          Layer.
        """
        kwargs = {
            'allreduce_method': self.allreduce_method,
            'factor_dtype': self.factor_dtype,
            'grad_scaler': self.grad_scaler,
            'inv_dtype': self.inv_dtype,
            'symmetry_aware': self.symmetry_aware,
        }
        if self.compute_method == ComputeMethod.EIGEN:
            kwargs[
                'prediv_eigenvalues'
            ] = self.compute_eigenvalue_outer_product

        layer_list = get_kfac_layers(
            module,
            method=self.compute_method,
            tdc=self.tdc,
            **kwargs,
        )
        for module, kfac_layer in layer_list:
            if dist.get_rank() == 0:
                logger.log(
                    self.loglevel,
                    'Registered {}: {}'.format(
                        name if name is not None else '',
                        kfac_layer.__class__.__name__,
                    ),
                )
            self.layers[module] = kfac_layer
            module.register_forward_pre_hook(self._save_input)
            module.register_full_backward_hook(self._save_grad_output)

    def register_submodules(
        self,
        parent_module: torch.nn.Module,
        prefix: str = '',
    ) -> None:
        """Iterate over and register submodules that KFAC supports."""
        for name, module in parent_module.named_children():
            name = prefix + ('.' if prefix != '' else '') + name
            module_name = module.__class__.__name__.lower()
            if module_name in self.skip_layers:
                pass
            elif module_name not in self.known_modules:
                self.register_submodules(module, prefix=name)
            elif module_requires_grad(module) and module not in self.layers:
                self.register_module(module, name)

    def register_model(self, model: torch.nn.Module) -> None:
        """Registers a model to KFAC."""
        if len(list(model.children())) == 0:  # Handle if model is a module
            if (
                model.__class__.__name__.lower() in self.known_modules
                and model.__class__.__name__.lower() not in self.skip_layers
            ):
                self.register_module(model)
        else:
            self.register_submodules(model)

    @torch.no_grad()
    def step(self, closure: Callable[..., Any] | None = None) -> None:
        """Perform one K-FAC step

        Note:
        - This function should always be called before `optimizer.step()` as
          it modifies the gradients and does not modify the weights.
        - Gradients must be averaged across ranks before calling `step()`.
          This condition is guarenteed to be true if using the
          `DistributedDataParallel` model wrapper as gradients are
          communicated during `loss.backward()`.

        Args:
          closure: for compatibility with the base optimizer class
        """
        if closure is not None:
            raise ValueError('KFAC does not support closures')

        if (
            not self.update_factors_in_hook
            and self.steps % self.factor_update_steps == 0
        ):
            for module in reversed(self.layers):
                self.mini_steps[module] = 0
                self.layers[module].update_a_factor(alpha=self.factor_decay)
                self.layers[module].reduce_a_factor()
                self.layers[module].update_g_factor(alpha=self.factor_decay)
                self.layers[module].reduce_g_factor()

        # Flush last allreduce bucket from forward/backward pass.
        # Will be a no-op if bucketing was not used
        self.tdc.flush_allreduce_buckets()

        # Compute Inverses
        if self.steps % self.inv_update_steps == 0:
            for layer in reversed(self.layers.values()):
                layer.compute_a_inv(damping=self.damping)
                if (
                    self.distributed_strategy
                    is not DistributedStrategy.MEM_OPT
                ):
                    layer.broadcast_a_inv(src=0)
                layer.compute_g_inv(damping=self.damping)
                if (
                    self.distributed_strategy
                    is not DistributedStrategy.MEM_OPT
                ):
                    layer.broadcast_g_inv(src=0)
            self.tdc.flush_allreduce_buckets()

        # Compute Preconditioned Gradients
        for layer in reversed(self.layers.values()):
            layer.preconditioned_grad(damping=self.damping)
            if self.distributed_strategy is not DistributedStrategy.COMM_OPT:
                layer.broadcast_grad(src=0)
        self.tdc.flush_allreduce_buckets()

        scale = None if self.kl_clip is None else self._compute_grad_scale()

        # Update gradients in-place
        for layer in reversed(self.layers.values()):
            layer.update_grad(scale=scale)

        self.steps += 1
        self.mini_steps = defaultdict(int)

    def reset_batch(self) -> None:
        """Reset all KFAC data from last batch."""
        for layer in self.layers.values():
            layer.reset_batch()

    def memory_usage(self) -> dict[str, int]:
        """Returns current approximate memory usage for KFAC on this worker

        Returns:
            dict containing bytes used across all layers on this worker to
            store the factors and second-order information as well as the
            total.
        """
        sizes: dict[str, int] = defaultdict(int)

        # Need to flush buffered communication operations in case user
        # calls this method at a strange time (e.g., in the middle of
        # KFAC.step())
        self.tdc.flush_allreduce_buckets()
        for layer in self.layers.values():
            layer_sizes = layer.memory_usage()
            for key, size in layer_sizes.items():
                sizes[key] += size

        sizes['total'] = sum(sizes.values())
        return sizes

    def _compute_grad_scale(self) -> float | None:
        """Computes scale factor for preconditioned gradients

        Returns:
          sum_{layers} (sum_{gradients} precon_grad * grad * lr^2)
        """
        vg_sum = 0.0
        for layer in reversed(self.layers.values()):
            if layer.grad is None:
                raise RuntimeError(
                    'layer gradient has not been preconditioned',
                )
            w = layer.module.get_weight_grad()
            if layer.module.has_bias():
                b = layer.module.get_bias_grad()
                v1 = layer.grad[:, :-1].view(w.size())
                v2 = layer.grad[:, -1:].view(b.size())
            else:
                v1 = layer.grad.view(w.size())
            vg_sum += (v1 * w * self.lr**2).sum().item()
            if layer.module.has_bias():
                vg_sum += (v2 * b * self.lr**2).sum().item()
        if vg_sum == 0.0:
            return None
        return min(1.0, math.sqrt(self.kl_clip / abs(vg_sum)))

    @torch.no_grad()
    def _save_input(
        self,
        module: torch.nn.Module,
        input: list[torch.Tensor],
    ) -> None:
        if not module.training:
            return
        if self.steps % self.factor_update_steps == 0:
            self.layers[module].save_layer_input(input)
            # Update mini_step here because forward pass should always
            # happen before backward pass
            self.mini_steps[module] += 1
            if (
                self.update_factors_in_hook
                and self.mini_steps[module] % self.accumulation_steps == 0
            ):
                self.layers[module].update_a_factor(alpha=self.factor_decay)
                self.layers[module].reduce_a_factor()

    @torch.no_grad()
    def _save_grad_output(
        self,
        module: torch.nn.Module,
        grad_input: tuple[torch.Tensor, ...] | torch.Tensor,
        grad_output: tuple[torch.Tensor, ...] | torch.Tensor,
    ) -> None:
        if not module.training:
            return
        if self.steps % self.factor_update_steps == 0:
            if isinstance(grad_output, torch.Tensor):
                grad_output = (grad_output,)
            self.layers[module].save_layer_grad_output(grad_output)
            if (
                self.update_factors_in_hook
                and self.mini_steps[module] % self.accumulation_steps == 0
            ):
                self.layers[module].update_g_factor(alpha=self.factor_decay)
                self.layers[module].reduce_g_factor()

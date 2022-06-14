"""KFAC Preconditioner for GPT-NeoX."""
from __future__ import annotations

import logging
from typing import Any
from typing import Callable

import torch

from kfac.base_preconditioner import BaseKFACPreconditioner
from kfac.distributed import get_rank
from kfac.distributed import get_world_size
from kfac.distributed import TorchDistributedCommunicator
from kfac.enums import AllreduceMethod
from kfac.enums import AssignmentStrategy
from kfac.enums import ComputeMethod
from kfac.gpt_neox.assignment import GPTNeoXAssignment
from kfac.gpt_neox.layer import GPTNeoXKFACEigenLayer
from kfac.gpt_neox.modules import GPTNeoXLinearModuleHelper
from kfac.layers.base import KFACBaseLayer
from kfac.layers.register import get_flattened_modules
from kfac.layers.register import requires_grad

try:
    from deepspeed.pipe import PipelineModule

    deepspeed_import_error = None
except ImportError as e:  # pragma: no cover
    deepspeed_import_error = e

logger = logging.getLogger(__name__)


class GPTNeoXKFACPreconditioner(BaseKFACPreconditioner):
    """KFAC Distributed Gradient Preconditioner for GPT-NeoX.

    Integrates with DeepSpeed's PipelineModule that is used to enable
    3D parallelism in GPT-NeoX training.

    Implements the memory-optimized preconditioning scheme (gradient worker
    fraction of 1/world_size).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        factor_update_steps: Callable[[int], int] | int = 1,
        inv_update_steps: Callable[[int], int] | int = 1,
        # KFAC hyperparameters
        damping: Callable[[int], float] | float = 0.001,
        factor_decay: Callable[[int], float] | float = 0.95,
        kl_clip: Callable[[int], float] | float = 0.001,
        lr: Callable[[int], float] | float = 0.1,
        # Distribution strategy
        accumulation_steps: int = 1,
        allreduce_bucket_cap_mb: float = 25.0,
        assignment_strategy: (
            AssignmentStrategy | str
        ) = AssignmentStrategy.COMPUTE,
        compute_method: ComputeMethod | str = ComputeMethod.EIGEN,
        compute_eigenvalue_outer_product: bool = False,
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
        """Init KFACPreconditioner.

        Args:
            model (torch.nn.Module): model to precondition with KFAC.
            factor_update_steps (Callable, int): steps between computing and
                updating the running average of the Kronecker factors or
                callable that takes the K-FAC step and returns the value.
            inv_update_steps (Callble, int): steps between recomputing and
                communicating the second-order information or callable that
                takes the K-FAC step and returns the value.
            damping (Callable, float): Tikhonov damping parameter or a callable
                that takes the K-FAC step and returns the damping parameter
                as a float (default: 0.001).
            factor_decay (Callable, float): running average coefficient for
                Kronecker factors or callable that takes the K-FAC step and
                returns the factor_decay (default: 0.95).
            kl_clip (Callable, float): clipping parameter for gradient scaling
                or a callable that takes the K-FAC step and returns a float.
                If None, no scaling/clipping will be applied (default: 0.001).
            lr (Callable, float): learning rate or callable that takes the
                K-FAC step and returns learning rate (default: 0.1).
            accumulation_steps (int): number of forward/backward passes
                between optimization steps (default: 1).
            allreduce_bucket_cap_mb (float): maximum size in megabytes for
                allreduce bucketing. If zero, bucketing is not used
                (default: 25).
            assignment_strategy (AssignmentStrategy, str): See
                `AssignmentStrategy` for more details
                (default: AssignmentStrategy.COMPUTE).
            compute_method (ComputeMethod, str): See `ComputeMethod` for more
                details (default: ComputeMethod.EIGEN).
            compute_eigenvalue_outer_product (bool): when using the eigen
                compute method, precompute the element-wise inverse of the
                outer product of eigenvectors on the eigen decomposition worker
                rather to reduce computation in the gradient preconditioning
                stage. `colocate_factors` must be True (default: True).
            symmetry_aware (bool): communicate only the upper triangle of
                symmetric matrices. Can reduce communication time when factors
                are large (default: False).
            grad_scaler (torch.cuda.amp.GradScaler or callable): Gradient
                scaler used for Torch AMP training. Used to unscale the G
                factors as they are accumulated during the backward pass.
                Alternatively can be a callable which will return the current
                scale (default: None).
            factor_dtype (torch.dtype): force data type for storing factors.
                If None, defaults to data type of intermediate values in
                forward/backward pass (default: None).
            inv_dtype (torch.dtype): force data type for storing second-order
                data (e.g., inverses or eigen decompositions)
                (default: torch.float32).
            skip_layers (list): list of module names to ignore when registering
                layers. Passing the name of parent modules will prevent
                recursively registering child modules of the parent.
                Case-insensitive (default: []).
            update_factors_in_hook (bool): If True, running average of factors
                is updated in the module hook and the async commmunication is
                started. Otherwise, this will be performed at the start of
                step() (default: True).
            loglevel (int): logging level (default: logging.DEBUG).
        """
        if deepspeed_import_error is not None:  # pragma: no cover
            raise deepspeed_import_error

        if not isinstance(model, PipelineModule):
            raise ValueError(
                'model must be an instance of deepspeed.pipe.PipelineModule. '
                f'Got an instance of {type(model)}.',
            )

        if allreduce_bucket_cap_mb < 0:
            raise ValueError('allreduce_bucket_cap_mb must be >= 0')
        if isinstance(assignment_strategy, str):
            assignment_strategy = AssignmentStrategy[
                assignment_strategy.upper()
            ]
        if isinstance(compute_method, str):
            compute_method = ComputeMethod[compute_method.upper()]

        self.allreduce_bucket_cap_mb = allreduce_bucket_cap_mb
        self.assignment_strategy = assignment_strategy
        self.compute_eigenvalue_outer_product = (
            compute_eigenvalue_outer_product
        )
        self.compute_method = compute_method
        self.grad_scaler = grad_scaler
        self.factor_dtype = factor_dtype
        self.inv_dtype = inv_dtype
        self.skip_layers = [] if skip_layers is None else skip_layers
        self.symmetry_aware = symmetry_aware

        if self.allreduce_bucket_cap_mb > 0:
            self.allreduce_method = AllreduceMethod.ALLREDUCE_BUCKETED
        else:
            self.allreduce_method = AllreduceMethod.ALLREDUCE
        self.tdc = TorchDistributedCommunicator(
            bucket_cap_mb=self.allreduce_bucket_cap_mb,
        )

        layer_kwargs = dict(
            allreduce_method=self.allreduce_method,
            grad_scaler=self.grad_scaler,
            factor_dtype=self.factor_dtype,
            inv_dtype=self.inv_dtype,
            symmetry_aware=self.symmetry_aware,
            tdc=self.tdc,
        )

        if self.compute_method == ComputeMethod.EIGEN:
            pass
        elif self.compute_method == ComputeMethod.INVERSE:
            raise ValueError('Inverse method not supported with GPT NeoX.')
        else:
            raise AssertionError(f'Unknown {self.compute_method=}')

        kfac_layers = register_modules(
            model,
            model.mpu().get_model_parallel_group(),
            skip_layers=self.skip_layers,
            **layer_kwargs,
        )

        model_parallel_size = get_world_size() // (
            model.mpu().get_data_parallel_world_size()
            * model.mpu().get_pipe_parallel_world_size()
        )
        if model_parallel_size != 1:
            raise ValueError(
                'GPTNeoXKFACPreconditioner only supports model parallelism '
                'of 1 currently.',
            )

        data_parallel_ranks = [
            -1 for _ in range(model.mpu().get_data_parallel_world_size())
        ]
        torch.distributed.all_gather_object(
            object_list=data_parallel_ranks,
            obj=get_rank(),
            group=model.mpu().get_data_parallel_group(),
        )

        for name, kfac_layer in kfac_layers.values():
            logger.log(
                loglevel,
                f'Registered name="{name}": {repr(kfac_layer)} on '
                f'global-rank={get_rank()} and '
                f'pipeline-rank={model.mpu().get_pipe_parallel_rank()}',
            )

        if self.assignment_strategy == AssignmentStrategy.COMPUTE:
            cost_func = lambda n: n**3  # noqa: E731
        elif self.assignment_strategy == AssignmentStrategy.MEMORY:
            cost_func = lambda n: n**2  # noqa: E731
        else:
            raise AssertionError(
                f'Unknown assignment_strategy={self.assignment_strategy}',
            )

        work = {
            name: {
                'A': cost_func(kfac_layer.module.a_factor_shape[0]),
                'G': cost_func(kfac_layer.module.g_factor_shape[0]),
            }
            for name, kfac_layer in kfac_layers.values()
        }
        assignment = GPTNeoXAssignment(
            work,
            local_rank=get_rank(),
            data_parallel_ranks=data_parallel_ranks,
            data_parallel_group=model.mpu().get_data_parallel_group(),
        )
        logger.log(loglevel, f'KFAC layer assignments: {assignment}')

        defaults = {
            'allreduce_bucket_cap_mb': self.allreduce_bucket_cap_mb,
            'allreduce_method': self.allreduce_method,
            'assignment_strategy': self.assignment_strategy,
            'compute_eigenvalue_outer_product': (
                self.compute_eigenvalue_outer_product
            ),
            'compute_method': self.compute_method,
            'grad_scaler': self.grad_scaler is not None,
            'factor_dtype': self.factor_dtype,
            'inv_dtype': self.inv_dtype,
            'skip_layers': self.skip_layers,
            'symmetry_aware': self.symmetry_aware,
        }

        super().__init__(
            kfac_layers,
            factor_update_steps=factor_update_steps,
            inv_update_steps=inv_update_steps,
            factor_decay=factor_decay,
            damping=damping,
            kl_clip=kl_clip,
            lr=lr,
            accumulation_steps=accumulation_steps,
            assignment=assignment,
            update_factors_in_hook=update_factors_in_hook,
            defaults=defaults,
            tdc=self.tdc,
            loglevel=loglevel,
        )


def register_modules(
    model: torch.nn.Module,
    model_parallel_group: torch.distributed.ProcessGroup,
    skip_layers: list[str],
    **layer_kwargs: Any,
) -> dict[torch.nn.Module, tuple[str, KFACBaseLayer]]:
    """Register supported modules in model with a KFACLayer.

    Args:
        model (torch.nn.Module): model to scan for modules to register.
        model_parallel_group (ProcessGroup): model parallelism group this
            rank belongs to.
        skip_layers (list[str]): names of layers to skip registering. Names
            can either by the name of the attribute or the name of the
            class of the layer. Matches are case insensitive.
        **layer_kwargs (dict[str, Any]): optional keyword arguments to
            pass to the kfac_layer_type constructor.
    """
    if deepspeed_import_error is not None:  # pragma: no cover
        raise deepspeed_import_error

    modules = get_flattened_modules(model)
    skip_layers = [s.lower() for s in skip_layers]

    kfac_layers: dict[torch.nn.Module, tuple[str, KFACBaseLayer]] = {}
    for name, module in modules:
        if (
            name.lower() not in skip_layers
            and module.__class__.__name__.lower() not in skip_layers
            and requires_grad(module)
        ):
            if module.__class__.__name__.lower() not in (
                'ColumnParallelLinear'.lower(),
                'RowParallelLinear'.lower(),
            ):
                continue

            kfac_layer = GPTNeoXKFACEigenLayer(
                GPTNeoXLinearModuleHelper(module, model_parallel_group),
                **layer_kwargs,
            )

            # get_flattened_modules() should never give us modules with the
            # same name
            assert module not in kfac_layers
            kfac_layers[module] = (name, kfac_layer)

    return kfac_layers

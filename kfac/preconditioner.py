"""Implementation of the KAISA preconditioner."""

from __future__ import annotations

import logging
import warnings
from typing import Callable
from typing import cast
from typing import List

import torch
import torch.distributed as dist

from kfac.assignment import KAISAAssignment
from kfac.base_preconditioner import BaseKFACPreconditioner
from kfac.distributed import get_rank
from kfac.distributed import get_world_size
from kfac.distributed import TorchDistributedCommunicator
from kfac.enums import AllreduceMethod
from kfac.enums import AssignmentStrategy
from kfac.enums import ComputeMethod
from kfac.enums import DistributedStrategy
from kfac.layers.base import KFACBaseLayer
from kfac.layers.eigen import KFACEigenLayer
from kfac.layers.inverse import KFACInverseLayer
from kfac.layers.register import register_modules

logger = logging.getLogger(__name__)


def _mock_new_group(x: list[int]) -> None:
    return None


class KFACPreconditioner(BaseKFACPreconditioner):
    """KFAC Distributed Gradient Preconditioner.

    Implements the KAISA preconditioning strategy where gradient workers and
    receivers are assigned based on the gradient worker fraction.

    Example:
        >>> model = torch.nn.parallel.DistributedDataParallel(model, ...)
        >>> optimizer = optim.SGD(model.parameters(), ...)
        >>> preconditioner = kfac.preconditioner.KFACPreconditioner(model, ...)
        >>>
        >>> for i, (data, target) in enumerate(train_loader):
        >>>     optimizer.zero_grad()
        >>>     output = model(data)
        >>>     loss = criterion(output, target)
        >>>     loss.backward()
        >>>     preconditioner.step()
        >>>     optimizer.step()
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
            colocate_factors (bool): assign both factors for a single layer to
                the same worker. Recommended when num_layers < world_size
                (default: True).
            compute_method (ComputeMethod, str): See `ComputeMethod` for more
                details (default: ComputeMethod.EIGEN).
            compute_eigenvalue_outer_product (bool): when using the eigen
                compute method, precompute the element-wise inverse of the
                outer product of eigenvectors on the eigen decomposition worker
                rather to reduce computation in the gradient preconditioning
                stage. `colocate_factors` must be True (default: True).
            grad_worker_fraction (DistributedStrategy, float): controls the
                fraction of workers assigned as gradient workers for each
                layer. Optionally, predefined configurations can be passed
                using the DistributedStrategy enum
                (default: DistributedStrategy.COMM_OPT).
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
            skip_layers (list[str]): regex patterns that if matched, will cause
                the layer to not be registered. The patterns will be applied
                against the layer's name and class name.
            update_factors_in_hook (bool): If True, running average of factors
                is updated in the module hook and the async communication is
                started. Otherwise, this will be performed at the start of
                step() (default: True).
            loglevel (int): logging level (default: logging.DEBUG).
        """
        if allreduce_bucket_cap_mb < 0:
            raise ValueError('allreduce_bucket_cap_mb must be >= 0')
        if (
            compute_method == ComputeMethod.EIGEN
            and compute_eigenvalue_outer_product
            and not colocate_factors
        ):
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

        size = get_world_size()
        if isinstance(grad_worker_fraction, DistributedStrategy):
            distributed_strategy = grad_worker_fraction
            if distributed_strategy == DistributedStrategy.COMM_OPT:
                grad_worker_fraction = 1.0
            elif distributed_strategy == DistributedStrategy.HYBRID_OPT:
                grad_worker_fraction = 0.5
            elif distributed_strategy == DistributedStrategy.MEM_OPT:
                grad_worker_fraction = 1.0 / size
            else:
                raise AssertionError(f'Unknown enum {grad_worker_fraction}')
        else:
            if not 0 <= grad_worker_fraction or not 1 >= grad_worker_fraction:
                raise ValueError('grad_worker_fraction must in [0, 1]')
            if grad_worker_fraction == 0:
                grad_worker_fraction = 1.0 / size
            if size % max(1, round(size * grad_worker_fraction)) != 0:
                raise ValueError(
                    'grad_worker_fraction must produce groups of equal size',
                )
            if grad_worker_fraction == 1:
                grad_worker_fraction = 1.0  # ensure float
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
                stacklevel=2,
            )
            colocate_factors = True

        self.allreduce_bucket_cap_mb = allreduce_bucket_cap_mb
        self.assignment_strategy = assignment_strategy
        self.colocate_factors = colocate_factors
        self.compute_eigenvalue_outer_product = (
            compute_eigenvalue_outer_product
        )
        self.compute_method = compute_method
        self.distributed_strategy = distributed_strategy
        self.grad_worker_fraction = grad_worker_fraction
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

        layer_type: type[KFACBaseLayer]
        if self.compute_method == ComputeMethod.EIGEN:
            layer_type = KFACEigenLayer
            layer_kwargs['prediv_eigenvalues'] = (
                self.compute_eigenvalue_outer_product
            )
        elif self.compute_method == ComputeMethod.INVERSE:
            layer_type = KFACInverseLayer
        else:
            raise AssertionError(
                f'Unknown compute_method={self.compute_method}',
            )

        kfac_layers = register_modules(
            model,
            kfac_layer_type=layer_type,
            skip_layers=self.skip_layers,
            **layer_kwargs,
        )
        for name, kfac_layer in kfac_layers.values():
            logger.log(
                loglevel,
                f'Registered name="{name}": {repr(kfac_layer)}',
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

        new_group = cast(
            Callable[[List[int]], dist.ProcessGroup],
            dist.new_group,
        )

        assignment = KAISAAssignment(
            work,
            local_rank=get_rank(),
            world_size=get_world_size(),
            grad_worker_fraction=self.grad_worker_fraction,
            group_func=new_group if dist.is_initialized() else _mock_new_group,
            colocate_factors=self.colocate_factors,
        )
        logger.log(loglevel, f'KFAC layer assignments: {assignment}')

        defaults = {
            'allreduce_bucket_cap_mb': self.allreduce_bucket_cap_mb,
            'allreduce_method': self.allreduce_method,
            'assignment_strategy': self.assignment_strategy,
            'colocate_factors': self.colocate_factors,
            'compute_eigenvalue_outer_product': (
                self.compute_eigenvalue_outer_product
            ),
            'compute_method': self.compute_method,
            'distributed_strategy': self.distributed_strategy,
            'grad_worker_fraction': self.grad_worker_fraction,
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

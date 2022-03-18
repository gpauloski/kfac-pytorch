from __future__ import annotations

from typing import Callable

import torch

from kfac.preconditioner import AssignmentStrategy
from kfac.preconditioner import ComputeMethod
from kfac.preconditioner import DistributedStrategy
from kfac.preconditioner import KFACPreconditioner

try:
    import deepspeed  # type: ignore
    from deepspeed.runtime.engine import DeepSpeedEngine  # type: ignore
except ImportError as e:
    deepspeed = e


class KFACDeepSpeedPreconditioner(KFACPreconditioner):
    """KFAC DeepSpeed Integration"""

    def __init__(
        self,
        engine: DeepSpeedEngine,
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
        inv_dtype: torch.dtype | None = None,
        skip_layers: list[str] | None = None,
        update_factors_in_hook: bool = True,
        verbose: bool = False,
    ) -> None:
        if isinstance(deepspeed, ImportError):
            raise deepspeed

        if not isinstance(engine, DeepSpeedEngine):
            raise TypeError(
                f'engine must be of type DeepSpeedEngine. Got {type(engine)}.',
            )

        self.engine = engine

        super().__init__(
            model=engine.module,
            factor_update_steps=factor_update_steps,
            inv_update_steps=inv_update_steps,
            damping=damping,
            factor_decay=factor_decay,
            kl_clip=kl_clip,
            lr=lr,
            accumulation_steps=accumulation_steps,
            allreduce_bucket_cap_mb=allreduce_bucket_cap_mb,
            assignment_strategy=assignment_strategy,
            colocate_factors=colocate_factors,
            compute_method=compute_method,
            compute_eigenvalue_outer_product=compute_eigenvalue_outer_product,
            grad_worker_fraction=grad_worker_fraction,
            symmetry_aware=symmetry_aware,
            grad_scaler=grad_scaler,
            factor_dtype=factor_dtype,
            inv_dtype=inv_dtype,
            skip_layers=skip_layers,
            update_factors_in_hook=update_factors_in_hook,
            verbose=verbose,
        )

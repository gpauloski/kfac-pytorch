import kfac
from kfac.distributed import
from kfac.layers.base import AllreduceMethod
from kfac.layers.base import BroadcastMethod

try:
    import deepspeed
    from deepspeed.runtime.engine import DeepSpeedEngine
except ImportError as e:
    deepspeed = e


class KFAC(kfac.KFAC):
    """KFAC DeepSpeed Integration"""

    def __init__(
        self,
        engine: DeepSpeedEngine,
        factor_update_steps: int,
        inv_update_steps: int,
        # KFAC hyperparameters
        damping: Union[Callable, float] = 0.001,
        factor_decay: Union[Callable, float] = 0.95,
        kl_clip: Union[Callable, float] = 0.001,
        lr: Union[Callable, float] = 0.1,
        # Distribution strategy
        accumulation_steps: int = 1,
        allreduce_bucket_cap_mb: float = 25.0,
        assignment_strategy: Union[
            kfac.AssignmentStrategy,
            str,
        ] = kfac.AssignmentStrategy.COMPUTE,
        colocate_factors: bool = True,
        compute_method: Union[kfac.ComputeMethod, str] = kfac.ComputeMethod.EIGEN,
        compute_eigenvalue_outer_product: bool = True,
        grad_worker_fraction: Union[
            kfac.DistributedStrategy,
            float,
        ] = kfac.DistributedStrategy.COMM_OPT,
        symmetry_aware: bool = False,
        # Optional other parameters
        grad_scaler: Optional[
            Union[torch.cuda.amp.GradScaler, Callable]
        ] = None,
        factor_dtype: Optional[torch.dtype] = None,
        inv_dtype: Optional[torch.dtype] = None,
        skip_layers: List[str] = [],
        update_factors_in_hook: bool = True,
        verbose=False,
    ) -> None:
        if isinstance(deepspeed, ImportError):
            raise deepspeed

        if not isinstance(engine, DeepSpeedEngine):
            raise TypeError(
                f'engine must be of type DeepSpeedEngine. Got {type(engine)}.'
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

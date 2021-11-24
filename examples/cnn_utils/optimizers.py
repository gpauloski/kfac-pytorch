import sys
import kfac
import torch.distributed as dist
import torch.optim as optim

sys.path.append("..")
from utils import create_lr_schedule  # noqa: E402


def get_optimizer(model, args):
    use_kfac = True if args.kfac_inv_update_steps > 0 else False

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if args.kfac_strategy == "comm-opt":
        grad_worker_fraction = kfac.DistributedStrategy.COMM_OPT
    elif args.kfac_strategy == "mem-opt":
        grad_worker_fraction = kfac.DistributedStrategy.MEM_OPT
    elif args.kfac_strategy == "hybrid-opt":
        grad_worker_fraction = args.kfac_grad_worker_fraction
    else:
        raise ValueError(
            "Unknown KFAC Comm Method: {}".format(args.kfac_strategy)
        )

    if use_kfac:
        preconditioner = kfac.KFAC(
            model,
            factor_update_steps=args.kfac_factor_update_steps,
            inv_update_steps=args.kfac_inv_update_steps,
            damping=args.kfac_damping,
            factor_decay=args.kfac_factor_decay,
            kl_clip=args.kfac_kl_clip,
            lr=args.base_lr,
            accumulation_steps=args.batches_per_allreduce,
            colocate_factors=args.kfac_colocate_factors,
            compute_method=kfac.ComputeMethod.INVERSE
            if args.kfac_inv_method
            else kfac.ComputeMethod.EIGEN,
            grad_worker_fraction=grad_worker_fraction,
            grad_scaler=args.grad_scaler if "grad_scaler" in args else None,
            skip_layers=args.kfac_skip_layers,
        )
        kfac_param_scheduler = kfac.KFACParamScheduler(
            preconditioner,
            damping_alpha=args.kfac_damping_alpha,
            damping_schedule=args.kfac_damping_decay,
            update_freq_alpha=args.kfac_update_steps_alpha,
            update_freq_schedule=args.kfac_update_steps_decay,
        )
    else:
        preconditioner = None

    if args.horovod:
        import horovod.torch as hvd

        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            compression=hvd.Compression.none,
            op=hvd.Average,
            backward_passes_per_step=args.batches_per_allreduce,
        )

        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    lrs = create_lr_schedule(
        dist.get_world_size(), args.warmup_epochs, args.lr_decay
    )
    lr_scheduler = [optim.lr_scheduler.LambdaLR(optimizer, lrs)]
    if use_kfac:
        lr_scheduler.append(optim.lr_scheduler.LambdaLR(preconditioner, lrs))
        lr_scheduler.append(kfac_param_scheduler)

    return optimizer, preconditioner, lr_scheduler

import sys
import kfac
import torch.distributed as dist
import torch.optim as optim

sys.path.append("..")
from utils import create_lr_schedule  # noqa: E402


def get_optimizer(model, args):
    use_kfac = True if args.kfac_update_freq > 0 else False

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if args.kfac_comm_method == "comm-opt":
        grad_worker_fraction = kfac.DistributedStrategy.COMM_OPT
    elif args.kfac_comm_method == "mem-opt":
        grad_worker_fraction = kfac.DistributedStrategy.MEM_OPT
    elif args.kfac_comm_method == "hybrid-opt":
        grad_worker_fraction = args.kfac_grad_worker_fraction
    else:
        raise ValueError(
            "Unknwon KFAC Comm Method: {}".format(args.kfac_comm_method)
        )

    if use_kfac:
        preconditioner = kfac.KFAC(
            model,
            factor_update_steps=args.kfac_cov_update_freq,
            inv_update_steps=args.kfac_update_freq,
            damping=args.damping,
            factor_decay=args.stat_decay,
            kl_clip=args.kl_clip,
            lr=args.base_lr,
            accumulation_steps=args.batches_per_allreduce,
            colocate_factors=args.coallocate_layer_factors,
            compute_method=kfac.ComputeMethod.INVERSE
            if args.use_inv_kfac
            else kfac.ComputeMethod.EIGEN,
            grad_worker_fraction=grad_worker_fraction,
            grad_scaler=args.grad_scaler if "grad_scaler" in args else None,
            skip_layers=args.skip_layers,
        )
        kfac_param_scheduler = kfac.KFACParamScheduler(
            preconditioner,
            damping_alpha=args.damping_alpha,
            damping_schedule=args.damping_decay,
            update_freq_alpha=args.kfac_update_freq_alpha,
            update_freq_schedule=args.kfac_update_freq_decay,
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

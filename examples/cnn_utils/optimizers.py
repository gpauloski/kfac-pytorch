import sys
import kfac
import torch.optim as optim

sys.path.append('..')
from utils import create_lr_schedule

def get_optimizer(model, args, batch_first=True):
    use_kfac = True if args.kfac_update_freq > 0 else False

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr, 
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    if args.kfac_comm_method == 'comm-opt':
        comm_method=kfac.CommMethod.COMM_OPT
    elif args.kfac_comm_method == 'mem-opt':
        comm_method=kfac.CommMethod.MEM_OPT
    elif args.kfac_comm_method == 'hybrid-opt':
        comm_method=kfac.CommMethod.HYBRID_OPT
    else:
        raise ValueError('Unknwon KFAC Comm Method: {}'.format(
                args.kfac_comm_method))

    if use_kfac:
        preconditioner = kfac.KFAC(
            model, 
            damping=args.damping, 
            factor_decay=args.stat_decay,
            factor_update_freq=args.kfac_cov_update_freq,
            inv_update_freq=args.kfac_update_freq,
            kl_clip=args.kl_clip,
            lr=args.base_lr, 
            batch_first=batch_first,
            comm_method=comm_method,
            distribute_layer_factors=not args.coallocate_layer_factors,
            grad_scaler=args.grad_scaler if 'grad_scaler' in args else None,
            grad_worker_fraction = args.kfac_grad_worker_fraction,
            skip_layers=args.skip_layers,
            use_eigen_decomp=not args.use_inv_kfac,
        ) 
        kfac_param_scheduler = kfac.KFACParamScheduler(
            preconditioner,
            damping_alpha=args.damping_alpha,
            damping_schedule=args.damping_decay,
            update_freq_alpha=args.kfac_update_freq_alpha,
            update_freq_schedule=args.kfac_update_freq_decay
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
            backward_passes_per_step=args.batches_per_allreduce
        )

        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    lrs = create_lr_schedule(args.backend.size(), args.warmup_epochs, args.lr_decay)
    lr_scheduler = [optim.lr_scheduler.LambdaLR(optimizer, lrs)]
    if use_kfac:
        lr_scheduler.append(optim.lr_scheduler.LambdaLR(preconditioner, lrs))
        lr_scheduler.append(kfac_param_scheduler)

    return optimizer, preconditioner, lr_scheduler

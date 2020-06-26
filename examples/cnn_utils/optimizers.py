import sys
import kfac
import torch.optim as optim

sys.path.append('..')
from utils import create_lr_schedule

def get_optimizer(model, args):
    use_kfac = True if args.kfac_update_freq > 0 else False

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr, 
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    if use_kfac:
        preconditioner = kfac.KFAC(
            model, 
            lr=args.base_lr, 
            factor_decay=args.stat_decay,
            damping=args.damping, 
            kl_clip=args.kl_clip,
            fac_update_freq=args.kfac_cov_update_freq,
            kfac_update_freq=args.kfac_update_freq,
            use_eigen_decomp=not args.use_inv_kfac,
            distribute_layer_factors=not args.coallocate_layer_factors
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

    lrs = create_lr_schedule(args.backend.size(), args.warmup_epochs, args.lr_decay)
    lr_scheduler = [optim.lr_scheduler.LambdaLR(optimizer, lrs)]
    if use_kfac:
        lr_scheduler.append(optim.lr_scheduler.LambdaLR(preconditioner, lrs))
        lr_scheduler.append(kfac_param_scheduler)

    return optimizer, preconditioner, lr_scheduler, lrs

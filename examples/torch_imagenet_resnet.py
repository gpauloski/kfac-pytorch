"""ImageNet and ResNet training script."""

from __future__ import annotations

import argparse
import datetime
import os
import time
import warnings

import torch
import torch.distributed as dist
import torchvision.models as models
from torch.utils import collect_env
from torch.utils.tensorboard import SummaryWriter

import examples.vision.datasets as datasets
import examples.vision.engine as engine
import examples.vision.optimizers as optimizers
from examples.utils import LabelSmoothLoss
from examples.utils import save_checkpoint

try:
    from torch.cuda.amp import GradScaler

    TORCH_FP16 = True
except ImportError:
    TORCH_FP16 = False

warnings.filterwarnings('ignore', '(Possibly )?corrupt EXIF data', UserWarning)


def parse_args() -> argparse.Namespace:
    """Get cmd line args."""
    # General settings
    parser = argparse.ArgumentParser(
        description='PyTorch ImageNet Example',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--train-dir',
        default='/tmp/imagenet/ILSVRC2012_img_train/',
        help='path to training data',
    )
    parser.add_argument(
        '--val-dir',
        default='/tmp/imagenet/ILSVRC2012_img_val/',
        help='path to validation data',
    )
    parser.add_argument(
        '--log-dir',
        default='./logs/torch_imagenet',
        help='TensorBoard/checkpoint log directory',
    )
    parser.add_argument(
        '--checkpoint-format',
        default='checkpoint_{epoch}.pth.tar',
        help='checkpoint file format',
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        metavar='S',
        help='random seed (default: 42)',
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        default=False,
        help='use torch.cuda.amp for fp16 training (default: false)',
    )

    # Default settings from https://arxiv.org/abs/1706.02677.
    parser.add_argument(
        '--model',
        default='resnet50',
        help='Model (resnet{35,50,101,152})',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='input batch size for training',
    )
    parser.add_argument(
        '--val-batch-size',
        type=int,
        default=32,
        help='input batch size for validation',
    )
    parser.add_argument(
        '--batches-per-allreduce',
        type=int,
        default=1,
        help='number of batches processed locally before '
        'executing allreduce across workers; it multiplies '
        'total batch size.',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=55,
        help='number of epochs to train (default: 55)',
    )
    parser.add_argument(
        '--base-lr',
        type=float,
        default=0.0125,
        help='learning rate for a single GPU (default: 0.0125)',
    )
    parser.add_argument(
        '--lr-decay',
        nargs='+',
        type=int,
        default=[25, 35, 40, 45, 50],
        help='epoch intervals to decay lr (default: 25,35,40,45,50)',
    )
    parser.add_argument(
        '--warmup-epochs',
        type=float,
        default=5,
        help='number of warmup epochs',
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='SGD momentum',
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.00005,
        help='weight decay',
    )
    parser.add_argument(
        '--label-smoothing',
        type=float,
        default=0.1,
        help='label smoothing (default 0.1)',
    )
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=5,
        help='epochs between checkpoints',
    )

    # KFAC Parameters
    parser.add_argument(
        '--kfac-inv-update-steps',
        type=int,
        default=100,
        help='iters between kfac inv ops (0 disables kfac) (default: 100)',
    )
    parser.add_argument(
        '--kfac-factor-update-steps',
        type=int,
        default=10,
        help='iters between kfac cov ops (default: 10)',
    )
    parser.add_argument(
        '--kfac-update-steps-alpha',
        type=float,
        default=10,
        help='KFAC update freq multiplier (default: 10)',
    )
    parser.add_argument(
        '--kfac-update-steps-decay',
        nargs='+',
        type=int,
        default=None,
        help='KFAC update freq decay schedule (default None)',
    )
    parser.add_argument(
        '--kfac-inv-method',
        action='store_true',
        default=False,
        help='Use inverse KFAC update instead of eigen (default False)',
    )
    parser.add_argument(
        '--kfac-factor-decay',
        type=float,
        default=0.95,
        help='Alpha value for covariance accumulation (default: 0.95)',
    )
    parser.add_argument(
        '--kfac-damping',
        type=float,
        default=0.001,
        help='KFAC damping factor (defaultL 0.001)',
    )
    parser.add_argument(
        '--kfac-damping-alpha',
        type=float,
        default=0.5,
        help='KFAC damping decay factor (default: 0.5)',
    )
    parser.add_argument(
        '--kfac-damping-decay',
        nargs='+',
        type=int,
        default=None,
        help='KFAC damping decay schedule (default None)',
    )
    parser.add_argument(
        '--kfac-kl-clip',
        type=float,
        default=0.001,
        help='KL clip (default: 0.001)',
    )
    parser.add_argument(
        '--kfac-skip-layers',
        nargs='+',
        type=str,
        default=[],
        help='Layer types to ignore registering with KFAC (default: [])',
    )
    parser.add_argument(
        '--kfac-colocate-factors',
        action='store_true',
        default=True,
        help='Compute A and G for a single layer on the same worker. ',
    )
    parser.add_argument(
        '--kfac-strategy',
        type=str,
        default='comm-opt',
        help='KFAC communication optimization strategy. One of comm-opt, '
        'mem-opt, or hybrid_opt. (default: comm-opt)',
    )
    parser.add_argument(
        '--kfac-grad-worker-fraction',
        type=float,
        default=0.25,
        help='Fraction of workers to compute the gradients '
        'when using HYBRID_OPT (default: 0.25)',
    )

    parser.add_argument(
        '--backend',
        type=str,
        default='nccl',
        help='backend for distribute training (default: nccl)',
    )
    # Set automatically by torch distributed launch
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help='local rank for distributed training',
    )

    args = parser.parse_args()
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()

    torch.distributed.init_process_group(
        backend=args.backend,
        init_method='env://',
    )
    torch.distributed.barrier()

    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.manual_seed(args.seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True

    args.base_lr = (
        args.base_lr * dist.get_world_size() * args.batches_per_allreduce
    )
    args.verbose = dist.get_rank() == 0

    if args.verbose:
        print('Collecting env info...')
        print(collect_env.get_pretty_env_info())
        print()

    for r in range(torch.distributed.get_world_size()):
        if r == torch.distributed.get_rank():
            print(
                f'Global rank {torch.distributed.get_rank()} initialized: '
                f'local_rank = {args.local_rank}, '
                f'world_size = {torch.distributed.get_world_size()}',
            )
        torch.distributed.barrier()

    train_sampler, train_loader, _, val_loader = datasets.get_imagenet(args)
    if args.model.lower() == 'resnet50':
        model = models.resnet50()
    elif args.model.lower() == 'resnet101':
        model = models.resnet101()
    elif args.model.lower() == 'resnet152':
        model = models.resnet152()

    device = 'cpu' if not args.cuda else 'cuda'
    model.to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
    )

    os.makedirs(args.log_dir, exist_ok=True)
    args.checkpoint_format = os.path.join(args.log_dir, args.checkpoint_format)
    args.log_writer = SummaryWriter(args.log_dir) if args.verbose else None

    # If set > 0, will resume training from a given checkpoint.
    args.resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            args.resume_from_epoch = try_epoch
            break

    scaler = None
    if args.fp16:
        if not TORCH_FP16:
            raise ValueError(
                'The installed version of torch does not '
                'support torch.cuda.amp fp16 training. This '
                'requires torch version >= 1.16',
            )
        scaler = GradScaler()
    args.grad_scaler = scaler

    (
        optimizer,
        preconditioner,
        (lr_scheduler, kfac_scheduler),
    ) = optimizers.get_optimizer(
        model,
        args,
    )
    if args.verbose and preconditioner is not None:
        print(preconditioner)
    loss_func = LabelSmoothLoss(args.label_smoothing)

    # Restore from a previous checkpoint, if initial_epoch is specified.
    if args.resume_from_epoch > 0:
        filepath = args.checkpoint_format.format(epoch=args.resume_from_epoch)
        map_location = {'cuda:0': f'cuda:{args.local_rank}'}
        checkpoint = torch.load(filepath, map_location=map_location)
        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['lr_scheduler'] is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if (
            checkpoint['preconditioner'] is not None
            and preconditioner is not None
        ):
            preconditioner.load_state_dict(checkpoint['preconditioner'])

    start = time.time()

    for epoch in range(args.resume_from_epoch + 1, args.epochs + 1):
        engine.train(
            epoch,
            model,
            optimizer,
            preconditioner,
            loss_func,
            train_sampler,
            train_loader,
            args,
        )
        engine.test(epoch, model, loss_func, val_loader, args)
        lr_scheduler.step()
        if kfac_scheduler is not None:
            kfac_scheduler.step(step=epoch)
        if (
            epoch > 0
            and epoch % args.checkpoint_freq == 0
            and dist.get_rank() == 0
        ):
            # Note: save model.module b/c model may be Distributed wrapper
            # so saving the underlying model is more generic
            save_checkpoint(
                model.module,
                optimizer,
                preconditioner,
                lr_scheduler,
                args.checkpoint_format.format(epoch=epoch),
            )

    if args.verbose:
        print(
            '\nTraining time: {}'.format(
                datetime.timedelta(seconds=time.time() - start),
            ),
        )

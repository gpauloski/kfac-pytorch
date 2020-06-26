import argparse
import time
import os
import sys
import datetime
import warnings
import kfac
import torch
import torch.distributed as dist

import torchvision.models as models
import cnn_utils.datasets as datasets
import cnn_utils.engine as engine
import cnn_utils.optimizers as optimizers

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from utils import LabelSmoothLoss, save_checkpoint

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

def parse_args():
    # General settings
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train-dir', default='/tmp/imagenet/ILSVRC2012_img_train/',
                        help='path to training data')
    parser.add_argument('--val-dir', default='/tmp/imagenet/ILSVRC2012_img_val/',
                        help='path to validation data')    
    parser.add_argument('--log-dir', default='./logs',
                        help='TensorBoard log directory')
    parser.add_argument('--checkpoint-format', default='checkpoint-{epoch}.pth.tar',
                        help='checkpoint file format')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

   # Default settings from https://arxiv.org/abs/1706.02677.
    parser.add_argument('--model', default='resnet50',
                        help='Model (resnet{35,50,101,152})')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=32,
                        help='input batch size for validation')
    parser.add_argument('--batches-per-allreduce', type=int, default=1,
                        help='number of batches processed locally before '
                             'executing allreduce across workers; it multiplies '
                             'total batch size.')
    parser.add_argument('--epochs', type=int, default=55,
                        help='number of epochs to train (default: 55)')
    parser.add_argument('--base-lr', type=float, default=0.0125,
                        help='learning rate for a single GPU (default: 0.0125)')
    parser.add_argument('--lr-decay', nargs='+', type=int, default=[25,35,40,45,50],
                        help='epoch intervals to decay lr (default: 25,35,40,45,50)')
    parser.add_argument('--warmup-epochs', type=float, default=5,
                        help='number of warmup epochs')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.00005,
                        help='weight decay')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='label smoothing (default 0.1)')

    # KFAC Parameters
    parser.add_argument('--kfac-update-freq', type=int, default=100,
                        help='iters between kfac inv ops (0 disables kfac) (default: 100)')
    parser.add_argument('--kfac-cov-update-freq', type=int, default=10,
                        help='iters between kfac cov ops (default: 10)')
    parser.add_argument('--kfac-update-freq-alpha', type=float, default=10,
                        help='KFAC update freq multiplier (default: 10)')
    parser.add_argument('--kfac-update-freq-decay', nargs='+', type=int, default=None,
                        help='KFAC update freq decay schedule (default None)')
    parser.add_argument('--use-inv-kfac', action='store_true', default=False,
                        help='Use inverse KFAC update instead of eigen (default False)')
    parser.add_argument('--stat-decay', type=float, default=0.95,
                        help='Alpha value for covariance accumulation (default: 0.95)')
    parser.add_argument('--damping', type=float, default=0.001,
                        help='KFAC damping factor (defaultL 0.001)')
    parser.add_argument('--damping-alpha', type=float, default=0.5,
                        help='KFAC damping decay factor (default: 0.5)')
    parser.add_argument('--damping-decay', nargs='+', type=int, default=None,
                        help='KFAC damping decay schedule (default None)')
    parser.add_argument('--kl-clip', type=float, default=0.001,
                        help='KL clip (default: 0.001)')
    parser.add_argument('--skip-layers', nargs='+', type=str, default=[],
                        help='Layer types to ignore registering with KFAC (default: [])')
    parser.add_argument('--coallocate-layer-factors', action='store_true', default=False,
                        help='Compute A and G for a single layer on the same worker. ')

    # Set automatically by torch distributed launch
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


if __name__ == '__main__': 
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()

    if args.cuda:
        torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['LOCAL_RANK'] = str(args.local_rank)

    args.backend = kfac.utils.get_comm_backend()
    args.base_lr = args.base_lr * args.backend.size() * args.batches_per_allreduce
    args.verbose = True if args.backend.rank() == 0 else False
    args.horovod = False

    train_sampler, train_loader, _, val_loader = datasets.get_imagenet(args)
    if args.model.lower() == 'resnet50':
        model = models.resnet50()
    elif args.model.lower() == 'resnet101':
        model = models.resnet101()
    elif args.model.lower() == 'resnet152':
        model = models.resnet152()

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        model.cuda()
    torch.backends.cudnn.benchmark = True

    model = torch.nn.parallel.DistributedDataParallel(model,
            device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True)

    if args.verbose:
        summary(model, (3, 32, 32))

    args.log_dir = os.path.join(args.log_dir, 
             "imagenet_{}_kfac{}_gpu_{}_{}".format(
             args.model, args.kfac_update_freq, args.backend.size(),
             datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    args.checkpoint_format=os.path.join(args.log_dir, args.checkpoint_format)
    os.makedirs(args.log_dir, exist_ok=True)
    args.log_writer = SummaryWriter(args.log_dir) if args.verbose else None

    # If set > 0, will resume training from a given checkpoint.
    args.resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            args.resume_from_epoch = try_epoch
            break

    optimizer, precon, lr_schedules, lrs, = optimizers.get_optimizer(model, args)
    loss_func = LabelSmoothLoss(args.label_smoothing)

    # Restore from a previous checkpoint, if initial_epoch is specified.
    if args.resume_from_epoch > 0:
        filepath = args.checkpoint_format.format(epoch=args.resume_from_epoch)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    start = time.time()

    for epoch in range(args.epochs):
        engine.train(epoch, model, optimizer, precon, lrs, loss_func, 
                     train_sampler, train_loader, args)
        for scheduler in lr_schedules:
            scheduler.step()
        engine.test(epoch, model, loss_func, val_loader, args)
        if args.backend.rank() == 0:
            save_checkpoint(model, optimizer, args.checkpoint_format, epoch)

    if args.verbose:
        print('\nTraining time: {}'.format(datetime.timedelta(seconds=time.time() - start)))


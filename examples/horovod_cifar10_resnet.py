import argparse
import time
import os
import sys
import datetime
import kfac
import torch
import horovod.torch as hvd

import cnn_utils.cifar_resnet as models
import cnn_utils.datasets as datasets
import cnn_utils.engine as engine
import cnn_utils.optimizers as optimizers

from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint


def parse_args():
    # General settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--data-dir', type=str, default='/tmp/cifar10', metavar='D',
                        help='directory to download cifar10 dataset to')
    parser.add_argument('--log-dir', default='./logs/horovod_cifar10',
                        help='TensorBoard/checkpoint log directory')
    parser.add_argument('--checkpoint-format', default='checkpoint_{epoch}.pth.tar',
                        help='checkpoint file format')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    # Training settings
    parser.add_argument('--model', type=str, default='resnet32',
                        help='ResNet model to use [20, 32, 56]')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--val-batch-size', type=int, default=128,
                        help='input batch size for validation (default: 128)')
    parser.add_argument('--batches-per-allreduce', type=int, default=1,
                        help='number of batches processed locally before '
                             'executing allreduce across workers; it multiplies '
                             'total batch size.')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--base-lr', type=float, default=0.1, metavar='LR',
                        help='base learning rate (default: 0.1)')
    parser.add_argument('--lr-decay', nargs='+', type=int, default=[35, 75, 90],
                        help='epoch intervals to decay lr (default: [35, 75, 90])')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='WE',
                        help='number of warmup epochs (default: 5)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                        help='SGD weight decay (default: 5e-4)')
    parser.add_argument('--checkpoint-freq', type=int, default=10,
                        help='epochs between checkpoints')

    # KFAC Parameters
    parser.add_argument('--kfac-update-freq', type=int, default=10,
                        help='iters between kfac inv ops (0 disables kfac) (default: 10)')
    parser.add_argument('--kfac-cov-update-freq', type=int, default=1,
                        help='iters between kfac cov ops (default: 1)')
    parser.add_argument('--kfac-update-freq-alpha', type=float, default=10,
                        help='KFAC update freq multiplier (default: 10)')
    parser.add_argument('--kfac-update-freq-decay', nargs='+', type=int, default=None,
                        help='KFAC update freq decay schedule (default None)')
    parser.add_argument('--use-inv-kfac', action='store_true', default=False,
                        help='Use inverse KFAC update instead of eigen (default False)')
    parser.add_argument('--stat-decay', type=float, default=0.95,
                        help='Alpha value for covariance accumulation (default: 0.95)')
    parser.add_argument('--damping', type=float, default=0.003,
                        help='KFAC damping factor (defaultL 0.003)')
    parser.add_argument('--damping-alpha', type=float, default=0.5,
                        help='KFAC damping decay factor (default: 0.5)')
    parser.add_argument('--damping-decay', nargs='+', type=int, default=None,
                        help='KFAC damping decay schedule (default None)')
    parser.add_argument('--kl-clip', type=float, default=0.001,
                        help='KL clip (default: 0.001)')
    parser.add_argument('--skip-layers', nargs='+', type=str, default=[],
                        help='Layer types to ignore registering with KFAC (default: [])')
    parser.add_argument('--coallocate-layer-factors', action='store_true', default=True,
                        help='Compute A and G for a single layer on the same worker. ')
    parser.add_argument('--kfac-comm-method', type=str, default='comm-opt',
                        help='KFAC communication optimization strategy. One of comm-opt, '
                             'mem-opt, of hybrid-opt. (default: comm-opt)')
    parser.add_argument('--kfac-grad-worker-fraction', type=float, default=0.25,
                        help='Fraction of workers to compute the gradients '
                             'when using HYBRID_OPT (default: 0.25)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

def main():
    args = parse_args()

    os.environ['HOROVOD_FUSION_THRESHOLD'] = "0"
    hvd.init()
    kfac.comm.init_comm_backend()
    args.local_rank = hvd.local_rank()

    if args.cuda:
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True



    print('rank = {}, world_size = {}, device_ids = {}'.format(
            hvd.rank(), hvd.size(), args.local_rank))

    args.backend = kfac.comm.backend
    args.base_lr = args.base_lr * hvd.size() * args.batches_per_allreduce
    args.verbose = True if hvd.rank() == 0 else False
    args.horovod = True

    train_sampler, train_loader, _, val_loader = datasets.get_cifar(args)
    model = models.get_model(args.model)

    device = 'cpu' if not args.cuda else 'cuda'
    model.to(device)
    
    if args.verbose:
        summary(model, (args.batch_size, 3, 32, 32), device=device)

    os.makedirs(args.log_dir, exist_ok=True)
    args.checkpoint_format = os.path.join(args.log_dir, args.checkpoint_format)
    args.log_writer = SummaryWriter(args.log_dir) if args.verbose else None

    args.resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            args.resume_from_epoch = try_epoch
            break

    optimizer, preconditioner, lr_schedules = optimizers.get_optimizer(model, args)
    loss_func = torch.nn.CrossEntropyLoss()

    if args.resume_from_epoch > 0:
        filepath = args.checkpoint_format.format(epoch=args.resume_from_epoch)
        map_location = {'cuda:0': 'cuda:{}'.format(args.local_rank)}
        checkpoint = torch.load(filepath, map_location=map_location)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if isinstance(checkpoint['schedulers'], list):
            for sched, state in zip(lr_schedules, checkpoint['schedulers']):
                sched.load_state_dict(state)
        if (checkpoint['preconditioner'] is not None and
                preconditioner is not None):
            preconditioner.load_state_dict(checkpoint['preconditioner'])

    start = time.time()

    for epoch in range(args.resume_from_epoch + 1, args.epochs + 1):
        engine.train(epoch, model, optimizer, preconditioner, loss_func, 
                     train_sampler, train_loader, args)
        engine.test(epoch, model, loss_func, val_loader, args)
        for scheduler in lr_schedules:
            scheduler.step()
        if (epoch > 0 and epoch % args.checkpoint_freq == 0 and
                hvd.rank() == 0):
            save_checkpoint(model, optimizer, preconditioner, lr_schedules,
                            args.checkpoint_format.format(epoch=epoch))

    if args.verbose:
        print('\nTraining time: {}'.format(datetime.timedelta(seconds=time.time() - start)))

if __name__ == '__main__':
    main()

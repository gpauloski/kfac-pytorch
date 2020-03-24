from __future__ import print_function

import time
from datetime import datetime, timedelta
import argparse
import os
import math
import sys
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.optim.lr_scheduler import LambdaLR
from torchvision import datasets, transforms, models
import horovod.torch as hvd
from tqdm import tqdm
from distutils.version import LooseVersion
from utils import *

sys.path.append("./kfac")
import kfac

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

def initialize():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train-dir', default='/tmp/imagenet/ILSVRC2012_img_train/',
                        help='path to training data')
    parser.add_argument('--val-dir', default='/tmp/imagenet/ILSVRC2012_img_val/',
                        help='path to validation data')
    parser.add_argument('--log-dir', default='./logs',
                        help='tensorboard/checkpoint log directory')
    parser.add_argument('--checkpoint-format', default='checkpoint-{epoch}.pth.tar',
                        help='checkpoint file format')
    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')

    # Default settings from https://arxiv.org/abs/1706.02677.
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=32,
                        help='input batch size for validation')
    parser.add_argument('--epochs', type=int, default=90,
                        help='number of epochs to train')
    parser.add_argument('--base-lr', type=float, default=0.0125,
                        help='learning rate for a single GPU')
    parser.add_argument('--lr-decay', nargs='+', type=int, default=[30, 60, 80],
                        help='epoch intervals to decay lr')
    parser.add_argument('--warmup-epochs', type=float, default=5,
                        help='number of warmup epochs')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--wd', type=float, default=0.00005,
                        help='weight decay')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='label smoothing (default 0.1)')

    # KFAC Parameters
    parser.add_argument('--kfac-update-freq', type=int, default=10,
                        help='iters between kfac inv ops (0 = no kfac) (default: 10)')
    parser.add_argument('--kfac-cov-update-freq', type=int, default=1,
                        help='iters between kfac cov ops (default: 1)')
    parser.add_argument('--stat-decay', type=float, default=0.95,
                        help='Alpha value for covariance accumulation (default: 0.95)')
    parser.add_argument('--damping', type=float, default=0.002,
                        help='KFAC damping factor (default 0.003)')
    parser.add_argument('--kl-clip', type=float, default=0.001,
                        help='KL clip (default: 0.001)')
    parser.add_argument('--inv-block-count', type=int, default=1,
                        help='Number of blocks to approx inv with (default: 1)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--single-threaded', action='store_true', default=False,
                        help='disables multi-threaded dataloading')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    hvd.init()
    torch.manual_seed(args.seed)
    args.verbose = 1 if hvd.rank() == 0 else 0
    if args.verbose:
        print(args)

    if args.cuda:
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    args.log_dir = os.path.join(args.log_dir, 
                                "imagenet_resnet50_kfac{}_gpu_{}_{}".format(
                                args.kfac_update_freq, hvd.size(),
                                datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    args.checkpoint_format=os.path.join(args.log_dir, args.checkpoint_format)
    os.makedirs(args.log_dir, exist_ok=True)

    # If set > 0, will resume training from a given checkpoint.
    args.resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            args.resume_from_epoch = try_epoch
            break

    # Horovod: broadcast resume_from_epoch from rank 0 (which will have
    # checkpoints) to other ranks.
    resume_from_epoch = hvd.broadcast(torch.tensor(args.resume_from_epoch),
                                      root_rank=0,
                                      name='resume_from_epoch').item()

    # Horovod: write TensorBoard logs on first worker.
    try:
        if LooseVersion(torch.__version__) >= LooseVersion('1.2.0'):
            from torch.utils.tensorboard import SummaryWriter
        else:
            from tensorboardX import SummaryWriter
        args.log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None
    except ImportError:
        args.log_writer = None

    return args

def get_datasets(args):
    # Horovod: limit # of CPU threads to be used per worker.
    if args.single_threaded:
        torch.set_num_threads(4)
        kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    else:
        torch.set_num_threads(4)
        kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    train_dataset = datasets.ImageFolder(
            args.train_dir,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]))
    val_dataset = datasets.ImageFolder(
            args.val_dir,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]))

    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=train_sampler, **kwargs)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.val_batch_size,
            sampler=val_sampler, **kwargs)

    return train_sampler, train_loader, val_sampler, val_loader

def get_model(args):
    # Set up standard ResNet-50 model.
    model = models.resnet50()
    if args.cuda:
        model.cuda()

    # Horovod: scale learning rate by the number of GPUs.
    args.base_lr = args.base_lr * hvd.size()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr,
                          momentum=args.momentum, weight_decay=args.wd)

    if args.kfac_update_freq > 0:
        preconditioner = kfac.KFAC(
                model, lr=args.base_lr, stat_decay=args.stat_decay,
                damping=args.damping, kl_clip=args.kl_clip,
                TCov=args.kfac_cov_update_freq,
                TInv=args.kfac_update_freq,
                inv_block_count=args.inv_block_count)
    else:
        # KFAC guarentees grads are equal across ranks before opt.step() is 
        # called so if we do not use kfac we need to wrap the optimizer 
        # with horovod
        compression = hvd.Compression.fp16 if args.fp16_allreduce \
                                           else hvd.Compression.none
        optimizer = hvd.DistributedOptimizer(
                optimizer, named_parameters=model.named_parameters(),
                compression=compression, op=hvd.Average)
        preconditioner = None

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights 
    # to other workers.
    if args.resume_from_epoch > 0 and hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=args.resume_from_epoch)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    lrs = create_lr_schedule(hvd.size(), args.warmup_epochs, args.lr_decay)
    lr_scheduler = [LambdaLR(optimizer, lrs)]
    if preconditioner is not None:
        lr_scheduler.append(LambdaLR(preconditioner, lrs))

    loss_func = LabelSmoothLoss(args.label_smoothing)

    return model, optimizer, preconditioner, lr_scheduler, lrs, loss_func

def train(epoch, model, optimizer, preconditioner, lr_schedules, lrs,
          loss_func, train_sampler, train_loader, args):

    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    for scheduler in lr_schedules:
        scheduler.step()

    with tqdm(total=len(train_loader), 
              desc='Epoch {:3d}/{:3d}'.format(epoch + 1, args.epochs),
              disable=not args.verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)

            loss = loss_func(output, target)
            train_loss.update(loss)
            train_accuracy.update(accuracy(output, target))
            loss.backward()

            if preconditioner is not None:
                preconditioner.step()
            optimizer.step()

            t.set_postfix_str("loss: {:.4f}, acc: {:.2f}%".format(
                    train_loss.avg.item(), 100*train_accuracy.avg.item()))
            t.update(1)

    if args.log_writer is not None:
        args.log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        args.log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)
        args.log_writer.add_scalar('train/lr', args.base_lr * lrs(epoch), epoch)

def validate(epoch, model, loss_func, val_loader, args):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              bar_format='{l_bar}{bar}|{postfix}',
              desc='             '.format(epoch + 1, args.epochs),
              disable=not args.verbose) as t:
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                val_loss.update(loss_func(output, target))
                val_accuracy.update(accuracy(output, target))

                t.update(1)
                if i + 1 == len(val_loader):
                    t.set_postfix_str("\b\b val_loss: {:.4f}, val_acc: {:.2f}%".format(
                            val_loss.avg.item(), 100*val_accuracy.avg.item()),
                            refresh=False)

    if args.log_writer is not None:
        args.log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        args.log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    args = initialize()

    train_sampler, train_loader, _, val_loader = get_datasets(args)
    model, opt, preconditioner, lr_schedules, lrs, loss_func = get_model(args)

    start = time.time()

    for epoch in range(args.resume_from_epoch, args.epochs):
        train(epoch, model, opt, preconditioner, lr_schedules, lrs,
             loss_func, train_sampler, train_loader, args)
        validate(epoch, model, loss_func, val_loader, args)
        save_checkpoint(model, opt, args.checkpoint_format, epoch)

    if args.verbose:
        print("\nTraining time:", str(timedelta(seconds=time.time() - start)))

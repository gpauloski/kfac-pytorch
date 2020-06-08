from __future__ import print_function
import argparse
import time
import os
import sys
import datetime
import math
from distutils.version import LooseVersion

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torchvision import datasets, transforms, models
import torch.utils.data.distributed

from torchsummary import summary
import cifar_resnet as resnet
import horovod.torch as hvd
from tqdm import tqdm
from utils import *

import kfac

STEP_FIRST = LooseVersion(torch.__version__) < LooseVersion('1.1.0')

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
parser.add_argument('--model', type=str, default='resnet32',
                    help='ResNet model to use [20, 32, 56]')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='WE',
                    help='number of warmup epochs (default: 5)')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')

# Optimizer Parameters
parser.add_argument('--base-lr', type=float, default=0.1, metavar='LR',
                    help='base learning rate (default: 0.1)')
parser.add_argument('--lr-decay', nargs='+', type=int, default=[100, 150],
                    help='epoch intervals to decay lr')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                    help='SGD weight decay (default: 5e-4)')

# KFAC Parameters
parser.add_argument('--kfac-update-freq', type=int, default=10,
                    help='iters between kfac inv ops (0 for no kfac updates) (default: 10)')
parser.add_argument('--kfac-cov-update-freq', type=int, default=1,
                    help='iters between kfac cov ops (default: 1)')
parser.add_argument('--kfac-update-freq-alpha', type=float, default=10,
                    help='KFAC update freq multiplier (default: 10)')
parser.add_argument('--kfac-update-freq-schedule', nargs='+', type=int, default=None,
                    help='KFAC update freq schedule (default None)')
parser.add_argument('--stat-decay', type=float, default=0.95,
                    help='Alpha value for covariance accumulation (default: 0.95)')
parser.add_argument('--damping', type=float, default=0.003,
                    help='KFAC damping factor (defaultL 0.003)')
parser.add_argument('--damping-alpha', type=float, default=0.5,
                    help='KFAC damping decay factor (default: 0.5)')
parser.add_argument('--damping-schedule', nargs='+', type=int, default=None,
                    help='KFAC damping decay schedule (default None)')
parser.add_argument('--kl-clip', type=float, default=0.001,
                    help='KL clip (default: 0.001)')
parser.add_argument('--diag-blocks', type=int, default=1,
                    help='Number of blocks to approx layer factor with (default: 1)')
parser.add_argument('--diag-warmup', type=int, default=5,
                    help='Epoch to start diag block approximation at (default: 5)')
parser.add_argument('--distribute-layer-factors', action='store_true', default=False,
                    help='Compute A and G for a single layer on different workers')

# Other Parameters
parser.add_argument('--log-dir', default='./logs',
                    help='TensorBoard log directory')
parser.add_argument('--dir', type=str, default='/tmp/cifar10', metavar='D',
                    help='directory to download cifar10 dataset to')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Horovod: initialize library.
hvd.init()
torch.manual_seed(args.seed)
verbose = True if hvd.rank() == 0 else False

if args.cuda:
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)

torch.backends.cudnn.benchmark = True

args.log_dir = os.path.join(args.log_dir, 
                            "cifar10_{}_kfac{}_gpu_{}_{}".format(
                            args.model, args.kfac_update_freq, hvd.size(),
                            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
os.makedirs(args.log_dir, exist_ok=True)
log_writer = SummaryWriter(args.log_dir) if verbose else None

# Horovod: limit # of CPU threads to be used per worker.
torch.set_num_threads(4)

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

download = True if hvd.local_rank() == 0 else False
if not download: hvd.allreduce(torch.tensor(1), name="barrier")
train_dataset = datasets.CIFAR10(root=args.dir, train=True, 
                                 download=download, transform=transform_train)
test_dataset = datasets.CIFAR10(root=args.dir, train=False,
                                download=download, transform=transform_test)
if download: hvd.allreduce(torch.tensor(1), name="barrier")

# Horovod: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size * args.batches_per_allreduce, 
        sampler=train_sampler, **kwargs)

# Horovod: use DistributedSampler to partition the test data.
test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
test_loader = torch.utils.data.DataLoader(test_dataset, 
        batch_size=args.test_batch_size, sampler=test_sampler, **kwargs)

if args.model.lower() == "resnet20":
    model = resnet.resnet20()
elif args.model.lower() == "resnet32":
    model = resnet.resnet32()
elif args.model.lower() == "resnet44":
    model = resnet.resnet44()
elif args.model.lower() == "resnet56":
    model = resnet.resnet56()
elif args.model.lower() == "resnet110":
    model = resnet.resnet110()

if args.cuda:
    model.cuda()

if verbose:
    summary(model, (3, 32, 32))

criterion = nn.CrossEntropyLoss()
args.base_lr = args.base_lr * hvd.size()
use_kfac = True if args.kfac_update_freq > 0 else False

optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum,
                      weight_decay=args.weight_decay)

if use_kfac:
    preconditioner = kfac.KFAC(model, lr=args.base_lr, factor_decay=args.stat_decay, 
                               damping=args.damping, kl_clip=args.kl_clip, 
                               fac_update_freq=args.kfac_cov_update_freq, 
                               kfac_update_freq=args.kfac_update_freq,
                               diag_blocks=args.diag_blocks,
                               diag_warmup=args.diag_warmup,
                               distribute_layer_factors=args.distribute_layer_factors)
    kfac_param_scheduler = kfac.KFACParamScheduler(preconditioner,
            damping_alpha=args.damping_alpha,
            damping_schedule=args.damping_schedule,
            update_freq_alpha=args.kfac_update_freq_alpha,
            update_freq_schedule=args.kfac_update_freq_schedule)

# KFAC guarentees grads are equal across ranks before opt.step() is called
# so if we do not use kfac we need to wrap the optimizer with horovod
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
optimizer = hvd.DistributedOptimizer(optimizer, 
                                     named_parameters=model.named_parameters(),
                                     compression=compression,
                                     op=hvd.Average,
                                     backward_passes_per_step=args.batches_per_allreduce)

hvd.broadcast_optimizer_state(optimizer, root_rank=0)
hvd.broadcast_parameters(model.state_dict(), root_rank=0)

lrs = create_lr_schedule(hvd.size(), args.warmup_epochs, args.lr_decay)
lr_scheduler = [LambdaLR(optimizer, lrs)]
if use_kfac:
    lr_scheduler.append(LambdaLR(preconditioner, lrs))

def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    
    if STEP_FIRST:
        for scheduler in lr_scheduler:
            scheduler.step()
        if use_kfac:
            kfac_param_scheduler.step(epoch)
    
    with tqdm(total=len(train_loader), 
              desc='Epoch {:3d}/{:3d}'.format(epoch + 1, args.epochs),
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            for i in range(0, len(data), args.batch_size): 
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                output = model(data_batch)

                loss = criterion(output, target_batch)
                with torch.no_grad():
                    train_loss.update(loss)
                    train_accuracy.update(accuracy(output, target_batch))
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()

            optimizer.synchronize()
            if use_kfac:
                preconditioner.step(epoch=epoch)
            with optimizer.skip_synchronize():
                optimizer.step()

            t.set_postfix_str("loss: {:.4f}, acc: {:.2f}%".format(
            train_loss.avg.item(), 100*train_accuracy.avg.item()))
            t.update(1)

    if not STEP_FIRST:
        for scheduler in lr_scheduler:
            scheduler.step()
        if use_kfac:
            kfac_param_scheduler.step(epoch)

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)

def test(epoch):
    model.eval()
    test_loss = Metric('val_loss')
    test_accuracy = Metric('val_accuracy')
    
    with tqdm(total=len(test_loader),
              bar_format='{l_bar}{bar}|{postfix}',
              desc='             '.format(epoch + 1, args.epochs),
              disable=not verbose) as t:
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                test_loss.update(criterion(output, target))
                test_accuracy.update(accuracy(output, target))
                
                t.update(1)
                if i + 1 == len(test_loader):
                    t.set_postfix_str("\b\b test_loss: {:.4f}, test_acc: {:.2f}%".format(
                            test_loss.avg.item(), 100*test_accuracy.avg.item()),
                            refresh=False)

    if log_writer:
        log_writer.add_scalar('test/loss', test_loss.avg, epoch)
        log_writer.add_scalar('test/accuracy', test_accuracy.avg, epoch)


start = time.time()

for epoch in range(args.epochs):
    train(epoch)
    test(epoch)

if verbose:
    print("\nTraining time:", str(datetime.timedelta(seconds=time.time() - start)))


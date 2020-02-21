from __future__ import print_function
import argparse
import time
import os
import sys
from datetime import timedelta

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
import torch.utils.data.distributed

from torchsummary import summary
import cifar_resnet as resnet
import horovod.torch as hvd
from tqdm import tqdm

sys.path.append("./kfac")
import kfac


# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=140, metavar='N',
                    help='number of epochs to train (default: 140)')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='WE',
                    help='number of warmup epochs (default: 5)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                    help='SGD weight decay (default: 5e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--log-dir', default='./logs',
                    help='TensorBoard log directory')
parser.add_argument('--dir', type=str, default='/tmp/cifar10', metavar='D',
                    help='directory to download cifar10 dataset to')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Horovod: initialize library.
hvd.init()
torch.manual_seed(args.seed)
verbose = True if hvd.rank() == 0 else False

os.makedirs(args.log_dir, exist_ok=True)
log_writer = SummaryWriter(args.log_dir) if verbose else None

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)

torch.backends.cudnn.benchmark = True

# Horovod: limit # of CPU threads to be used per worker.
torch.set_num_threads(1)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

train_dataset = datasets.CIFAR10(root=args.dir, train=True, 
                                 download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root=args.dir, train=False,
                                download=True, transform=transform_test)

# Horovod: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, sampler=train_sampler, **kwargs)

# Horovod: use DistributedSampler to partition the test data.
test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
test_loader = torch.utils.data.DataLoader(test_dataset, 
        batch_size=args.test_batch_size, sampler=test_sampler, **kwargs)

model = resnet.resnet56()

if args.cuda:
    # Move model to GPU.
    model.cuda()
    # If using GPU Adasum allreduce, scale learning rate by local_size.
    if args.use_adasum and hvd.nccl_built():
        lr_scaler = hvd.local_size()

if verbose:
    summary(model, (3, 32, 32))

criterion = nn.CrossEntropyLoss()

# Horovod: scale learning rate by lr_scaler.
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Cross compatibility w/ Horovod versions, if hvd.Average exists then it
# is version >=0.19
if hasattr(hvd, "Average"):
    hvd_kwargs = {"op": hvd.Adasum if args.use_adasum else hvd.Average}
else:
    hvd_kwargs = {}
    
# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression,
                                     **hvd_kwargs)

preconditioner = kfac.KFAC(model)

def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    
    with tqdm(total=len(train_loader), 
              desc='Epoch {:3d}/{:3d}'.format(epoch + 1, args.epochs),
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            adjust_learning_rate(epoch, batch_idx)
            
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            preconditioner.step()
            optimizer.step()

            train_accuracy.update(accuracy(output, target))
            train_loss.update(loss)

            t.set_postfix_str("loss: {:.4f}, acc: {:.2f}%".format(
                    train_loss.avg.item(), 100*train_accuracy.avg.item()))
            t.update(1)

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


def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) /
                                    args.warmup_epochs + 1)
    elif epoch < 60:
        lr_adj = 1.
    elif epoch < 90:
        lr_adj = 1e-1
    elif epoch < 110:
        lr_adj = 1e-2
    elif epoch < 130:
        lr_adj = 1e-3
    else:
        lr_adj = 1e-4
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * hvd.size() * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


start = time.time()

for epoch in range(args.epochs):
    train(epoch)
    test(epoch)

if verbose:
    print("\nTraining time:", str(timedelta(seconds=time.time() - start)))


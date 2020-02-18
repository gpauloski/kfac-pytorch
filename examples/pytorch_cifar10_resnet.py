from __future__ import print_function
import argparse
import time
from datetime import timedelta
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import torch.utils.data.distributed
import horovod.torch as hvd
from kfac_optim import KFACOptimizer

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=140, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--num-thread', type=int, default=2,
                    help='Number of data loader threads per worker')
parser.add_argument('--dir', type=str, default='/tmp/cifar10', metavar='D',
                    help='directory to download cifar10 dataset to')
parser.add_argument('--kfac-update', type=int, default=0,
                    help='KFAC update frequency, 0 for no KFAC')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Horovod: initialize library.
hvd.init()
torch.manual_seed(args.seed)

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)

torch.backends.cudnn.benchmark = True

# Horovod: limit # of CPU threads to be used per worker.
torch.set_num_threads(args.num_threads)

kwargs = {'num_workers': args.num_threads, 'pin_memory': True} if args.cuda else {}

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

trainset = datasets.CIFAR10(root=args.dir, train=True, 
                            download=True, transform=transform_train)
testset = datasets.CIFAR10(root=args.dir, train=False,
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

model = models.resnext50_32x4d(pretrained=False)

# By default, Adasum doesn't need scaling up learning rate.
lr_scaler = hvd.size() if not args.use_adasum else 1

if args.cuda:
    # Move model to GPU.
    model.cuda()
    # If using GPU Adasum allreduce, scale learning rate by local_size.
    if args.use_adasum and hvd.nccl_built():
        lr_scaler = hvd.local_size()

# Horovod: scale learning rate by lr_scaler.
optimizer = optim.SGD(model.parameters(), lr=args.lr * lr_scaler,
                      momentum=args.momentum)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    
# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression,
                                     op=hvd.Adasum if args.use_adasum else hvd.Average)

def train(epoch):
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def test():
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_sampler)
    test_accuracy /= len(test_sampler)

    # Horovod: average metric values across workers.
    test_loss = metric_average(test_loss, 'avg_loss')
    test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))


start = time.time()

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()

if hvd.rank() == 0:
    print("Training time:", str(timedelta(seconds=time.time() - start)))

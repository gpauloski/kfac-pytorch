# PyTorch Distributed K-FAC Preconditioner

[![DOI](https://zenodo.org/badge/240976400.svg)](https://zenodo.org/badge/latestdoi/240976400)

PyTorch Distributed K-FAC Preconditioner

The KFAC code was originally forked from Chaoqi Wang's [KFAC-PyTorch](https://github.com/alecwangcq/KFAC-Pytorch).
The ResNet models for Cifar10 are from Yerlan Idelbayev's [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10).
The CIFAR-10 and ImageNet-1k training scripts are modeled afer Horovod's example PyTorch training scripts.

## Install

### Requirements

KFAC supports [Horovod](https://github.com/horovod/horovod) and `torch.distributed` distributed training backends.

This code is validated to run with PyTorch >=1.1.0, Horovod >=0.19.0, and CUDA 10.{0,1,2}.

### Installation

```
$ git clone https://github.com/gpauloski/kfac_pytorch.git
$ cd kfac_pytorch
$ pip install .
```

## Usage

The K-FAC Preconditioner can be easily added to exisiting training scripts that use `horovod.DistributedOptimizer()`.

```Python
from kfac import KFAC
... 
optimizer = optim.SGD(model.parameters(), ...)
optimizer = hvd.DistributedOptimizer(optimizer, ...)
preconditioner = KFAC(model, ...)
... 
for i, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.synchronize()
    preconditioner.step()
    with optimizer.skip_synchronize():
        optimizer.step()
...
```

Note that the K-FAC Preconditioner expects gradients to be averaged across workers before calling `preconditioner.step()` so we call `optimizer.synchronize()` before hand (Normally `optimizer.synchronize()` is not called until `optimizer.step()`). 

For `torch.distributed` or non-distributed scripts, just call `KFAC.step()` before `optimizer.step()`. KFAC will automatically determine if training is being done in a distributed way and what backend is being used.

## Example Scripts

Example scripts for K-FAC + SGD training on CIFAR-10 and ImageNet-1k are provided. For a full list of training parameters, use `--help`, e.g. `python examples/horovod_cifar10_resnet.py --help`.

### Horovod

```
$ mpiexec -hostfile /path/to/hostfile -N $NGPU_PER_NODE python examples/horovod_{cifar10,imagenet}_resnet.py
```

### torch.distributed

#### Single Node, Multi-GPU
```
$ python -m torch.distributed.launch --nproc_per_node=$NGPU_PER_NODE examples/torch_{cifar10,imagenet}_resnet.py
```

#### Multi-Node, Multi-GPU
On each node, run:
```
$ python -m torch.distributed.launch \
          --nproc_per_node=$NGPU_PER_NODE --nnodes=$NODE_COUNT \
          --node_rank=$NODE_RANK --master_addr=$MASTER_HOSTNAME \
      examples/torch_{cifar10,imagenet}_resnet.py
```

## Citation

Coming soon.

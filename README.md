# PyTorch Distributed KFAC Optimizer

Distributed KFAC Optimizer in PyTorch using [Horovod](https://github.com/horovod/horovod) for communication.

The KFAC code was originally forked from Chaoqi Wang's [KFAC-PyTorch](https://github.com/alecwangcq/KFAC-Pytorch).
The ResNet models for Cifar10 are from Yerlan Idelbayev's [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10).
The Cifar10 training scripts are modeled afer Horovod's example PyTorch training scripts.

## Cifar10 Training

KFAC on 4 GPUs with the KFAC update applied every 10 steps
```
$ mpiexec -hostfile /path/to/hostfile -N 4 python examples/pytorch_cifar10_resnet.py --lr 0.1 --epochs 100 --kfac-update-freq 10 --model resnet56 --lr-decay 60 80
```
Note: if `--kfac-update-freq 0`, SGD will be used instead with the standard Horovod Distributed Optimizer wrapper.

# PyTorch Distributed KFac Optimizer

## Cifar10 Training

KFAC on 4 GPUs with the KFAC update applied every 10 steps
```
$ mpiexec -hostfile /path/to/hostfile -N 4 python examples/pytorch_cifar10_resnet.py --lr 0.1 --epochs 100 --kfac-update-freq 10 --model resnet56 --lr-decay 60 80
```
Note: if `--kfac-update-freq 0`, SGD will be used instead with the standard Horovod Distributed Optimizer wrapper.

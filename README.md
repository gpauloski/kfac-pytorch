# PyTorch Distributed K-FAC Preconditioner

[![DOI](https://zenodo.org/badge/240976400.svg)](https://zenodo.org/badge/latestdoi/240976400)

Code for the papers "[Convolutional Neural Network Training with Distributed K-FAC](https://dl.acm.org/doi/10.5555/3433701.3433826)" and "[KAISA: An Adaptive Second-order Optimizer Framework for Deep Neural Networks](https://arxiv.org/abs/2107.01739)."

This K-FAC implementation is designed to be scalable and efficiently distribute K-FAC computations to reduce training times in distributed environments.
The K-FAC distributed preconditioner supports [Horovod](https://github.com/horovod/horovod) and [torch.distributed](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) for data-parallel training.

The K-FAC code was originally forked from Chaoqi Wang's [KFAC-PyTorch](https://github.com/alecwangcq/KFAC-Pytorch).
The ResNet models for Cifar10 are from Yerlan Idelbayev's [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10).
The CIFAR-10 and ImageNet-1k training scripts are modeled after Horovod's example PyTorch training scripts.

## Table of Contents

- [Install](#install)
- [Example Scripts](#example-scripts)
  - [Horovod](#horovod)
  - [Torch Distributed](#torch-distributed)
- [Usage](#usage)
  - [PyTorch](#pytorch-standard-and-distributed)
  - [Horovod](#horovod)
  - [Communication Strategies](#communication-strategies)
  - [Gradient Accumulation](#gradient-accumulation)
  - [Mixed-Precision Training](#mixed-precision-training)
  - [Model Parallel](#model-parallel)
  - [Module Support](#module-support)
  - [Parameter Scheduling](#parameter-scheduling)
  - [State Dict](#state-dict)
- [Related Code](#related-code)
- [Citation](#citation)

## Install

### Requirements

K-FAC only requires PyTorch.
This code is validated to run with PyTorch >=1.1.0, Horovod >=0.19.0, and CUDA >=10.0.
Some PyTorch features may not be available depending on the PyTorch version.
For example, PyTorch 1.6 or newer is needed for mixed precision training.
The requirements for the provided training examples are given [here](examples/README.md).

### Installation

```
$ git clone https://github.com/gpauloski/kfac_pytorch.git
$ cd kfac_pytorch
$ pip install .
```

## Example Scripts

Example scripts for K-FAC + SGD training on CIFAR-10 and ImageNet-1k are provided.
For a full list of training parameters, use `--help`, e.g. `python examples/horovod_cifar10_resnet.py --help`.
Package requirements for the examples are given in [examples/README.md](examples/README.md).
The code used to produce the results from the paper is frozen in the [kfac-lw](https://github.com/gpauloski/kfac_pytorch/tree/kfac-lw) and [kfac-opt](https://github.com/gpauloski/kfac_pytorch/tree/kfac-opt) branches.

### Horovod

```
$ mpiexec -hostfile /path/to/hostfile -N $NGPU_PER_NODE python examples/horovod_{cifar10,imagenet}_resnet.py
```

### Torch Distributed

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

## Usage

K-FAC requires minimial code to add to existing training scripts.
K-FAC will automatically determine if distributed training is being used and what backend to use.
See the [K-FAC docstring](kfac/preconditioner.py) for a detailed list of K-FAC parameters.

### PyTorch Standard and Distributed

```Python
from kfac import KFAC
...
model = torch.nn.parallel.DistributedDataParallel(...)
optimizer = optim.SGD(model.parameters(), ...)
preconditioner = KFAC(model, ...)
... 
for i, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    preconditioner.step()
    optimizer.step()
...
```

### Horovod

```Python
from kfac import KFAC
... 
model = ...
optimizer = optim.SGD(model.parameters(), ...)
optimizer = hvd.DistributedOptimizer(optimizer, ...)
preconditioner = KFAC(model, ...)
... 
for i, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.synchronize()  # sync grads before K-FAC step
    preconditioner.step()
    with optimizer.skip_synchronize():
        optimizer.step()
...
```

### Communication Strategies

K-FAC has support for three communication strategies: `COMM_OPT`, `MEM_OPT`, and `HYBRID_OPT` that can be selected when initializing K-FAC.
E.g. `kfac.KFAC(model, comm_method=kfac.CommMethod.COMM_OPT)`.

`COMM_OPT` is the default communication method and the design introduced in the paper.
`COMM_OPT` is designed to reduce communication frequency in non-K-FAC update steps and increase maximum worker utilization.

`MEM_OPT` is based on the communication strategy of [Osawa et al. (2019)](https://arxiv.org/pdf/1811.12019.pdf) and is designed to reduce memory usage at the cost of increased communication frequency.

`HYBRID_OPT` combines features of `COMM_OPT` and `MEM_OPT` such that some fraction of workers will simultaneously compute the preconditioned gradients for a layer and broadcast the results to a subset of the remaining workers that are not responsible for computing the gradient.
This results in memory usage that is more than `COMM_OPT` but less than `HYBRID_OPT`.
K-FAC will use the parameter `grad_worker_fraction` to determine how many workers should simultaneously compute the preconditioned gradient for a layer.
Note that `COMM_OPT` is just `HYBRID_OPT` with `grad_worker_fraction=1` because all workers compute the preconditioned gradient while `MEM_OPT` is just `HYBRID_OPT` with `grad_worker_fraction=1/world_size` because only one worker compute the preconditioned gradient and broadcasts it to other workers.

### Gradient Accumulation

If using gradient accumulation to simulate larger batch sizes, K-FAC will by default accumulate data from the forward and backward passes until `KFAC.step()` is called.
Thus, the memory usage of K-FAC will linearly increase with the number of forward/backward passes between calls to `KFAC.step()`.
To reduce memory usage, set `compute_factor_in_hook=True`.
This will compute the factors during the forward/backward pass instead of accumulating the data over multiple passes and computing the factors during the K-FAC step.
This will slow down training but it will reduce memory usage.

### Mixed-Precision Training
K-FAC does **not** support NVIDIA AMP because some operations used in K-FAC (`torch.inverse` and `torch.symeig`) do not support half-precision inputs, and NVIDIA AMP does not have functionality for disabling autocast in certain code regions.

K-FAC suports training with `torch.cuda.amp` and `torch.nn.parallel.DistributedDataParallel`
Torch AMP was added in PyTorch 1.6.
Factors for a given layer will be stored in the data type used during the forward/backward pass.
For most PyTorch layers, this will be float16.
However, there are exceptions such as layers that take integer inputs (often embeddings).
Inverses of the factors are stored in float32, but this can be overridden to float16 to save memory at the cost of potential numerical instability.
The `GradScaler` object can be passed to K-FAC such that K-FAC can appropriately unscale the backward pass data.
If the `GradScaler` is provided, G factors will be cast to float32 to prevent underflow when unscaling the gradients.
.
When using `torch.cuda.amp` for mixed precision training, be sure to call `KFAC.step()` outside of an `autocast()` region. E.g.

```Python
from kfac import KFAC
... 
model = ...
optimizer = optim.SGD(model.parameters(), ...)
scaler = GradScaler()
preconditioner = kfac.KFAC(model, grad_scaler=scaler)
...
for i, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # Unscale gradients before KFAC.step()
    preconditioner.step()
    scaler.step(optimizer)
    scaler.update()
...
```
See the [PyTorch mixed-precision docs](https://pytorch.org/docs/stable/notes/amp_examples.html#typical-mixed-precision-training) for more information.

### Model Parallel

K-FAC is designed for data-parallel training but will work with model-parallel training.
However, depending on how the distributed computing environment is set up, it is possible that only one GPU per model copy is used for distributing and computing the factors.

### Module Support

K-FAC supports Linear and Conv2d modules.
K-FAC will iterate over the model and register all supported modules that it finds.
Modules can be excluded from K-FAC by passing a list of module names to K-FAC.
E.g. `kfac.KFAC(module, skip_layers=['Linear', 'ModelBasicBlock']`.
In this example, K-FAC will not register any modules named `Linear` or `ModelBasicBlock`, and this feature works recursively, i.e. K-FAC will not register any submodules of `ModelBasicBlock`.

K-FAC also experimentally supports Embedding and LSTM modules.
Note that for LSTM support, you must use the LSTM modules provided in `kfac.modules`.
The LSTM module provided by PyTorch uses certain CUDA optimizations that make extracting the forward/backward pass data in module hooks impossible.

### Parameter Scheduling

The learning rate in K-FAC can be scheduled using standard PyTorch LR schedulers. E.g. `scheduler = torch.optim.LambdaLR(preconditioner, func)`.
To schedule the K-FAC damping or update freqencies parameters, see the [KFACParamScheduler](kfac/scheduler.py) and example usage [here](examples/cnn_utils/optimizers.py).

### State Dict

The `KFAC` and `KFACParamScheduler` classes support loading/saving using state dicts.
```Python
state = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'preconditioner': preconditioner.state_dict(),
}
torch.save(state, filepath)

checkpoint = torch.load(filepath)
preconditioner.load_state_dict(checkpoint['preconditioner'])
```
Note by default the state dict will contain the running average of the K-FAC factors for all layers and can produce fairly large checkpoints.
The inverse factors are not saved by default and will be recomputed from the saved factors when loading the state dict.
Loading the state dict will fail if the number of registered K-FAC modules changes.
E.g. because the model changed or K-FAC was initialized to skip different modules.

## Related Code

- [BERT Pretraining with K-FAC](https://github.com/gpauloski/BERT-PyTorch)
- [Mask R-CNN Training with K-FAC](https://github.com/gpauloski/DeepLearningExamples/tree/master/PyTorch/Segmentation/MaskRCNN)
- [U-Net with K-FAC Example](https://github.com/HQ01/kfac_pytorch/blob/unet/examples/torch_brain_unet.py)

## Citation

```
@inproceedings{pauloski2020kfac,
    author = {Pauloski, J. Gregory and Zhang, Zhao and Huang, Lei and Xu, Weijia and Foster, Ian T.},
    title = {Convolutional {N}eural {N}etwork {T}raining with {D}istributed {K}-{FAC}},
    year = {2020},
    isbn = {9781728199986},
    publisher = {IEEE Press},
    booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
    articleno = {94},
    numpages = {14},
    location = {Atlanta, Georgia},
    series = {SC '20},
    doi = {10.5555/3433701.3433826}
}

@misc{pauloski2021kaisa,
      title={KAISA: An Adaptive Second-order Optimizer Framework for Deep Neural Networks}, 
      author={J. Gregory Pauloski and Qi Huang and Lei Huang and Shivaram Venkataraman and Kyle Chard and Ian Foster and Zhao Zhang},
      year={2021},
      eprint={2107.01739},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

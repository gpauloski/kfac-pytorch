# KAISA: a K-FAC Preconditioner for Distributed PyTorch

[![DOI](https://zenodo.org/badge/240976400.svg)](https://zenodo.org/badge/latestdoi/240976400)

KAISA is a **K-FAC**-enabled, **A**daptable, **I**mproved, and **S**c**A**lable second-order optimizer framework.
KAISA enables efficient second-order optimization with K-FAC to reduce time-to-convergence in [PyTorch distributed training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).

## Publications

- J. Gregory Pauloski, Qi Huang, Lei Huang, Shivaram Venkataraman, Kyle Chard, Ian Foster, and Zhao Zhang. 2021. [KAISA: An Adaptive Second-order Optimizer Framework for Deep Neural Networks](https://dl.acm.org/doi/10.1145/3458817.3476152). In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '21). Association for Computing Machinery, New York, NY, USA, Article 13, 1–14.
- J. Gregory Pauloski, Zhao Zhang, Lei Huang, Weijia Xu, and Ian T. Foster. 2020. [Convolutional Neural Network Training with Distributed K-FAC](https://dl.acm.org/doi/10.5555/3433701.3433826). In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC ‘20). IEEE Press, Article 94, 1–14.

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [Example Scripts](#example-scripts)
- [Features](#features)
  - [Distribution Strategies](#distribution-strategies)
  - [Gradient Accumulation](#gradient-accumulation)
  - [Mixed-Precision Training](#mixed-precision-training)
  - [Model Parallel](#model-parallel)
  - [Module Support](#module-support)
  - [Parameter Scheduling](#parameter-scheduling)
  - [State Dict](#state-dict)
- [Developing](#developing)
- [Related Code](#related-code)
- [Citations and References](#citations-and-references)

## Install

### Requirements

K-FAC only requires PyTorch 1.8 or later.
The example scripts have additional requirements defined in [examples/README.md](examples/README.md).

### Installation

```
$ git clone https://github.com/gpauloski/kfac_pytorch.git
$ cd kfac_pytorch
$ pip install .  # Use -e to install in development mode
```

## Usage

K-FAC requires minimial code to incorporate with existing training scripts.
See the [K-FAC docstring](kfac/preconditioner.py) for a detailed list of K-FAC parameters.

```Python
from kfac import KFAC
...
model = torch.nn.parallel.DistributedDataParallel(...)
optimizer = optim.SGD(model.parameters(), ...)
preconditioner = KFAC(model, ...)
...
for data, target in train_loader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    preconditioner.step()
    optimizer.step()
...
```

## Example Scripts

Example scripts for K-FAC + SGD training on CIFAR-10 and ImageNet-1k are provided.
For a full list of training parameters, use `--help`, e.g. `python examples/torch_cifar10_resnet.py --help`.
Package requirements for the examples are given in [examples/README.md](examples/README.md).

**Note**: These examples use the `torchrun` launcher which is only avaible in PyTorch 1.10 and later.
Use `python -m torch.distributed.run` in Pytorch 1.9 or `python -m torch.distributed.launch` in Pytorch 1.8 and older.

#### Single Node, Multi-GPU
```
$ torchrun --standalone --nnodes 1 --nproc_per_node=$NGPUS \
      examples/torch_{cifar10,imagenet}_resnet.py $ARGS
```

#### Multi-Node, Multi-GPU
On each node, run:
```
$ torchrun --nnodes=$NNODES --nproc_per_node=$NGPUS --rdzv_backend=c10d --rdzv_endpoint=$HOST_ADDR \
      examples/torch_{cifar10,imagenet}_resnet.py $ARGS
```

## Features

### Distribution Strategies

K-FAC distributes second-order computations based on the `grad_worker_fraction` which specifies the fraction of workers in the world that should be assigned as gradient workers for a layer.
Larger `grad_worker_fraction`s reduce communication in non-KFAC update steps (i.e., steps where second-order information is not recomputed) by caching data on more workers.
Lower `grad_worker_fraction`s reduce memory usage at the cost of more frequent communication.

The package provides common `grad_worker_fraction`s in the form of the `kfac.DistributedStrategy` enum.
| Strategy                         | Gradient Worker Fraction |
| :------------------------------: | :----------------------: |
| `DistributedStrategy.COMM_OPT`   | 1                        |
| `DistributedStrategy.HYBRID_OPT` | 0.5                      |
| `DistributedStrategy.MEM_OPT`    | 1/world_size             |

```Python
import kfac

# Both are equivalent
preconditioner = kfac.KFAC(model, grad_worker_fraction=1)
preconditioner = kfac.KFAC(model, grad_worker_fraction=kfac.DistributedStrategy.COMM_OPT)
```

For more details on distribution strategies see the [KAISA Paper](https://dl.acm.org/doi/10.1145/3458817.3476152).

### Gradient Accumulation

If using gradient accumulation, pass `accumulation_steps=num_accumulation_steps` to `KFAC()`.

### Mixed-Precision Training

K-FAC does **not** support [NVIDIA Apex Amp](https://github.com/NVIDIA/apex) because some operations used in K-FAC (`torch.linalg.inv` and `torch.linalg.eig*`) do not support half-precision inputs, and NVIDIA Apex Amp does not have functionality for disabling autocast in certain code regions.
Additionally, the [native PyTorch AMP is preferred over NVIDIA Apex Amp](https://discuss.pytorch.org/t/torch-cuda-amp-vs-nvidia-apex/74994/3).

K-FAC supports training with `torch.cuda.amp` and `torch.nn.parallel.DistributedDataParallel`
The `GradScaler` object can optionally be passed to K-FAC such that K-FAC can appropriately unscale the backward pass data.
Example:

```Python
from kfac import KFAC
...
model = torch.nn.parallel.DistributedDataParallel(...)
optimizer = optim.SGD(model.parameters(), ...)
scaler = GradScaler()
preconditioner = kfac.KFAC(model, grad_scaler=scaler)
...
for data, target in train_loader:
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


### Parameter Scheduling

Many of the `KFAC()` parameters can take callables that will return the parameter.
E.g., if using a learning rate decay schedule a callable that will return the learning rate at the current step can be passed to `KFAC()` instead of a static float value.
Alternatively, the learning rate in K-FAC can be scheduled using standard PyTorch LR schedulers. E.g. `scheduler = torch.optim.LambdaLR(kfac_preconditioner, func)`.
To schedule the K-FAC damping or update freqencies parameters, see the [KFACParamScheduler](kfac/scheduler.py) and example usage [here](examples/cnn_utils/optimizers.py).

### State Dict

The `KFAC` and `KFACParamScheduler` classes support loading/saving using state dicts.
```Python
state = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'preconditioner': kfac_preconditioner.state_dict(),
}
torch.save(state, filepath)

checkpoint = torch.load(filepath)
preconditioner.load_state_dict(checkpoint['preconditioner'])
```
Note the state dict will contain the running average of the K-FAC factors for all layers and can produce fairly large checkpoints.
The inverse factors are not saved by default and will be recomputed from the saved factors when loading the state dict.
Loading the state dict will fail if the number of registered K-FAC modules changes.
E.g. because the model changed or K-FAC was initialized to skip different modules.

## Developing

Pre-commit is used for development.
The provided pre-commit config files will run a variety of linters and code formatters.

```
$ pip install -r requirements-dev.txt
$ pre-commit install
$ pre-commit run --all-files
```

Unit test, contained in `test/`, can be run with `pytest`.

Additionally,

## Related Code

- [BERT Pretraining with K-FAC](https://github.com/gpauloski/BERT-PyTorch)
- [Mask R-CNN Training with K-FAC](https://github.com/gpauloski/DeepLearningExamples/tree/master/PyTorch/Segmentation/MaskRCNN)
- [U-Net with K-FAC Example](https://github.com/HQ01/kfac_pytorch/blob/unet/examples/torch_brain_unet.py)

## Citations and References

The K-FAC code is based on Chaoqi Wang's [KFAC-PyTorch](https://github.com/alecwangcq/KFAC-Pytorch).
The ResNet models for Cifar10 are from Yerlan Idelbayev's [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10).
The CIFAR-10 and ImageNet-1k training scripts are modeled after Horovod's example PyTorch training scripts.

The code used in "[Convolutional Neural Network Training with Distributed K-FAC](https://dl.acm.org/doi/10.5555/3433701.3433826)"  is frozen in the `kfac-lw` and `kfac-opt` branches.
The code used in "[KAISA: An Adaptive Second-order Optimizer Framework for Deep Neural Networks](https://dl.acm.org/doi/10.1145/3458817.3476152)" is frozen in the `hybrid-opt` branch.

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

@inproceedings{pauloski2021kaisa,
    author = {Pauloski, J. Gregory and Huang, Qi and Huang, Lei and Venkataraman, Shivaram and Chard, Kyle and Foster, Ian and Zhang, Zhao},
    title = {KAISA: {A}n {A}daptive {S}econd-{O}rder {O}ptimizer {F}ramework for {D}eep {N}eural {N}etworks},
    year = {2021},
    isbn = {9781450384421},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3458817.3476152},
    doi = {10.1145/3458817.3476152},
    booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
    articleno = {13},
    numpages = {14},
    location = {St. Louis, Missouri},
    series = {SC '21}
}

```

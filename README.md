# Distributed K-FAC Preconditioner for PyTorch

[![DOI](https://zenodo.org/badge/240976400.svg)](https://zenodo.org/badge/latestdoi/240976400)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/gpauloski/kfac_pytorch/main.svg)](https://results.pre-commit.ci/latest/github/gpauloski/kfac_pytorch/main)
[![Tests](https://github.com/gpauloski/kfac_pytorch/actions/workflows/tests.yml/badge.svg)](https://github.com/gpauloski/kfac_pytorch/actions)
[![Integration](https://github.com/gpauloski/kfac_pytorch/actions/workflows/integration.yml/badge.svg)](https://github.com/gpauloski/kfac_pytorch/actions)

K-FAC, Kronecker-factored Approximate Curvature, is a second-order optimization method based on an efficient approximation of the Fisher information matrix (see the [original paper](https://arxiv.org/abs/1503.05671)).
This repository provides a PyTorch implementation of K-FAC as a preconditioner to standard PyTorch optimizers with support for single-device or distributed training.
The distributed strategy is implemented using KAISA, a **K-FAC**-enabled, **A**daptable, **I**mproved, and **S**c**A**lable second-order optimizer framework, where the placement of the second-order computations and gradient preconditioning is controlled by the *gradient worker fraction* parameter (see the [paper](https://arxiv.org/abs/2107.01739) for more details).
KAISA has been shown to reduce time-to-convergence in [PyTorch distributed training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) applications such as ResNet-50, Mask R-CNN, and BERT.

## Publications

- J. Gregory Pauloski, Lei Huang, Weijia Xu, Kyle Chard, Ian T. Foster, and Zhao Zhang. 2022. Deep Neural Network Training with Distributed K-FAC. To appear in Transactions on Parallel and Distributed Systems (TPDS).
- J. Gregory Pauloski, Qi Huang, Lei Huang, Shivaram Venkataraman, Kyle Chard, Ian Foster, and Zhao Zhang. 2021. [KAISA: An Adaptive Second-order Optimizer Framework for Deep Neural Networks](https://dl.acm.org/doi/10.1145/3458817.3476152). International Conference for High Performance Computing, Networking, Storage and Analysis (SC '21). Association for Computing Machinery, New York, NY, USA, Article 13, 1–14.
- J. Gregory Pauloski, Zhao Zhang, Lei Huang, Weijia Xu, and Ian T. Foster. 2020. [Convolutional Neural Network Training with Distributed K-FAC](https://dl.acm.org/doi/10.5555/3433701.3433826). International Conference for High Performance Computing, Networking, Storage and Analysis (SC ‘20). IEEE Press, Article 94, 1–14.

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [Examples](#examples)
- [Developing](#developing)
- [Citations and References](#citations-and-references)

## Install

### Requirements

K-FAC only requires PyTorch 1.8 or later.
The example scripts have additional requirements defined in [examples/requirements.txt](examples/requirements.txt).

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
from kfac.preconditioner import KFACPreconditioner

...

model = torch.nn.parallel.DistributedDataParallel(...)
optimizer = optim.SGD(model.parameters(), ...)
preconditioner = KFACPreconditioner(model, ...)

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

See the [wiki](https://github.com/gpauloski/kfac_pytorch/wiki) for more details on K-FAC's features.

## Examples

Example scripts for training ResNet models on Cifar10 and ImageNet-1k are provided in [examples/](examples/).

## Developing

[tox](https://tox.wiki/en/latest/index.html) and [pre-commit](https://pre-commit.com) are used for development.
Pre-commit enforces the code formatting, linting, and type-checking in this repository.

To get started with local development:
```
$ tox --devenv venv
$ . venv/bin/activate
$ pre-commit install
```

To verify code passes pre-commit, run:
```
$ pre-commit run --all-files
```

Tox can also be used to run the test suite:
```
$ tox -e py39  # run all tests in Python 3.9
```

## Citations and References

The K-FAC code is based on Chaoqi Wang's [KFAC-PyTorch](https://github.com/alecwangcq/KFAC-Pytorch).
The ResNet models for Cifar10 are from Yerlan Idelbayev's [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10).
The CIFAR-10 and ImageNet-1k training scripts are modeled after Horovod's example PyTorch training scripts.

The code used in "[Convolutional Neural Network Training with Distributed K-FAC](https://dl.acm.org/doi/10.5555/3433701.3433826)"  is frozen in the `kfac-lw` and `kfac-opt` branches.
The code used in "[KAISA: An Adaptive Second-order Optimizer Framework for Deep Neural Networks](https://dl.acm.org/doi/10.1145/3458817.3476152)" is frozen in the `hybrid-opt` branch.

If you use this code in your work, please cite the SC '20 and '21 papers.

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

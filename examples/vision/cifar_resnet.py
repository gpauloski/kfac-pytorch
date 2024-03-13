"""Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparison and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

Author: Yerlan Idelbayev
Source: https://raw.githubusercontent.com/akamaster/pytorch_resnet_cifar10
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import avg_pool2d
from torch.nn.functional import pad
from torch.nn.functional import relu

__all__ = [
    'ResNet',
    'resnet20',
    'resnet32',
    'resnet44',
    'resnet56',
    'resnet110',
    'resnet1202',
    'get_model',
]


def get_model(model: str) -> torch.nn.Module:
    """Get PyTorch model by name."""
    if model.lower() == 'resnet20':
        model_ = resnet20()
    elif model.lower() == 'resnet32':
        model_ = resnet32()
    elif model.lower() == 'resnet44':
        model_ = resnet44()
    elif model.lower() == 'resnet56':
        model_ = resnet56()
    elif model.lower() == 'resnet110':
        model_ = resnet110()
    return model_


def _weights_init(m: torch.nn.Module) -> None:
    """Initialize weights of linear or conv2d module."""
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    """LambdaLayer."""

    def __init__(self, lambd: Callable[[torch.Tensor], torch.Tensor]):
        """Init LambdaLayer."""
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass which applies lambda to x."""
        return self.lambd(x)


class BasicBlock(nn.Module):
    """Basic ResNet block implementation."""

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        option: str = 'A',
    ) -> None:
        """Init BasicBlock."""
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut: torch.nn.Module = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        'constant',
                        0,
                    ),
                )
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module):
    """ResNet model implementation."""

    def __init__(
        self,
        block: type[BasicBlock],
        num_blocks: list[int],
        num_classes: int = 10,
    ) -> None:
        """Init ResNet."""
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(
        self,
        block: type[BasicBlock],
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> torch.nn.Sequential:
        """Make individual layer."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20() -> ResNet:
    """Get ResNet20 model."""
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32() -> ResNet:
    """Get ResNet20 model."""
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44() -> ResNet:
    """Get ResNet20 model."""
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56() -> ResNet:
    """Get ResNet20 model."""
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110() -> ResNet:
    """Get ResNet20 model."""
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202() -> ResNet:
    """Get ResNet20 model."""
    return ResNet(BasicBlock, [200, 200, 200])


def test(net: ResNet) -> None:
    """Print sizes of all ResNet models."""
    import numpy as np

    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print('Total number of params', total_params)
    print(
        'Total layers',
        len(
            list(
                filter(
                    lambda p: p.requires_grad and len(p.data.size()) > 1,
                    net.parameters(),
                ),
            ),
        ),
    )


if __name__ == '__main__':
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()

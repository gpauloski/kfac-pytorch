"""PyTorch Models for Testing.

Examples borrowed from:
https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html
"""

from __future__ import annotations

import torch
from torch.nn import functional


class TinyModel(torch.nn.Module):
    """Tiny model with two linear layers."""

    def __init__(self):
        """Init TinyModel."""
        super().__init__()

        self.linear1 = torch.nn.Linear(10, 20, bias=False)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(20, 10)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        """Forward pass."""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


class LeNet(torch.nn.Module):
    """LeNet implementation."""

    def __init__(self):
        """Init LeNet."""
        super().__init__()
        # 1 input image channel (black & white), 6 output channels,
        # 5x5 square convolution kernel
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        """Forward pass."""
        # Max pooling over a (2, 2) window
        x = functional.max_pool2d(functional.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = functional.max_pool2d(functional.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        """Return number of flat features in x."""
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

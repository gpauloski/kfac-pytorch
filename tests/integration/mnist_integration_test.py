"""MNIST integration test.

Source: https://github.com/pytorch/examples/blob/0cb38ebb1b6e50426464b3485435c0c6affc2b65/mnist/main.py
"""  # noqa: E501

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from kfac.preconditioner import KFACPreconditioner


class FastMNIST(MNIST):
    """Fast MNIST dataset wrapper.

    Source: https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init FastMNIST."""
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data: torch.Tensor = self.data.unsqueeze(1).float().div(255)
        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(0.1307).div_(0.3081)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get training image and target class."""
        return self.data[index], self.targets[index]


class Net(nn.Module):
    """MNIST Classifier Network."""

    def __init__(self) -> None:
        """Init Net."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.conv2 = nn.Conv2d(4, 4, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(576, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)


def train(
    model: torch.nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: optim.Optimizer,
    preconditioner: KFACPreconditioner | None,
) -> None:
    """Train model for one epoch."""
    model.train()
    for data, target in train_loader:
        for param in model.parameters():
            param.grad = None
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        if preconditioner is not None:
            preconditioner.step()
        optimizer.step()


def evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
) -> float:
    """Measure accuracy on test dataset."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    total_samples = len(test_loader.dataset)  # type: ignore
    return 100 * (correct / total_samples)


def train_and_evaluate(precondition: bool, epochs: int) -> float:
    """Train and test."""
    torch.manual_seed(42)

    train_dataset = FastMNIST('/tmp/MNIST-data', train=True, download=True)
    test_dataset = FastMNIST('/tmp/MNIST-data', train=False, download=True)

    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=0,
    )

    model = Net()
    optimizer = optim.Adadelta(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    if precondition:
        preconditioner = KFACPreconditioner(
            model,
            factor_update_steps=10,
            inv_update_steps=100,
            lr=lambda x: optimizer.param_groups[0]['lr'],
            update_factors_in_hook=False,
        )
    else:
        preconditioner = None

    accuracy = 0.0
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train(model, train_loader, optimizer, preconditioner)
        accuracy = evaluate(model, test_loader)
        scheduler.step()
        end = time.perf_counter()
        print(
            f'Epoch {epoch}: accuracy={accuracy:.2f}%, '
            f'time={end - start:.2f} seconds',
        )

    return accuracy


def main() -> bool:
    """MNIST integration test runner.

    Returns:
        True if training with KFAC produces a higher final validation
        accuracy than without, otherwise returns False.
    """
    start = time.perf_counter()
    print('Starting MNIST integration test...')
    print('Training without KFAC:')
    adadelta_acc = train_and_evaluate(False, 5)
    print('Training with KFAC:')
    kfac_acc = train_and_evaluate(True, 5)
    failure = kfac_acc <= adadelta_acc
    runtime = time.perf_counter() - start
    print(f'Integration test runtime: {runtime:.2f} seconds.')
    if failure:
        print(
            'Failure: KFAC accuracy is worse than default. '
            f'KFAC acc. = {kfac_acc} vs. default acc. = {adadelta_acc}.',
        )
    else:
        print('Success.')
    return failure


if __name__ == '__main__':
    raise SystemExit(main())

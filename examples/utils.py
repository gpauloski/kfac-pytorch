"""Training utilities."""

from __future__ import annotations

from typing import Callable

import torch
import torch.distributed as dist
from torch.nn.functional import log_softmax

import kfac


def accuracy(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Get prediction accuracy."""
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).float().mean()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    preconditioner: kfac.preconditioner.KFACPreconditioner | None,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    filepath: str,
) -> None:
    """Save model checkpoint."""
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'preconditioner': preconditioner.state_dict()
        if preconditioner is not None
        else None,
        'lr_scheduler': lr_scheduler.state_dict()
        if lr_scheduler is not None
        else None,
    }
    torch.save(state, filepath)


class LabelSmoothLoss(torch.nn.Module):
    """Label smoothing loss."""

    def __init__(self, smoothing: float = 0.0):
        """Init LabelSmoothLoss."""
        super().__init__()
        self.smoothing = smoothing

    def forward(
        self,
        input_: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        log_prob = log_softmax(input_, dim=-1)
        weight = (
            input_.new_ones(input_.size())
            * self.smoothing
            / (input_.size(-1) - 1.0)
        )
        weight.scatter_(-1, target.unsqueeze(-1), (1.0 - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


class Metric:
    """Metric tracking class."""

    def __init__(self, name: str):
        """Init Metric."""
        self.name = name
        self.total = torch.tensor(0.0)
        self.n = torch.tensor(0.0)

    def update(self, val: torch.Tensor, n: int = 1) -> None:
        """Update metric.

        Args:
            val (float): new value to add.
            n (int): weight of new value.
        """
        dist.all_reduce(val, async_op=False)
        self.total += val.cpu() / dist.get_world_size()
        self.n += n

    @property
    def avg(self) -> torch.Tensor:
        """Get average of metric."""
        return self.total / self.n


def create_lr_schedule(
    workers: int,
    warmup_epochs: int,
    decay_schedule: list[int],
    alpha: float = 0.1,
) -> Callable[[int], float]:
    """Return lr scheduler lambda."""

    def lr_schedule(epoch: int) -> float:
        """Compute lr scale factor."""
        lr_adj = 1.0
        if epoch < warmup_epochs:
            lr_adj = (
                1.0 / workers * (epoch * (workers - 1) / warmup_epochs + 1)
            )
        else:
            decay_schedule.sort(reverse=True)
            for e in decay_schedule:
                if epoch >= e:
                    lr_adj *= alpha
        return lr_adj

    return lr_schedule

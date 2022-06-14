"""Testing utilties for GPT NeoX code."""
from __future__ import annotations

import torch


class ColumnParallelLinear(torch.nn.Linear):
    """Mock ColumnParallelLinear from Megatron."""

    pass


class RowParallelLinear(torch.nn.Linear):
    """Mock RowParallelLinear from Megatron."""

    pass


def sequential_model(layers: int, hidden_dim: int) -> torch.nn.Sequential:
    """Returns simple sequential linear model."""
    if layers <= 0:
        raise ValueError('Num layers must be greater than 0')

    ls: torch.nn.Module = []
    ls.append(ColumnParallelLinear(hidden_dim, hidden_dim))
    layers -= 1
    ls.extend(
        [RowParallelLinear(hidden_dim, hidden_dim) for _ in range(layers)],
    )

    return torch.nn.Sequential(*ls)

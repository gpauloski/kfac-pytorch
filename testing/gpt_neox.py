"""Testing utilities for GPT NeoX code."""

from __future__ import annotations

from typing import Any
from unittest import mock

import torch
import torch.distributed as dist
from deepspeed.pipe import PipelineModule  # type: ignore
from deepspeed.runtime.pipe.topology import (  # type: ignore
    PipeModelDataParallelTopology,  # type: ignore
)


class ColumnParallelLinear(torch.nn.Linear):
    """Mock ColumnParallelLinear from Megatron."""

    pass


class RowParallelLinear(torch.nn.Linear):
    """Mock RowParallelLinear from Megatron."""

    pass


def get_pipeline_module(*args: Any, **kwargs: Any) -> PipelineModule:
    """Create pipeline module with correct topology type."""
    with mock.patch.object(PipelineModule, 'to', mock.MagicMock()):
        m = PipelineModule(*args, **kwargs)
    m._topo = PipeModelDataParallelTopology(
        num_pp=m.num_stages,
        num_dp=dist.get_world_size(m.world_group) // m.num_stages,
        num_mp=1,
    )
    return m


def sequential_model(layers: int, hidden_dim: int) -> torch.nn.Sequential:
    """Returns simple sequential linear model."""
    if layers <= 0:
        raise ValueError('Num layers must be greater than 0')

    ls: list[torch.nn.Module] = []
    ls.append(ColumnParallelLinear(hidden_dim, hidden_dim))
    layers -= 1
    ls.extend(
        [RowParallelLinear(hidden_dim, hidden_dim) for _ in range(layers)],
    )

    return torch.nn.Sequential(*ls)

"""Unit Tests for testing/distributed.py.

Taken from: https://github.com/EleutherAI/DeeperSpeed/blob/eb7f5cff36678625d23db8a8fe78b4a93e5d2c75/tests/unit/test_dist.py
"""  # noqa: E501

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from testing.distributed import distributed_test


@distributed_test(world_size=3)
def test_distributed_test_init() -> None:
    """Test distributed wrapper initialized torch.distributed."""
    assert dist.is_initialized()
    assert dist.get_world_size() == 3
    assert dist.get_rank() < 3


@pytest.mark.parametrize('number,color', [(1138, 'purple')])
def test_dist_args(number: int, color: str) -> None:
    """Outer test function with inputs from pytest.mark.parametrize()."""

    @distributed_test(world_size=2)
    def _test_dist_args_helper(x: int, color: str = 'red') -> None:
        """Test distributed initialized and parameters are correctly passed."""
        assert dist.get_world_size() == 2
        assert x == 1138
        assert color == 'purple'

    # Ensure that we can parse args to distributed_test decorated functions.
    _test_dist_args_helper(number, color=color)


@distributed_test(world_size=[1, 2, 4])
def test_dist_allreduce() -> None:
    """Test collective communication operations work in simulated env."""
    x = torch.ones(1, 3) * (dist.get_rank() + 1)
    sum_of_ranks = (dist.get_world_size() * (dist.get_world_size() + 1)) // 2
    result = torch.ones(1, 3) * sum_of_ranks
    dist.all_reduce(x)
    assert torch.all(x == result)

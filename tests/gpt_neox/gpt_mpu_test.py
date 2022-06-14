"""Test for custom GPT NeoX Module Helpers."""
from __future__ import annotations

import pytest
import torch

from kfac.gpt_neox.mpu import gather_from_model_parallel_region
from testing.distributed import distributed_test


@pytest.mark.parametrize(
    'world_size,shape,dtype,fp32_allreduce',
    (
        (1, (1,), torch.float, False),
        (2, (10, 10), torch.bfloat16, True),
        (4, (4, 4, 4), torch.float, False),
    ),
)
def test_gather_model_parallel(
    world_size: int,
    shape: tuple[int],
    dtype: torch.dtype,
    fp32_allreduce: bool,
) -> None:
    """Test gather_from_model_parallel_region."""

    @distributed_test(world_size)
    def _test() -> None:
        group = torch.distributed.new_group()
        rank = torch.distributed.get_rank(group)
        world_size = torch.distributed.get_world_size(group)
        dst = 0

        partial = torch.ones(shape, dtype=dtype) * rank
        result = gather_from_model_parallel_region(
            partial,
            dst,
            group,
            fp32_allreduce,
        )

        if rank != dst:
            assert result is None
        else:
            expected_size = list(shape)
            expected_size[-1] = expected_size[-1] * world_size

            assert isinstance(result, torch.Tensor)
            assert result.size() == tuple(expected_size)

    _test()

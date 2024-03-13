"""Test for custom GPT NeoX Module Helpers."""

from __future__ import annotations

import pytest
import torch

from kfac.gpt_neox.mpu import gather_from_model_parallel_region
from kfac.gpt_neox.mpu import get_group_with_rank
from kfac.gpt_neox.mpu import split_tensor_along_dim
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


@pytest.mark.parametrize(
    'rank,groups,result,error',
    (
        (0, [[0], [1], [2]], [0], False),
        (-1, [[0], [1], [2]], None, True),
        (0, [[0, 1], [2], [0]], [0, 1], False),
        (4, [[0, 1, 2, 3], [2, 3, 4]], [2, 3, 4], False),
    ),
)
def test_get_group_with_rank(
    rank: int,
    groups: list[list[int]],
    result: list[int] | None,
    error: bool,
) -> None:
    """Test get_group_with_rank."""
    if error:
        with pytest.raises(ValueError):
            get_group_with_rank(rank, groups)
    else:
        assert get_group_with_rank(rank, groups) == result


def test_split_tensor_along_dim() -> None:
    """Test split_tensor_along_dim."""
    x = torch.zeros([1, 4])
    with pytest.raises(ValueError):
        split_tensor_along_dim(x, 2, 0)

    x = torch.zeros([2, 11])
    with pytest.raises(ValueError):
        split_tensor_along_dim(x, 2, -1)

    x = torch.zeros([6, 18])
    xs = split_tensor_along_dim(x, 6, -1)

    # Every split should be same size
    shapes = [t.size() for t in xs]
    assert len(set(shapes)) == 1
    assert shapes[0] == (6, 3)
    for t in xs:
        assert not t.is_contiguous()

    x = torch.zeros([6, 18])
    xs = split_tensor_along_dim(x, 6, -1, True)
    for t in xs:
        assert t.is_contiguous()

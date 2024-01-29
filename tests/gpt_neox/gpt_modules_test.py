"""Test for custom GPT NeoX Module Helpers."""

from __future__ import annotations

import pytest

from kfac.gpt_neox.modules import GPTNeoXLinearModuleHelper
from testing.distributed import distributed_test
from testing.gpt_neox import ColumnParallelLinear
from testing.gpt_neox import RowParallelLinear


@pytest.mark.parametrize('world_size,bias', ((1, True), (2, False), (4, True)))
def test_linear_module(world_size: int, bias: bool) -> None:
    """Test custom module helper for GPT NeoX."""

    @distributed_test(world_size)
    def _test() -> None:
        import torch.distributed as dist

        in_shape = 10
        out_shape = 5

        row_linear = RowParallelLinear(in_shape, out_shape, bias=bias)
        helper = GPTNeoXLinearModuleHelper(
            row_linear,
            dist.new_group(),
            parallelism='input',
        )

        a_dim_size = (in_shape * dist.get_world_size()) + int(bias)
        assert helper.a_factor_shape == (a_dim_size, a_dim_size)
        assert helper.g_factor_shape == (out_shape, out_shape)

        col_linear = ColumnParallelLinear(in_shape, out_shape, bias=bias)
        helper = GPTNeoXLinearModuleHelper(
            col_linear,
            dist.new_group(),
            parallelism='output',
        )

        a_dim_size = in_shape + int(bias)
        g_dim_size = out_shape * dist.get_world_size()
        assert helper.a_factor_shape == (a_dim_size, a_dim_size)
        assert helper.g_factor_shape == (g_dim_size, g_dim_size)

    _test()

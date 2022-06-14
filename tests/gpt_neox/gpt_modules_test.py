"""Test for custom GPT NeoX Module Helpers."""
from __future__ import annotations

import pytest

from kfac.gpt_neox.modules import GPTNeoXLinearModuleHelper
from testing.distributed import distributed_test
from testing.gpt_neox import RowParallelLinear


@pytest.mark.parametrize('world_size,bias', ((1, True), (2, False), (4, True)))
def test_linear_module(world_size: int, bias: bool) -> None:
    """Test custom module helper for GPT NeoX."""

    @distributed_test(world_size)
    def _test() -> None:
        import torch.distributed as dist

        in_shape = 10
        out_shape = 5

        linear = RowParallelLinear(in_shape, out_shape, bias=bias)
        helper = GPTNeoXLinearModuleHelper(linear, dist.new_group())

        a_dim_size = (in_shape * dist.get_world_size()) + int(bias)
        assert helper.a_factor_shape == (a_dim_size, a_dim_size)
        assert helper.g_factor_shape == (out_shape, out_shape)

    _test()

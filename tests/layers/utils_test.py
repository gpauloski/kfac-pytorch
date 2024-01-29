"""Unit tests for kfac/layers/utils.py."""

from __future__ import annotations

import pytest
import torch

from kfac.layers.utils import append_bias_ones
from kfac.layers.utils import get_cov
from kfac.layers.utils import reshape_data


@pytest.mark.parametrize(
    'shape,out_shape',
    [((1,), (2,)), ((4, 6), (4, 7)), ((1, 2, 3), (1, 2, 4))],
)
def test_append_bias_ones(shape: tuple[int], out_shape: tuple[int]) -> None:
    """Test append_bias_ones."""
    x = torch.rand(shape)
    x_out = append_bias_ones(x)
    assert x_out.shape == out_shape
    assert x_out[..., -1].sum() == x_out[..., -1].numel()


@pytest.mark.parametrize(
    'a,b,scale,expected',
    [
        (torch.ones([2, 2]), None, None, torch.ones([2, 2])),
        (torch.ones([2, 2]), None, 4, 0.5 * torch.ones([2, 2])),
        (torch.ones([2, 2]), torch.zeros([2, 2]), None, torch.zeros([2, 2])),
        (
            torch.ones([2, 2]),
            10 * torch.ones([2, 2]),
            5,
            4 * torch.ones([2, 2]),
        ),
        (
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            None,
            None,
            torch.tensor(
                [[22.0, 26.0, 30.0], [26.0, 31.0, 36.0], [30.0, 36.0, 42.0]],
            ),
        ),
        (
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            torch.tensor([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]),
            2,
            torch.tensor(
                [[27.0, 21.0, 15.0], [36.0, 28.5, 21.0], [45.0, 36.0, 27.0]],
            ),
        ),
        # ValueError cases:
        (torch.ones([2]), None, None, torch.ones([2])),
        (torch.ones([2, 2]), torch.ones([2]), None, torch.ones([2])),
    ],
)
def test_get_cov(
    a: torch.Tensor,
    b: torch.Tensor | None,
    scale: float | None,
    expected: torch.Tensor,
) -> None:
    """Test get_cov."""
    if len(a.shape) != 2 or (b is not None and a.shape != b.shape):
        with pytest.raises(ValueError):
            get_cov(a, b, scale)
    else:
        out = get_cov(a, b, scale)
        assert torch.equal(out, expected)
        if b is None:
            assert torch.equal(out, out.t())


@pytest.mark.parametrize(
    'shapes,collapse_dims,expected',
    [
        (((2, 2),), False, (2, 2)),
        (((2, 2, 2),), False, (2, 2, 2)),
        (((2, 2, 2),), True, (4, 2)),
        (((2, 2), (4, 2)), False, (6, 2)),
        (((2, 2, 2), (4, 2, 2)), False, (6, 2, 2)),
        (((2, 2, 2), (4, 2, 2)), True, (12, 2)),
    ],
)
def test_reshape_data(
    shapes: tuple[tuple[int]],
    collapse_dims: bool,
    expected: tuple[int],
) -> None:
    """Test reshape_data."""
    # TODO: this test does not check batch_first = False (which assumes the
    # batch is the second dimension which is a little strange).
    tensors = [torch.ones(shape) for shape in shapes]
    out = reshape_data(tensors, batch_first=True, collapse_dims=collapse_dims)
    assert out.shape == expected

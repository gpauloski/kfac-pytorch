"""Unit tests for kfac/layers/modules.py."""

from __future__ import annotations

import pytest
import torch

from kfac.layers.modules import Conv2dModuleHelper
from kfac.layers.modules import LinearModuleHelper


@pytest.mark.parametrize(
    'in_ch,out_ch,kernel_size,stride,padding,batch_size,hin,win,bias',
    [(3, 5, 2, 1, 0, 4, 10, 10, True), (3, 5, 3, 2, 1, 4, 10, 10, False)],
)
def test_conv2d_module(
    in_ch: int,
    out_ch: int,
    kernel_size: int,
    stride: int,
    padding: int,
    batch_size: int,
    hin: int,
    win: int,
    bias: bool,
) -> None:
    """Test Conv2dModuleHelper."""
    hout = int((hin + 2 * padding - 1 * (kernel_size - 1) - 1) / stride) + 1
    wout = int((win + 2 * padding - 1 * (kernel_size - 1) - 1) / stride) + 1

    conv2d = torch.nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size,
        stride,
        padding,
        bias=bias,
    )
    helper = Conv2dModuleHelper(conv2d)
    assert isinstance(repr(helper), str)
    assert conv2d.weight.device == helper.device

    data = torch.rand((batch_size, in_ch, hin, win))
    target = torch.rand((batch_size, out_ch, hout, wout))
    loss = (conv2d(data) - target).sum()
    loss.backward()

    grad_shape = (out_ch, in_ch * kernel_size * kernel_size + int(bias))
    assert helper.get_grad().shape == grad_shape
    assert helper.has_bias() == bias
    assert helper.has_symmetric_factors()

    old_weight_grad = helper.get_weight_grad()
    if bias:
        old_bias_grad = helper.get_bias_grad()
    merged_grad = helper.get_grad()

    # Test set_grad() sets weight and bias (if exists)
    helper.set_grad(merged_grad)
    assert torch.equal(old_weight_grad, helper.get_weight_grad())
    if bias:
        assert torch.equal(old_bias_grad, helper.get_bias_grad())

    a = helper.get_a_factor(data)
    g = helper.get_g_factor(target)
    assert (
        a.shape
        == helper.a_factor_shape
        == (
            in_ch * kernel_size * kernel_size + int(bias),
            in_ch * kernel_size * kernel_size + int(bias),
        )
    )
    assert g.shape == helper.g_factor_shape == (out_ch, out_ch)


@pytest.mark.parametrize('bias', [True, False])
def test_linear_module(bias: bool) -> None:
    """Test LinearModuleHelper."""
    in_shape = 5
    out_shape = 3
    batch_size = 4

    linear = torch.nn.Linear(in_shape, out_shape, bias=bias)
    helper = LinearModuleHelper(linear)
    assert isinstance(repr(helper), str)
    assert linear.weight.device == helper.device

    data = torch.rand(in_shape)
    target = torch.rand(out_shape)
    loss = (linear(data) - target).sum()
    loss.backward()

    grad_shape = (out_shape, in_shape + int(bias))
    assert helper.get_grad().shape == grad_shape
    assert helper.has_bias() == bias
    assert helper.has_symmetric_factors()

    old_weight_grad = helper.get_weight_grad()
    if bias:
        old_bias_grad = helper.get_bias_grad()
    merged_grad = helper.get_grad()

    # Test set_grad() sets weight and bias (if exists)
    helper.set_grad(merged_grad)
    assert torch.equal(old_weight_grad, helper.get_weight_grad())
    if bias:
        assert torch.equal(old_bias_grad, helper.get_bias_grad())

    a = helper.get_a_factor(torch.rand([batch_size, in_shape]))
    g = helper.get_g_factor(torch.rand([batch_size, out_shape]))
    assert (
        a.shape
        == helper.a_factor_shape
        == (in_shape + int(bias), in_shape + int(bias))
    )
    assert g.shape == helper.g_factor_shape == (out_shape, out_shape)

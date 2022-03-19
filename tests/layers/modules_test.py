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

    data = torch.rand((batch_size, in_ch, hin, win))
    target = torch.rand((batch_size, out_ch, hout, wout))
    loss = (conv2d(data) - target).sum()
    loss.backward()

    if bias:
        grad_shape = (out_ch, in_ch * kernel_size * kernel_size + 1)
    else:
        grad_shape = (out_ch, in_ch * kernel_size * kernel_size)
    assert helper.get_grad().shape == grad_shape
    assert helper.has_bias() == bias
    assert helper.has_symmetric_factors()

    a = helper.get_a_factor(data)
    g = helper.get_g_factor(target)
    assert a.shape == (
        in_ch * kernel_size * kernel_size + int(bias),
        in_ch * kernel_size * kernel_size + int(bias),
    )
    assert g.shape == (out_ch, out_ch)


@pytest.mark.parametrize('bias', [True, False])
def test_linear_module(bias: bool) -> None:
    in_shape = 5
    out_shape = 3
    batch_size = 4

    linear = torch.nn.Linear(in_shape, out_shape, bias=bias)
    helper = LinearModuleHelper(linear)

    data = torch.rand(in_shape)
    target = torch.rand(out_shape)
    loss = (linear(data) - target).sum()
    loss.backward()

    if bias:
        grad_shape = (out_shape, in_shape + 1)
    else:
        grad_shape = (out_shape, in_shape)
    assert helper.get_grad().shape == grad_shape
    assert helper.has_bias() == bias
    assert helper.has_symmetric_factors()

    a = helper.get_a_factor(torch.rand([batch_size, in_shape]))
    g = helper.get_g_factor(torch.rand([batch_size, out_shape]))
    assert a.shape == (in_shape + int(bias), in_shape + int(bias))
    assert g.shape == (out_shape, out_shape)

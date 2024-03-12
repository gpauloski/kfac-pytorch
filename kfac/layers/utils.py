"""Utilities for KFAC computations."""

from __future__ import annotations

import torch


def append_bias_ones(tensor: torch.Tensor) -> torch.Tensor:
    """Appends vector of ones to last dimension of tensor.

    For examples, if the input is of shape [4, 6], then the outputs has shape
    [4, 7] where the slice [:, -1] is a tensor of all ones.
    """
    shape = list(tensor.shape[:-1]) + [1]
    return torch.cat([tensor, tensor.new_ones(shape)], dim=-1)


def get_cov(
    a: torch.Tensor,
    b: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Computes the empirical second moment of a 2D tensor.

    Reference:
      - https://github.com/tensorflow/kfac/blob/master/kfac/python/ops/fisher_factors.py#L220
      - https://arxiv.org/pdf/1602.01407.pdf#subsection.2.2

    Args:
        a (tensor): 2D tensor to compute second moment of using
            cov_a = a^T @ a.
        b (tensor, optional): optional tensor of equal shape to a such that
            cov_a = a^T @ b.
        scale (float, optional): optional tensor to divide cov_a by. Default
            is a.size(0).

    Returns:
        square tensor representing the second moment of a.
    """  # noqa: E501
    if len(a.shape) != 2:
        raise ValueError(
            'Input tensor must have 2 dimensions. Got tensor with shape '
            f'{a.shape}',
        )
    if b is not None and a.shape != b.shape:
        raise ValueError(
            'Input tensors must have same shape. Got tensors of '
            'shape {} and {}.'.format(a.shape, b.shape),
        )

    if scale is None:
        scale = a.size(0)

    if b is None:
        cov_a = a.t() @ (a / scale)
        # TODO(gpauloski): is this redundant?
        return (cov_a + cov_a.t()) / 2.0
    else:
        return a.t() @ (b / scale)


def reshape_data(
    data_list: list[torch.Tensor],
    batch_first: bool = True,
    collapse_dims: bool = False,
) -> torch.Tensor:
    """Concat input/output data and clear buffers.

    Args:
        data_list (list): list of tensors of equal, arbitrary shape where the
            batch_dim is either 0 or 1 depending on self.batch_first.
        batch_first (bool, optional): is batch dim first. (default: True)
        collapse_dims (bool, optional): if True, collapse all but the last dim
            together forming a 2D output tensor.

    Returns:
        single tensor with all tensors from data_list concatenated across
        batch_dim. Guaranteed to be 2D if collapse_dims=True.
    """
    d = torch.cat(data_list, dim=int(not batch_first))
    if collapse_dims and len(d.shape) > 2:
        d = d.view(-1, d.shape[-1])
    return d

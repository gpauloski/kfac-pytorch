from __future__ import annotations

from typing import cast

import torch

from kfac.layer.utils import get_cov
from kfac.layers.utils import append_bias_ones


class ModuleHelper:
    def __init__(self, module: torch.nn.Module):
        self.module = module

    def get_a_factor(self, a: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_g_factor(self, g: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_grad(self) -> torch.Tensor:
        """Get formated gradients (weight and bias) of module

        Returns:
            gradient of shape [ouput_dim, input_dim]. If bias != None,
            concats bias.
        """
        g = cast(torch.Tensor, self.module.weight.grad)
        if self.has_bias:
            g = torch.cat(
                [g, self.module.bias.grad.view(-1, 1)],  # type: ignore
                1,
            )
        return g

    def has_bias(self) -> bool:
        return hasattr(self.module, 'bias') and self.module.bias is not None

    def has_symmetric_factors(self) -> bool:
        return True


class LinearModuleHelper(ModuleHelper):
    def get_a_factor(self, a: torch.Tensor) -> torch.Tensor:
        # a: batch_size * in_dim
        a = a.view(-1, a.shape[-1])
        if self.has_bias():
            a = append_bias_ones(a)
        return get_cov(a)

    def get_g_factor(self, g: torch.Tensor) -> torch.Tensor:
        # g: batch_size * out_dim
        g = g.reshape(-1, g.shape[-1])
        return get_cov(g)


class Conv2dModuleHelper(ModuleHelper):
    def __init__(self, module: torch.nn.Conv2d):
        self.module = module

    def get_a_factor(self, a: torch.Tensor) -> torch.Tensor:
        a = self._extract_patches(a)
        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        if self.has_bias():
            a = append_bias_ones(a)
        a = a / spatial_size
        return get_cov(a)

    def get_g_factor(self, g: torch.Tensor) -> torch.Tensor:
        # g: batch_size * n_filters * out_h * out_w
        # n_filters is actually the output dimension
        # (analogous to Linear layer)
        spatial_size = g.size(2) * g.size(3)
        g = g.transpose(1, 2).transpose(2, 3)
        g = g.reshape(-1, g.size(-1))
        g = g / spatial_size
        return get_cov(g)

    def get_grad(self) -> torch.Tensor:
        grad = cast(
            torch.Tensor,
            self.module.weight.grad.view(
                self.module.weight.grad.size(0),  # type: ignore
                -1,
            ),
        )
        if self.has_bias():
            grad = torch.cat(
                [grad, self.module.bias.grad.view(-1, 1)],  # type: ignore
                1,
            )
        return grad

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patches from convolutional layer

        Args:
            x: The input feature maps.  (batch_size, in_c, h, w)

        Returns:
            Tensor of shape (batch_size, out_h, out_w, in_c*kh*kw)
        """
        padding = cast(list[int], self.module.padding)
        kernel_size = cast(list[int], self.module.kernel_size)
        stride = cast(list[int], self.module.stride)
        if padding[0] + padding[1] > 0:
            x = torch.nn.functional.pad(
                x,
                (padding[1], padding[1], padding[0], padding[0]),
            ).data
        x = x.unfold(2, kernel_size[0], stride[0])
        x = x.unfold(3, kernel_size[1], stride[1])
        x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
        x = x.view(
            x.size(0),
            x.size(1),
            x.size(2),
            x.size(3) * x.size(4) * x.size(5),
        )
        return x

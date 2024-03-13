"""Helper wrappers for supported PyTorch modules."""

from __future__ import annotations

from typing import cast
from typing import List

import torch

from kfac.layers.utils import append_bias_ones
from kfac.layers.utils import get_cov


class ModuleHelper:
    """PyTorch module helper.

    This base class provides the interface which the KFACBaseLayer expects
    as input. Namely, the interface provides methods to compute the factors
    of a module, get the shapes of the factors, and get and set the gradients.
    """

    def __init__(self, module: torch.nn.Module):
        """Init ModuleHelper.

        Args:
            module (torch.nn.Module): module in model to wrap.
        """
        self.module = module

    def __repr__(self) -> str:
        """Representation of the ModuleHelper instance."""
        return f'{self.__class__.__name__}({repr(self.module)})'

    @property
    def a_factor_shape(self) -> tuple[int, int]:
        """Get shape of A factor."""
        raise NotImplementedError

    @property
    def g_factor_shape(self) -> tuple[int, int]:
        """Get shape of G factor."""
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """Get device that the modules parameters are on."""
        return next(self.module.parameters()).device

    def get_a_factor(self, a: torch.Tensor) -> torch.Tensor:
        """Compute A factor with the input from the forward pass."""
        raise NotImplementedError

    def get_g_factor(self, g: torch.Tensor) -> torch.Tensor:
        """Compute G factor with the gradient w.r.t. the output."""
        raise NotImplementedError

    def get_grad(self) -> torch.Tensor:
        """Get formatted gradients (weight and bias) of module.

        Returns:
            gradient of shape If bias != None,
            concats bias.
        """
        g = cast(torch.Tensor, self.module.weight.grad)
        if self.has_bias():
            g = torch.cat(
                [g, self.module.bias.grad.view(-1, 1)],  # type: ignore
                1,
            )
        return g

    def get_bias_grad(self) -> torch.Tensor:
        """Get the gradient of the bias."""
        return cast(torch.Tensor, self.module.bias.grad)

    def get_weight_grad(self) -> torch.Tensor:
        """Get the gradient of the weight."""
        return cast(torch.Tensor, self.module.weight.grad)

    def has_bias(self) -> bool:
        """Check if module has a bias parameter."""
        return hasattr(self.module, 'bias') and self.module.bias is not None

    def has_symmetric_factors(self) -> bool:
        """Check if module has symmetric factors."""
        return True

    def set_grad(self, grad: torch.Tensor) -> None:
        """Update the gradient of the module."""
        if self.has_bias():
            weight_grad = grad[:, :-1].view(self.get_weight_grad().size())
            bias_grad = grad[:, -1:].view(self.get_bias_grad().size())
        else:
            weight_grad = grad.view(self.get_weight_grad().size())

        if self.has_bias():
            self.module.bias.grad = bias_grad.contiguous()
        self.module.weight.grad = weight_grad.contiguous()


class LinearModuleHelper(ModuleHelper):
    """ModuleHelper for torch.nn.Linear modules."""

    @property
    def a_factor_shape(self) -> tuple[int, int]:
        """Get shape of A factor.

        A shape = (in_features + int(has_bias), in_features + int(has_bias))
        """
        x = self.module.weight.size(1) + int(self.has_bias())  # type: ignore
        return (x, x)

    @property
    def g_factor_shape(self) -> tuple[int, int]:
        """Get shape of G factor.

        G shape = (out_features, out_features)
        """
        return (
            self.module.weight.size(0),  # type: ignore
            self.module.weight.size(0),  # type: ignore
        )

    def get_a_factor(self, a: torch.Tensor) -> torch.Tensor:
        """Compute A factor with the input from the forward pass.

        Args:
            a (torch.Tensor): tensor with shape batch_size * in_dim.
        """
        a = a.view(-1, a.size(-1))
        if self.has_bias():
            a = append_bias_ones(a)
        return get_cov(a)

    def get_g_factor(self, g: torch.Tensor) -> torch.Tensor:
        """Compute G factor with the gradient w.r.t. the output.

        Args:
            g (torch.Tensor): tensor with shape batch_size * out_dim.
        """
        g = g.reshape(-1, g.size(-1))
        return get_cov(g)


class Conv2dModuleHelper(ModuleHelper):
    """ModuleHelper for torch.nn.Conv2d layers."""

    def __init__(self, module: torch.nn.Conv2d):
        """Init ModuleHelper.

        Args:
            module (torch.nn.Conv2d): Conv2d module in model to wrap.
        """
        self.module = module

    @property
    def a_factor_shape(self) -> tuple[int, int]:
        """Get shape of A factor."""
        ksize0: int = self.module.kernel_size[0]  # type: ignore
        ksize1: int = self.module.kernel_size[1]  # type: ignore
        in_ch: int = self.module.in_channels  # type: ignore
        x = in_ch * ksize0 * ksize1 + int(self.has_bias())
        return (x, x)

    @property
    def g_factor_shape(self) -> tuple[int, int]:
        """Get shape of G factor."""
        out_ch: int = self.module.out_channels  # type: ignore
        return (out_ch, out_ch)

    def get_a_factor(self, a: torch.Tensor) -> torch.Tensor:
        """Compute A factor with the input from the forward pass."""
        a = self._extract_patches(a)
        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        if self.has_bias():
            a = append_bias_ones(a)
        a = a / spatial_size
        return get_cov(a)

    def get_g_factor(self, g: torch.Tensor) -> torch.Tensor:
        """Compute G factor with the gradient w.r.t. the output.

        Args:
            g (torch.Tensor): tensor with shape batch_size * n_filters *
                out_h * out_w n_filters is actually the output dimension
                (analogous to Linear layer).
        """
        spatial_size = g.size(2) * g.size(3)
        g = g.transpose(1, 2).transpose(2, 3)
        g = g.reshape(-1, g.size(-1))
        g = g / spatial_size
        return get_cov(g)

    def get_grad(self) -> torch.Tensor:
        """Get formmated gradients (weight and bias) of module."""
        grad = cast(
            torch.Tensor,
            self.module.weight.grad.view(  # type: ignore
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
        """Extract patches from convolutional layer.

        Args:
            x (torch.Tensor): input feature maps with shape
                (batch_size, in_c, h, w).

        Returns:
            tensor of shape (batch_size, out_h, out_w, in_c*kh*kw)
        """
        padding = cast(List[int], self.module.padding)
        kernel_size = cast(List[int], self.module.kernel_size)
        stride = cast(List[int], self.module.stride)
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

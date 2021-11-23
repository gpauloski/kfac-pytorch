import torch

from kfac.layers import utils


class ModuleHelper:
    def __init__(self, module):
        self.module = module

    def get_a_factor(self, a):
        raise NotImplementedError

    def get_g_factor(self, g):
        raise NotImplementedError

    def get_grad(self):
        """Get formated gradients (weight and bias) of module

        Returns:
          gradient of shape [ouput_dim, input_dim]. If bias != None,
          concats bias.
        """
        g = self.module.weight.grad
        if self.has_bias:
            g = torch.cat([g, self.module.bias.grad.view(-1, 1)], 1)
        return g

    def has_bias(self):
        return hasattr(self.module, "bias") and self.module.bias is not None

    def has_symmetric_factors(self):
        return True


class LinearModuleHelper(ModuleHelper):
    def get_a_factor(self, a):
        # a: batch_size * in_dim
        if self.has_bias():
            a = utils.append_bias_ones(a)
        return utils.get_cov(a)

    def get_g_factor(self, g):
        # g: batch_size * out_dim
        return utils.get_cov(g)


class Conv2dModuleHelper(ModuleHelper):
    def get_a_factor(self, a):
        a = self._extract_patches(a)
        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        if self.has_bias():
            a = utils.append_bias_ones(a)
        a = a / spatial_size
        return utils.get_cov(a)

    def get_g_factor(self, g):
        # g: batch_size * n_filters * out_h * out_w
        # n_filters is actually the output dimension (analogous to Linear layer)
        spatial_size = g.size(2) * g.size(3)
        g = g.transpose(1, 2).transpose(2, 3)
        g = g.reshape(-1, g.size(-1))
        g = g / spatial_size
        return utils.get_cov(g)

    def get_grad(self):
        grad = self.module.weight.grad.view(self.module.weight.grad.size(0), -1)
        if self.has_bias():
            grad = torch.cat([grad, self.module.bias.grad.view(-1, 1)], 1)
        return grad

    def _extract_patches(self, x):
        """Extract patches from convolutional layer
        Args:
          x: The input feature maps.  (batch_size, in_c, h, w)

        Returns:
          Tensor of shape (batch_size, out_h, out_w, in_c*kh*kw)
        """
        padding = self.module.padding
        kernel_size = self.module.kernel_size
        stride = self.module.stride
        if padding[0] + padding[1] > 0:
            x = torch.nn.functional.pad(
                x, (padding[1], padding[1], padding[0], padding[0])
            ).data
        x = x.unfold(2, kernel_size[0], stride[0])
        x = x.unfold(3, kernel_size[1], stride[1])
        x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
        x = x.view(
            x.size(0), x.size(1), x.size(2), x.size(3) * x.size(4) * x.size(5)
        )
        return x

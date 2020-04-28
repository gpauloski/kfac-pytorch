import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()

    return x

class cycle:
    def __init__(self, iterable):
        """Iterator that produces tuples indefinitely.

        Example:
          iterator = tuple_cycle([1,2,3], 2)
          assert iterator.next(2) == (1, 2)
          assert iterator.next(1) == (3,)
          assert iterator.next(4) == (1, 2, 3, 1)

        Args:
          iterable: Any iterable to iterate over indefinitely
        """
        self.iterable = iterable
        self.reset()

    def reset(self):
        """Reset iterable to start"""
        self.iterator = itertools.cycle(self.iterable)

    def next(self, size):
        """Get next tuple of size in rotation.

        Returns:
          iterator that returns a tuple of size each time next
          is called.
        """
        return tuple([next(self.iterator) for x in range(size)])

def get_block_boundary(index, block_count, shape):
    """Computes start and end indicies when block diagonalizing a matrix"""
    if index >= block_count:
        raise ValueError("Index ({}) greater than number of requested blocks "
                         "({})".format(index, block_count))
    if block_count > min(shape):
        raise ValueError("Requested blocks ({}) greater than minimum possible "
                         "blocks for shape {}".format(block_count, shape))
    block_shape = [x // block_count for x in shape]
    block_start = [x * index for x in block_shape]
    block_end = [x * (index+1) if (index+1) < block_count 
                           else shape[i] 
                 for i, x in enumerate(block_shape)]
    return block_start, block_end

def _extract_patches(x, kernel_size, stride, padding):
    """Extract patches from convolutional layer

    Args:
      x: The input feature maps.  (batch_size, in_c, h, w)
      kernel_size: the kernel size of the conv filter (tuple of two elements)
      stride: the stride of conv operation  (tuple of two elements)
      padding: number of paddings. be a tuple of two elements
    
    Returns:
      Tensor of shape (batch_size, out_h, out_w, in_c*kh*kw)
    """
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))
    return x


def update_running_avg(new, current, alpha):
    """Compute running average of matrix in-place

    current = alpha*new + (1-alpha)*current
    """
    current *= alpha / (1 - alpha)
    current += new
    current *= (1 - alpha)


class ComputeA:

    @classmethod
    def compute_cov_a(cls, a, layer):
        return cls.__call__(a, layer)

    @classmethod
    def __call__(cls, a, layer):
        if isinstance(layer, nn.Linear):
            cov_a = cls.linear(a, layer)
        elif isinstance(layer, nn.Conv2d):
            cov_a = cls.conv2d(a, layer)
        else:
            raise NotImplementedError("KFAC does not support layer: ".format(layer))

        return cov_a

    @staticmethod
    def conv2d(a, layer):
        batch_size = a.size(0)
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        a = a/spatial_size
        # FIXME(CW): do we need to divide the output feature map's size?
        return a.t() @ (a / batch_size)

    @staticmethod
    def linear(a, layer):
        # a: batch_size * in_dim
        batch_size = a.size(0)
        #if len(a.shape) > 2:
        #    a = a.view(-1, a.shape[-1])
        #    a = torch.mean(a, list(range(len(a.shape)))[1:-1])
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        return a.t() @ (a / batch_size)


class ComputeG:

    @classmethod
    def compute_cov_g(cls, g, layer, batch_averaged=False):
        """
        :param g: gradient
        :param layer: the corresponding layer
        :param batch_averaged: if the gradient is already averaged with the batch size?
        :return:
        """
        # batch_size = g.size(0)
        return cls.__call__(g, layer, batch_averaged)

    @classmethod
    def __call__(cls, g, layer, batch_averaged):
        if isinstance(layer, nn.Conv2d):
            cov_g = cls.conv2d(g, layer, batch_averaged)
        elif isinstance(layer, nn.Linear):
            cov_g = cls.linear(g, layer, batch_averaged)
        else:
            raise NotImplementedError("KFAC does not support layer: ".format(layer))

        return cov_g

    @staticmethod
    def conv2d(g, layer, batch_averaged):
        # g: batch_size * n_filters * out_h * out_w
        # n_filters is actually the output dimension (analogous to Linear layer)
        spatial_size = g.size(2) * g.size(3)
        batch_size = g.shape[0]
        g = g.transpose(1, 2).transpose(2, 3)
        g = try_contiguous(g)
        g = g.view(-1, g.size(-1))

        if batch_averaged:
            g = g * batch_size
        g = g * spatial_size
        cov_g = g.t() @ (g / g.size(0))

        return cov_g

    @staticmethod
    def linear(g, layer, batch_averaged):
        # g: batch_size * out_dim
        batch_size = g.size(0)
        #if len(g.shape) > 2:
        #    #g = g.view(-1, g.shape[-1])
        #    g = torch.mean(g, list(range(len(g.shape)))[1:-1])
        if batch_averaged:
            cov_g = g.t() @ (g * batch_size)
        else:
            cov_g = g.t() @ (g / batch_size)
        return cov_g

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


def update_running_avg(new, current, alpha):
    """Compute running average of matrix in-place

    current = alpha*new + (1-alpha)*current
    """
    current *= alpha / (1 - alpha)
    current += new
    current *= (1 - alpha)


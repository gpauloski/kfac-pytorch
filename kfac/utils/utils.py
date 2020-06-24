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


def load_balance(n_workers, work):
    """Assigned work to n_workers to minimize max work assigned to worker.

    Sorts work lengths by decreasing order and assigns each work the the
    worker with the lowest current load.

    Args:
      n_workers (int, required): number of workers
      work (list, required): list where work_i is the approximate time the i_th
          work will take.

    Returns:
      list of len(work) where each value is the index of the worker that work_i
          is assigned to.
    """
    if not n_workers > 0:
        raise ValueError('n_workers must be > 0')
    if len(work) == 0: 
        raise ValueError('work cannot be an empty list')
    load = [0] * n_workers
    assignments = [0] * len(work)
    work = zip(range(len(work)), work)
    work = sorted(work, key=lambda x: x[1], reverse=True)
    for i, w in work:
        min_worker = load.index(min(load))
        load[min_worker] += w
        assignments[i] = min_worker
    return assignments


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


import math
import time
import torch

from kfac import comm


_FUNC_TRACES = {}


def clear_trace():
    _FUNC_TRACES = {}


def get_trace(max_history=None):
    out = {}
    for fname, times in _FUNC_TRACES.items():
        if max_history is not None and len(times) > max_times:
            times = times[-max_times:]
        out[fname] = sum(times) / len(times)
    return out


def print_trace(max_history=None):
    """Print function execution times recorded with @trace

    To trace function execution times, use the @kfac.utils.trace()
    decorator on all functions to be traced. Then to get the average
    execution times, call kfac.utils.print_trace().

    Args:
        max_history (int, optional): most recent `max_histroy` times to use
            for average. If None, all are used.
    """
    if len(_FUNC_TRACES) == 0:
        return
    for fname, time in get_trace(max_history).items():
        print('{}: {}'.format(fname, time))


def trace(sync=False):
    def decorator(func):
        def func_timer(*args, **kwargs):
            if sync: comm.backend.barrier()
            t = time.time()
            out = func(*args, **kwargs)
            if sync: comm.backend.barrier()
            t = time.time() - t
 
            if func.__name__ not in _FUNC_TRACES:
                _FUNC_TRACES[func.__name__] = [t]
            else:
                _FUNC_TRACES[func.__name__].append(t)
            return out
        return func_timer
    return decorator


class WorkerAllocator(object):
    """Worker Allocator Abstraction

    Handles dividing up ranks into equal sized groups for sharing inverses
    and preconditioned gradients using Broadcast Groups. This class is only
    currently used by KFAC._assign_workers() to simplify the code in that
    function.

    Args:
      size (int): world size
      compute_grad_fraction (float): fraction of the workers in the world that
          should be responsible for computing the gradient for a given layer.

    Attributes:
      grad_workers (int): number of workers that will compute the gradient
          for a given layer.
      size (int): world size
      bcast_grad_ranks (list(list(int))): Each sublist represents a unique
          subset of ranks that can be used as a broadcast group for the
          gradients. I.e. in each sublist, one of the ranks will compute
          the gradient and the other ranks will receive the gradient.
      bcst_inv_ranks (list(list(int))): Each sublist represents a unqiue
          group of ranks that will compute the gradient for a given layer.
          All of the ranks computing a gradient for a given layer must have
          the inverses sent to them so each sublist is the list of ranks
          that need the inverse from the worker that will compute the inverse.
      bcast_grad_groups (list(BroadcastGroup)): list of broadcast groups for
          each sublist in bcast_grad_ranks.
      bcast_inv_groups (list(BroadcastGroup)): list of broadcast groups for
          each sublist in bcast_inv_ranks.
      grad_groups (int): number of unique gradient broadcast groups
      inv_groups (int): number of unique inverse broadcast groups
    """
    def __init__(self, size, compute_grad_fraction):
        grad_workers = max(1, round(size * compute_grad_fraction))
        if size % grad_workers != 0:
            raise ValueError('compute_grad_fraction must produce equally '
                             'size groups')
        self.size = size
        self.compute_grad_fraction = compute_grad_fraction
        # fraction is # of workers that will compute the gradient for a layer
        self.bcast_grad_ranks = partition_grad_ranks(size, grad_workers)
        self.bcast_inv_ranks = partition_inv_ranks(size, grad_workers)
        self.bcast_inv_groups = [self.make_group(ranks) 
                for ranks in self.bcast_inv_ranks]
        self.bcast_grad_groups = [self.make_group(ranks)
                for ranks in self.bcast_grad_ranks]

    @property
    def grad_groups(self):
        return len(self.bcast_grad_groups)

    @property
    def inv_groups(self):
        return len(self.bcast_inv_groups)

    def make_group(self, ranks):
        return comm.CommGroup(ranks)

    def get_grad_groups(self, src_ranks):
        pairs = []
        for rank in range(self.size):
            idx = self._get_list_index(rank, self.bcast_grad_ranks)
            grad_ranks = self.bcast_grad_ranks[idx]
            # src is the intersection of the src_ranks (ranks that compute the
            # gradients) and the broadcast group for the current rank
            src = list(set(src_ranks) & set(grad_ranks))[0]
            pairs.append((src, self.get_grad_group(rank)))
        return pairs

    def get_grad_ranks(self, rank):
        idx = self._get_list_index(rank, self.bcast_grad_ranks)
        return self.bcast_grad_ranks[idx]

    def get_inv_ranks(self, rank):
        idx = self._get_list_index(rank, self.bcast_inv_ranks)
        return self.bcast_inv_ranks[idx]

    def get_inv_group(self, rank):
        idx = self._get_list_index(rank, self.bcast_inv_ranks)
        return self.bcast_inv_groups[idx]

    def get_grad_group(self, rank):
        idx = self._get_list_index(rank, self.bcast_grad_ranks)
        return self.bcast_grad_groups[idx]

    def _get_list_index(self, item, nested_list):
        return [i for i, sub_list in enumerate(nested_list)
                if item in sub_list][0]


def partition_grad_ranks(size, grad_workers):
    # see test/worker_allocator.py for examples
    return [[j for j in range(i, size, grad_workers)]
            for i in range(0, grad_workers)]


def partition_inv_ranks(size, grad_workers):
    # see test/worker_allocator.py for examples
    return [list(range(i, min(i+grad_workers, size)))
            for i in range(0, size, grad_workers)] 


def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()

    return x


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


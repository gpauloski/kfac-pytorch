import torch
import math

from kfac import comm


class WorkerAllocator(object):
    def __init__(self, size, compute_grad_fraction):
        self.size = size
        self.compute_grad_fraction = compute_grad_fraction
        # fraction is fraction of ranks in grad group
        self.bcast_grad_ranks = self.partition_grad_ranks(size,
                compute_grad_fraction)
        self.bcast_inv_ranks = self.partition_inv_ranks(size,
                compute_grad_fraction)
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

    # TODO(gpauloski): make not static
    @staticmethod
    def partition_grad_ranks(size, compute_grad_fraction):
        grad_count = max(1, round(size * compute_grad_fraction))
        inv_count = math.ceil(size / grad_count)
        return [[j for j in range(i*grad_count, min((i+1)*grad_count, size))]
                for i in range(0, inv_count)]

    @staticmethod
    def partition_inv_ranks(size, compute_grad_fraction):
        grad_count = max(1, round(size * compute_grad_fraction))
        inv_count = math.ceil(size / grad_count)
        return [[j for j in range(i, size, grad_count)]
                for i in range(0, grad_count)]

    def make_group(self, ranks):
        return comm.BroadcastGroup(ranks)

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


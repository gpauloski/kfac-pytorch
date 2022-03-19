from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Iterable
from typing import TypeVar

Group = TypeVar('Group')


class WorkerAllocator:
    """Worker Allocator Abstraction

    Handles dividing up ranks into equal sized groups for sharing inverses
    and preconditioned gradients using Broadcast Groups. This class is only
    currently used by KFAC._assign_workers() to simplify the code in that
    function.

    Args:
      grad_worker_fraction (float): fraction of the workers in the world that
          should be responsible for computing the gradient for a given layer.
      local_rank (int): local rank of process as assigned by the distributed
          backend.
      world_size (int): number of workers in the environment.
      group_func (callable): optional callable for making communication process
          groups (e.g., torch.distributed.ProcessGroup). The callable
          should take an iterable of ranks in the group.
    """

    def __init__(
        self,
        grad_worker_fraction: float,
        local_rank: int,
        world_size: int,
        group_func: Callable[[list[int]], Group] | None = None,
    ) -> None:
        if 0 > grad_worker_fraction or 1 < grad_worker_fraction:
            raise ValueError(
                'grad_worker_fraction must be in [0, 1]. '
                f'Got {grad_worker_fraction}.',
            )
        if 0 > local_rank:
            raise ValueError('local_rank must be > 0')
        if 0 > world_size:
            raise ValueError('world_size must be > 0')
        grad_workers = world_size * grad_worker_fraction
        if grad_workers != int(grad_workers):
            raise ValueError(
                'world_size*grad_worker_fraction must produce an integer '
                f'value. Found {world_size}*{grad_worker_fraction}'
                f'={grad_workers}.',
            )
        else:
            grad_workers = int(grad_workers)
        if local_rank >= world_size:
            raise ValueError(
                'local_rank={local_rank} larger than world_size={world_size}',
            )
        self.grad_worker_fraction = grad_worker_fraction
        self.grad_workers = grad_workers
        self.local_rank = local_rank
        self.world_size = world_size
        self.group_func = group_func

        self.grad_worker_groups = self.partition_grad_workers(
            self.world_size,
            self.grad_workers,
        )
        self.grad_receiver_groups = self.partition_grad_receivers(
            self.world_size,
            self.grad_workers,
        )
        self._comm_groups: dict[frozenset[int], Group] = {}
        if self.group_func is not None:
            # TODO(gpauloski): skip making communication groups of size 1?
            for group in self.grad_worker_groups | self.grad_receiver_groups:
                assert group not in self._comm_groups
                _group = self.group_func(list(group))
                # TODO(gpauloski): some group configurations resulted in
                # dist.new_group returning the same handle for distinct
                # rank groups
                # assert _group not in self._comm_groups.values()
                self._comm_groups[group] = _group

    @staticmethod
    def partition_grad_workers(
        world_size: int,
        grad_workers: int,
    ) -> set[frozenset[int]]:
        """Returns set of sets of unique gradient workers

        Constructs an m x n grid of the ranks in the world where m=grad_workers
        and and n=world_size/grad_workers with ranks ordered in ascending
        order left-to-right, top-to-bottom. The gradient worker groups are the
        columns of this grid.

        Example:
            input: world_size = 8, grad_workers = 2

            |          grad_worker groups           |
            | group 1 | group 2 | group 3 | group 4 |
            | ------- | ------- | ------- | ------- |
            |    0    |    1    |    2    |    3    | <- grad receiver group 1
            |    4    |    5    |    6    |    7    | <- grad receiver group 2

            output: [[0, 4], [1, 5], [2, 6], [3, 7]]

        Returns:
            Set[Set[int]] where the total number of elements is equal to
            world_size and the size of each subset is equal to grad_workers.
        """
        if not 0 < world_size:
            raise ValueError('world_size must be > 0')
        if world_size % grad_workers != 0:
            raise ValueError(
                'world_size must be an integer multiple of the gradient '
                'worker count',
            )
        partitions = world_size // grad_workers
        return {
            frozenset(range(i, world_size, partitions))
            for i in range(partitions)
        }

    @staticmethod
    def partition_grad_receivers(
        world_size: int,
        grad_workers: int,
    ) -> set[frozenset[int]]:
        """Returns set of sets of unique gradient receiver groups

        Constructs the grid described in `partition_grad_receivers` and returns
        the rows.

        Returns:
            Set[Set[int]] where the total number of elements is equal to
            world_size and the size of each top-level set is equal to
            grad_workers.
        """
        if not 0 < world_size:
            raise ValueError('world_size must be > 0')
        if world_size % grad_workers != 0:
            raise ValueError(
                'world_size must be an integer multiple of the gradient '
                'worker count',
            )
        partitions = world_size // grad_workers
        return {
            frozenset(range(i * partitions, i * partitions + partitions))
            for i in range(grad_workers)
        }

    def comm_group(self, ranks: Iterable[int]) -> Group:
        """Returns corresponding communication group for the set of ranks"""
        if not isinstance(ranks, frozenset):
            ranks = frozenset(ranks)
        if len(self._comm_groups) == 0:
            raise ValueError(
                'Communication groups are not intialized. This is likely '
                'because group_func was not passed to WorkerAllocator.',
            )
        if ranks not in self._comm_groups:
            raise ValueError(f'{ranks} not a known communication group')
        return self._comm_groups[ranks]

    def _unconstrained_assign(
        self,
        work: dict[Any, list[float]],
    ) -> dict[Any, list[int]]:
        """Helper for unconstrained layer work assignments"""
        flat_work = [
            ((group, index), w)
            for group, value in work.items()
            for index, w in enumerate(value)
        ]
        sorted_flat_work = sorted(flat_work, key=lambda x: x[1], reverse=True)
        assignments = {key: [-1] * len(value) for key, value in work.items()}
        for (group, index), w in sorted_flat_work:
            least_loaded = self._worker_load.index(min(self._worker_load))
            self._worker_load[least_loaded] += w
            assignments[group][index] = least_loaded
        return assignments

    def _constrained_assign(
        self,
        work: dict[Any, list[float]],
        worker_groups: list[list[int]],
    ) -> dict[Any, list[int]]:
        """Helper for constrained layer work assignments"""
        summed_work = {k: sum(v) for k, v in work.items()}
        sorted_groups = [
            key
            for key, value in sorted(
                summed_work.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
        assignments = {key: [-1] * len(value) for key, value in work.items()}

        for group_key in sorted_groups:
            # Sum up loads across workers in each worker group
            worker_group_loads = [
                sum(self._worker_load[i] for i in group)
                for group in worker_groups
            ]
            # Get worker group with lowest current load
            worker_group = worker_groups[
                worker_group_loads.index(min(worker_group_loads))
            ]

            # Sort items in the item group by descending cost
            items = list(zip(range(len(work[group_key])), work[group_key]))
            items = sorted(items, key=lambda x: x[1], reverse=True)

            # Perform lowest current load greedy assignment within worker and
            # item group
            for index, w in items:
                worker_group_loads = [
                    self._worker_load[i] for i in worker_group
                ]
                min_worker = worker_group[
                    worker_group_loads.index(min(worker_group_loads))
                ]
                self._worker_load[min_worker] += w
                assignments[group_key][index] = min_worker

        return assignments

    def assign_layer_work(
        self,
        work: dict[Any, list[float]],
        worker_groups: Iterable[Iterable[int]] | None = None,
    ) -> dict[Any, list[int]]:
        """Creates a balanced assignment of workloads over the ranks

        Minimize the makespan of the work that needs to be performed across
        ranks by using a lowest-current load greedy algorithm.

        Args:
            work (Dict[Any, List[float]]): dictionary where values are lists
                of work costs to perform.
            worker_groups (Iterable[Iterable[int]]): optional list of groups of
                worker. If not None, work items within a group in `work`
                will be guarenteed to be performed within the same
                worker_group.

        Returns:
            Dict with same shape as work where values in each list represent
            the worker that was assigned for that item.
        """
        self._worker_load = [0.0] * self.world_size
        if worker_groups is not None:
            _worker_groups = []
            for g in worker_groups:
                if not isinstance(g, list):
                    g = list(g)
                _worker_groups.append(g)
            return self._constrained_assign(work, _worker_groups)
        return self._unconstrained_assign(work)

    def get_grad_src_rank(self, inv_rank: int) -> int:
        """Get rank that computes the gradient for the corresponding inv"""
        grad_worker_ranks = self.get_grad_worker_ranks(inv_rank)
        grad_receiver_ranks = self.get_grad_receiver_ranks()
        src = grad_worker_ranks & grad_receiver_ranks
        # Intersection should always be size 1 because left and right sets
        # represent and row and a column of a table respectively.
        assert len(src) == 1
        return set(src).pop()

    def get_grad_worker_ranks(self, inv_rank: int) -> frozenset[int]:
        """Get ranks that compute the gradient for the corresponding inv"""
        for group in self.grad_worker_groups:
            if inv_rank in group:
                return group
        raise ValueError(f'{inv_rank=} not found in any group.')

    def get_grad_worker_group(self, inv_rank: int) -> Group:
        """Get communication group for gradient workers"""
        return self.comm_group(self.get_grad_worker_ranks(inv_rank))

    def get_grad_receiver_ranks(self) -> frozenset[int]:
        """Get ranks in the grad receiver group for this local rank"""
        for group in self.grad_receiver_groups:
            if self.local_rank in group:
                return group
        raise AssertionError(f'{self.local_rank=} not found in any group.')

    def get_grad_receiver_group(self) -> Group:
        """Get gradient broadcast group for this rank"""
        return self.comm_group(self.get_grad_receiver_ranks())

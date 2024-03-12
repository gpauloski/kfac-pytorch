"""Work assignment interface and implementations."""

from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
from typing import Callable

import torch.distributed as dist


@dataclass(frozen=True)
class _Group:
    """Dataclass for tracking ranks and group handle together."""

    ranks: frozenset[int]
    group: Any


@dataclass(frozen=True)
class _LayerFactors:
    """Dataclass for tracking layer name and factors of the layer together."""

    layer: str
    factors: list[str]


class WorkAssignment(metaclass=ABCMeta):
    """Abstract Interface to a Work Assignment Class."""

    def __repr__(self) -> str:
        """String representation of the work assignment."""
        layer_strs = []
        for layer in self.get_layers():
            factors = self.get_factors(layer)
            invs = {
                factor: self.inv_worker(layer, factor) for factor in factors
            }
            layer_strs.append(
                f'  layer="{layer}": '
                f'is_grad_worker={self.is_grad_worker(layer)}, '
                f'src_grad_worker={self.src_grad_worker(layer)}, '
                f'inv_workers={invs}',
            )
        s = ',\n'.join(layer_strs)
        return f'{self.__class__.__name__}(\n{s}\n)'

    @abstractmethod
    def broadcast_gradients(self) -> bool:
        """Return if gradients need to be broadcast."""
        raise NotImplementedError

    @abstractmethod
    def broadcast_inverses(self) -> bool:
        """Return if inverses need to be broadcast."""
        raise NotImplementedError

    @abstractmethod
    def get_layers(self) -> tuple[str, ...]:
        """Return tuple of layers assigned."""
        raise NotImplementedError

    @abstractmethod
    def get_factors(self, layer: str) -> tuple[str, ...]:
        """Return tuple of factors associated with the layer."""
        raise NotImplementedError

    @abstractmethod
    def inv_worker(self, layer: str, factor: str) -> int:
        """Return rank that computes inverse factor for this layer."""
        raise NotImplementedError

    @abstractmethod
    def is_grad_worker(self, layer: str) -> bool:
        """Return if this rank is a gradient worker for this layer."""
        raise NotImplementedError

    @abstractmethod
    def src_grad_worker(self, layer: str) -> int:
        """Return rank that will share preconditioned gradient.

        If process is a gradient worker, this method should return the
        process rank. Otherwise, if the process is a gradient receiver, this
        method returns the rank that is responsible for sending the
        preconditioned gradient to this process.
        """
        raise NotImplementedError

    @abstractmethod
    def factor_group(
        self,
        layer: str,
        factor: str,
    ) -> dist.ProcessGroup | None:
        """Communication group for allreducing factors."""
        raise NotImplementedError

    @abstractmethod
    def grad_worker_group(self, layer: str) -> dist.ProcessGroup | None:
        """Return communication group for inverse factor broadcast.

        This communication group is used for the broadcasts of the inverses
        from the inverse worker to the remaining gradient workers for the
        layer.
        """
        raise NotImplementedError

    @abstractmethod
    def grad_receiver_group(self, layer: str) -> dist.ProcessGroup | None:
        """Return communication group for preconditioned gradient broadcast.

        This communication group is used for the broadcasts of the gradients
        from the gradient worker to the remaining gradient receivers for the
        layer.
        """
        raise NotImplementedError


class KAISAAssignment(WorkAssignment):
    """Work assignment strategy implementation for KAISA."""

    def __init__(
        self,
        work: dict[str, dict[str, float]],
        *,
        local_rank: int,
        world_size: int,
        grad_worker_fraction: float,
        group_func: Callable[[list[int]], dist.ProcessGroup | None],
        colocate_factors: bool = True,
    ) -> None:
        """Init KAISAAssignment.

        Args:
            work (dict[str, dict[str, int]]): dictionary mapping unique layer
                names to sub-dictionaries where the keys are the str names for
                each factor associated with the layer and the values are the
                cost of each factor computation for load balancing.
            local_rank (int): local rank of process as assigned by the
                distributed backend.
            world_size (int): number of workers in the environment.
            grad_worker_fraction (float): fraction of the workers in the world
                that should be responsible for computing the gradient for a
                given layer. I.e. the gradient worker count is max(1,
                world_size * grad_worker_fraction).
            group_func (callable): callable for making communication process
                groups (e.g., torch.distributed.ProcessGroup). The callable
                should take an iterable of ranks in the group.
            colocate_factors (bool): if True, assign all factors for a layer to
                the same inverse worker. Otherwise, distribute the factors
                across layers in the gradient worker group (default: False).
        """
        if 0 > grad_worker_fraction or 1 < grad_worker_fraction:
            raise ValueError(
                'grad_worker_fraction must be in [0, 1]. '
                f'Got {grad_worker_fraction}.',
            )
        if 0 > local_rank:
            raise ValueError('local_rank must be > 0')
        if 0 > world_size:
            raise ValueError('world_size must be > 0')
        grad_workers = max(1, world_size * grad_worker_fraction)
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
        self.local_rank = local_rank
        self.world_size = world_size
        self.grad_worker_fraction = grad_worker_fraction
        self.grad_workers = grad_workers
        self.group_func = group_func
        self.colocate_factors = colocate_factors

        grad_worker_ranks = self.partition_grad_workers(
            self.world_size,
            self.grad_workers,
        )
        grad_receiver_ranks = self.partition_grad_receivers(
            self.world_size,
            self.grad_workers,
        )

        ranks_to_communication_group: dict[
            frozenset[int],
            dist.ProcessGroup | None,
        ] = {}
        for ranks in grad_worker_ranks | grad_receiver_ranks:
            # TODO(gpauloski): some group configurations resulted in
            #   dist.new_group returning the same handle for distinct
            #   rank groups
            ranks_to_communication_group[ranks] = self.group_func(list(ranks))

        self._inv_assignments = self.greedy_assignment(
            work,
            [list(ranks) for ranks in grad_worker_ranks],
            self.world_size,
            self.colocate_factors,
        )

        self._grad_receiver_groups: dict[str, _Group] = {}
        self._grad_worker_groups: dict[str, _Group] = {}
        for layer in self._inv_assignments:
            inv_worker = list(self._inv_assignments[layer].values()).pop()
            for ranks in grad_worker_ranks:
                if inv_worker in ranks:
                    self._grad_worker_groups[layer] = _Group(
                        ranks=ranks,
                        group=ranks_to_communication_group[ranks],
                    )
            for ranks in grad_receiver_ranks:
                if self.local_rank in ranks:
                    self._grad_receiver_groups[layer] = _Group(
                        ranks=ranks,
                        group=ranks_to_communication_group[ranks],
                    )

    @staticmethod
    def greedy_assignment(
        work: dict[str, dict[str, float]],
        worker_groups: list[list[int]],
        world_size: int,
        colocate_factors: bool,
    ) -> dict[str, dict[str, int]]:
        """Greedy constrained layer work assignments.

        Assigns work units to ranks in a lowest-current load greedy approach.

        Args:
            work: dict mapping layer names to a sub-dict that maps work for
                the layer (e.g., factors) to the approximate cost of that
                work object.
            worker_groups: list of list of ranks where each sub-list of ranks
                represents a worker group. All work (e.g., factor computations)
                for a given layer will be constrained to be workers within
                a worker group. For example, if the worker groups are
                [[0, 1], [2, 3]], there will never be a case where the two
                factors for a given layer are performed on worker in separate
                groups.
            world_size (int): world_size
            colocate_factors (bool): if true, factors for a single layer will
                be assigned to the same worker. Otherwise, factors for a single
                layer can be computed on separate workers given those workers
                are in the same group.

        Returns:
            dict matching the structure of the work inputs except the values
            of the sub-dicts are the worker ranks that the corresponding factor
            should be computed on.
        """
        worker_loads = [0.0] * world_size
        assignments = {
            layer: {factor: -1 for factor in factors}
            for layer, factors in work.items()
        }

        summed_work = {
            layer: sum(factors.values()) for layer, factors in work.items()
        }
        sorted_groups = [
            layer
            for layer, _ in sorted(
                summed_work.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

        for layer in sorted_groups:
            # Sum up loads across workers in each worker group
            worker_group_loads = [
                sum(worker_loads[i] for i in group) for group in worker_groups
            ]
            # Get worker group with lowest current load
            worker_group = worker_groups[
                worker_group_loads.index(min(worker_group_loads))
            ]

            if colocate_factors:
                _worker_group_loads = [worker_loads[i] for i in worker_group]
                min_worker = worker_group[
                    _worker_group_loads.index(min(_worker_group_loads))
                ]
                worker_loads[min_worker] += summed_work[layer]
                for factor in work[layer]:
                    assignments[layer][factor] = min_worker
            else:
                # Sort items in the item group by descending cost
                factors = sorted(
                    work[layer].items(),
                    key=lambda x: (x[1], x[0]),
                    reverse=True,
                )
                # Perform lowest current load greedy assignment within worker
                # and layer factors group
                for factor, cost in factors:
                    _worker_group_loads = [
                        worker_loads[i] for i in worker_group
                    ]
                    min_worker = worker_group[
                        _worker_group_loads.index(min(_worker_group_loads))
                    ]
                    worker_loads[min_worker] += cost
                    assignments[layer][factor] = min_worker

        for layer in assignments:
            for factor in assignments[layer]:
                assert assignments[layer][factor] >= 0

        return assignments

    @staticmethod
    def partition_grad_workers(
        world_size: int,
        grad_workers: int,
    ) -> set[frozenset[int]]:
        """Returns set of sets of unique gradient workers.

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

        Args:
            world_size (int): world size.
            grad_workers (int): number of gradient workers.

        Returns:
            set[set[int]] where the total number of elements is equal to
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
        """Returns set of sets of unique gradient receiver groups.

        Constructs the grid described in `partition_grad_receivers` and returns
        the rows.

        Args:
            world_size (int): world size.
            grad_workers (int): number of gradient workers.

        Returns:
            set[set[int]] where the total number of elements is equal to
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

    def broadcast_gradients(self) -> bool:
        """Return if gradients need to be broadcast.

        In KAISA, this is True when the gradient worker count is less than
        world size (i.e., not the COMM-OPT case).
        """
        return self.grad_workers < self.world_size

    def broadcast_inverses(self) -> bool:
        """Return if inverses need to be broadcast.

        In KAISA, this is True when the gradient worker count is greater than
        1 (i.e., not the MEM-OPT case).
        """
        return self.grad_workers > 1

    def get_layers(self) -> tuple[str, ...]:
        """Return tuple of layers assigned."""
        return tuple(self._inv_assignments.keys())

    def get_factors(self, layer: str) -> tuple[str, ...]:
        """Return tuple of factors associated with the layer."""
        return tuple(self._inv_assignments[layer].keys())

    def inv_worker(self, layer: str, factor: str) -> int:
        """Return rank that computes inverse factor for this layer."""
        return self._inv_assignments[layer][factor]

    def is_grad_worker(self, layer: str) -> bool:
        """Return if this rank is a gradient worker for this layer."""
        return self.local_rank in self._grad_worker_groups[layer].ranks

    def src_grad_worker(self, layer: str) -> int:
        """Return rank that will share preconditioned gradient.

        If process is a gradient worker, this method should return the
        process rank. Otherwise, if the process is a gradient receiver, this
        method returns the rank that is responsible for sending the
        preconditioned gradient to this process.
        """
        return set(
            self._grad_worker_groups[layer].ranks
            & self._grad_receiver_groups[layer].ranks,
        ).pop()

    def factor_group(
        self,
        layer: str,
        factor: str,
    ) -> dist.ProcessGroup | None:
        """Communication group for allreducing factors.

        KAISA assumes strong data-parallel training, i.e., each rank in the
        world will contribute factors computed from its local mini-batch.
        Thus, this function simply returns the global process group.
        """
        return None

    def grad_worker_group(self, layer: str) -> dist.ProcessGroup | None:
        """Return communication group for inverse factor broadcast.

        This communication group is used for the broadcasts of the inverses
        from the inverse worker to the remaining gradient workers for the
        layer.
        """
        return self._grad_worker_groups[layer].group

    def grad_receiver_group(self, layer: str) -> dist.ProcessGroup | None:
        """Return communication group for preconditioned gradient broadcast.

        This communication group is used for the broadcasts of the gradients
        from the gradient worker to the remaining gradient receivers for the
        layer.
        """
        return self._grad_receiver_groups[layer].group

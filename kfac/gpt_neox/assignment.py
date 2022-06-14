"""Custom Assignment for GPT-NeoX."""
from __future__ import annotations

import torch.distributed as dist

from kfac.assignment import WorkAssignment


class GPTNeoXAssignment(WorkAssignment):
    """Pipeline parallel aware work assignment for GPT-NeoX."""

    def __init__(
        self,
        work: dict[str, dict[str, float]],
        *,
        local_rank: int,
        data_parallel_ranks: list[int],
        data_parallel_group: dist.ProcessGroup | None,
    ) -> None:
        """Init GPTNeoxAssignment.

        Args:
            work (dict[str, dict[str, int]]): dictionary mapping unique layer
                names to sub-dictionaries where the keys are the str names for
                each factor associated with the layer and the values are the
                cost of each factor computation for load balancing. Note: that
                this should only be the work performed by the data parallel
                group.
            local_rank (int): local rank of this process.
            data_parallel_ranks (list[int]): list of global ranks within the
                same stage of all pipelines.
            data_parallel_group (ProcessGroup): process group of all ranks
                within the same stage of all pipelines.
        """
        if local_rank not in data_parallel_ranks:
            raise ValueError(
                'The local rank ({local_rank}) must be a member of the data '
                'parallel ranks ({data_parallel_ranks}).',
            )
        self.local_rank = local_rank
        self.data_parallel_ranks = data_parallel_ranks
        self.data_parallel_group = data_parallel_group

        worker_loads = [0.0 for _ in self.data_parallel_ranks]
        self._inv_assignments = {
            layer: {factor: -1 for factor in factors}
            for layer, factors in work.items()
        }
        summed_work = [
            (layer, sum(factors.values())) for layer, factors in work.items()
        ]
        sorted_work = sorted(
            summed_work,
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )

        for layer, cost in sorted_work:
            min_worker_index = worker_loads.index(min(worker_loads))
            min_worker = self.data_parallel_ranks[min_worker_index]
            for factor in self._inv_assignments[layer]:
                self._inv_assignments[layer][factor] = min_worker
            worker_loads[min_worker_index] += cost

    def broadcast_gradients(self) -> bool:
        """Return if gradients need to be broadcast.

        GPT-NeoX uses MEM-OPT training (grad worker fraction = 1/world_size)
        so gradient broadcast is necessary.
        """
        return True

    def broadcast_inverses(self) -> bool:
        """Return if inverses need to be broadcast.

        GPT-NeoX uses MEM-OPT training (grad worker fraction = 1/world_size)
        so inverse broadcast is not necessary.
        """
        return False

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
        return self.local_rank in self._inv_assignments[layer].values()

    def src_grad_worker(self, layer: str) -> int:
        """Return rank that will share preconditioned gradient.

        If process is a gradient worker, this method should return the
        process rank. Otherwise, if the process is a gradient receiver, this
        method returns the rank that is responsible for sending the
        preconditioned gradient to this process.
        """
        ranks = list(self._inv_assignments[layer].values())
        assert ranks.count(ranks[0]) == len(ranks)
        return ranks[0]

    def factor_group(self, layer: str) -> dist.ProcessGroup | None:
        """Communication group for allreducing factors."""
        return self.data_parallel_group

    def grad_worker_group(self, layer: str) -> dist.ProcessGroup | None:
        """Return communication group for inverse factor broadcast.

        This communication group is used for the broadcasts of the inverses
        from the inverse worker to the remaining gradient workers for the
        layer.
        """
        raise NotImplementedError(
            'The GPT-NeoX assignment strategy only supports MEM-OPT '
            'and therefore should not be performing inverse factor '
            'communication.',
        )

    def grad_receiver_group(self, layer: str) -> dist.ProcessGroup | None:
        """Return communication group for preconditioned gradient broadcast.

        This communication group is used for the broadcasts of the gradients
        from the gradient worker to the remaining gradient receivers for the
        layer.
        """
        return self.data_parallel_group

"""Custom Assignment for GPT-NeoX."""

from __future__ import annotations

import torch.distributed as dist

from kfac.assignment import WorkAssignment
from kfac.gpt_neox.mpu import get_group_with_rank

try:
    from deepspeed.runtime.pipe.topology import (  # type: ignore
        PipeModelDataParallelTopology,  # type: ignore
    )

    deepspeed_import_error = None
except ImportError as e:  # pragma: no cover
    deepspeed_import_error = e


class GPTNeoXAssignment(WorkAssignment):
    """Pipeline parallel aware work assignment for GPT-NeoX."""

    def __init__(
        self,
        work: dict[str, dict[str, float]],
        *,
        local_rank: int,
        topology: PipeModelDataParallelTopology,
        data_parallel_group: dist.ProcessGroup | None,
        model_parallel_group: dist.ProcessGroup | None,
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
            topology (PipeModelDataParallelTopology): topology created
                by DeepSpeed.
            data_parallel_group (ProcessGroup): DeepSpeed data parallel
                process group.
            model_parallel_group (ProcessGroup): DeepSpeed model parallel
                process group.
        """
        if deepspeed_import_error is not None:  # pragma: no cover
            raise deepspeed_import_error
        if not isinstance(topology, PipeModelDataParallelTopology):
            raise TypeError(
                'Expected topology to be of type '
                f'{PipeModelDataParallelTopology.__name__} but got '
                f'{type(topology)} instead.',
            )

        self.local_rank = local_rank
        self.data_parallel_group = data_parallel_group
        self.model_parallel_group = model_parallel_group

        # global information
        self.data_parallel_groups = topology.get_axis_comm_lists('data')
        self.model_parallel_groups = topology.get_axis_comm_lists('model')
        self.pipe_parallel_groups = topology.get_axis_comm_lists('pipe')

        self.data_parallel_peers = get_group_with_rank(
            self.local_rank,
            self.data_parallel_groups,
        )
        self.model_parallel_peers = get_group_with_rank(
            self.local_rank,
            self.model_parallel_groups,
        )
        self.pipe_parallel_rank = topology.get_coord(self.local_rank).pipe
        # List of ranks with same pipe rank as us. These are the ranks that
        # have the same layers as us so they are all we care about for the
        # purpose of assigning work
        self.pipe_parallel_peers = [
            r
            for r in range(topology.world_size())
            if topology.get_coord(r).pipe == self.pipe_parallel_rank
        ]

        # Reuse existing groups if possible
        if set(self.pipe_parallel_peers) == set(self.model_parallel_peers):
            self.pipe_parallel_peer_group = self.model_parallel_group
        elif set(self.pipe_parallel_peers) == set(self.data_parallel_peers):
            self.pipe_parallel_peer_group = self.data_parallel_group
        else:
            self.pipe_parallel_peer_group = dist.new_group(
                self.pipe_parallel_peers,
            )

        worker_loads = [0.0 for _ in self.pipe_parallel_peers]
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
            min_worker = self.pipe_parallel_peers[min_worker_index]
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

    def factor_worker(self, layer: str, factor: str) -> int:
        """Worker that gathers the factor from model parallel group peers.

        Also referred to as the primary worker in the layer code.
        """
        inv_ranks = set(self._inv_assignments[layer].values())
        assert len(inv_ranks) == 1
        inv_rank = inv_ranks.pop()

        data_parallel_ranks = get_group_with_rank(
            inv_rank,
            self.data_parallel_groups,
        )
        factor_workers = set(data_parallel_ranks) & set(
            self.model_parallel_peers,
        )
        assert len(factor_workers) == 1
        return factor_workers.pop()

    def is_grad_worker(self, layer: str) -> bool:
        """Return if this rank is a gradient worker for this layer.

        GPTNeoXKFACEigen.precondition_grad() requires every worker in the
        model parallelism group of the inv_worker to enter
        to decide if the grad needs to be gathered to and scatter from the
        true grad worker within the model parallel group, so we just return
        True here and let that method handle which ranks actually do work.
        """
        return (
            len(
                set(self._inv_assignments[layer].values())
                & set(self.model_parallel_peers),
            )
            == 1
        )

    def src_grad_worker(self, layer: str) -> int:
        """Return rank that will share preconditioned gradient.

        If process is a gradient worker, this method should return the
        process rank. Otherwise, if the process is a gradient receiver, this
        method returns the rank that is responsible for sending the
        preconditioned gradient to this process.

        With model parallelism, the src rank is the rank that received the
        partial preconditioned gradient from the inv_worker.
        """
        ranks = list(self._inv_assignments[layer].values())
        assert ranks.count(ranks[0]) == len(ranks)
        # This is just the src rank that computes the preconditioned gradient
        # and then scatters it to the other ranks in its model parallel group
        src_rank = ranks[0]

        model_parallel_ranks = get_group_with_rank(
            src_rank,
            self.model_parallel_groups,
        )
        src = set(self.data_parallel_peers) & set(model_parallel_ranks)
        assert len(src) == 1
        return src.pop()

    def factor_group(
        self,
        layer: str,
        factor: str,
    ) -> dist.ProcessGroup | None:
        """Communication group for allreducing factors.

        The GPTNeoXKFACEigenLayer will ignore this.
        """
        return None

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

from __future__ import annotations

import torch.distributed as dist

from kfac.assignment import WorkAssignment


class LazyAssignment(WorkAssignment):
    def __init__(self, rank: int = 0, broadcast: bool = False):
        self.rank = rank
        self.broadcast = broadcast

    def broadcast_gradients(self) -> bool:
        """Return if gradients need to be broadcast."""
        return self.broadcast

    def broadcast_inverses(self) -> bool:
        """Return if inverses need to be broadcast."""
        return self.broadcast

    def get_layers(self) -> tuple[str, ...]:
        """Return tuple of layers assigned."""
        return tuple()

    def get_factors(self, layer: str) -> tuple[str, ...]:
        """Return tuple of factors associated with the layer."""
        return tuple()

    def inv_worker(self, layer: str, factor: str) -> int:
        """Return rank that computes inverse factor for this layer."""
        return self.rank

    def is_grad_worker(self, layer: str) -> bool:
        """Return if this rank is a gradient worker for this layer."""
        return True

    def src_grad_worker(self, layer: str) -> int:
        """Return rank that will share preconditioned gradient.

        If process is a gradient worker, this method should return the
        process rank. Otherwise, if the process is a gradient receiver, this
        method returns the rank that is responsible for sending the
        preconditioned gradient to this process.
        """
        return self.rank

    def grad_worker_group(self, layer: str) -> dist.ProcessGroup | None:
        """Return communication group for inverse factor broadcast.

        This communication group is used for the broadcasts of the inverses
        from the inverse worker to the remaining gradient workers for the
        layer.
        """
        return None

    def grad_receiver_group(self, layer: str) -> dist.ProcessGroup | None:
        """Return communication group for preconditioned gradient broadcast.
        This communication group is used for the broadcasts of the inverses
        from the inverse worker to the remaining gradient workers for the
        layer.
        """
        return None

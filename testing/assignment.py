"""Lazy WorkAssignment implementation for testing."""

from __future__ import annotations

import torch.distributed as dist

from kfac.assignment import WorkAssignment


class LazyAssignment(WorkAssignment):
    """Lazy assignment where every worker is an inverse worker.

    Used in unit tests force a KFACPreconditioner to execute all options
    in the distributed control flow.
    """

    def __init__(self, rank: int = 0, broadcast: bool = False) -> None:
        """Init LazyAssignment.

        Args:
            rank (int): process rank to simulate (default: 0).
            broadcast (bool): value to return by broadcast_gradients() and
                broadcast_inverses() (default: False).
        """
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

    def factor_group(
        self,
        layer: str,
        factor: str,
    ) -> dist.ProcessGroup | None:
        """Communication group for allreducing factors."""
        return None

    def grad_worker_group(self, layer: str) -> dist.ProcessGroup | None:
        """Return communication group for inverse factor broadcast.

        This communication group is used for the broadcasts of the inverses
        from the inverse worker to the remaining gradient workers for the
        layer.
        """
        return None

    def grad_receiver_group(self, layer: str) -> dist.ProcessGroup | None:
        """Return communication group for preconditioned gradient broadcast.

        This communication group is used for the broadcasts of the gradients
        from the gradient worker to the remaining gradient receivers for the
        layer.
        """
        return None

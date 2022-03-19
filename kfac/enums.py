from __future__ import annotations

from enum import Enum


class AllreduceMethod(Enum):
    ALLREDUCE = 1
    ALLREDUCE_BUCKETED = 2


class DistributedStrategy(Enum):
    """KFAC Distribution Strategy

    Shortcuts for common grad_worker_fractions.
      - COMM_OPT: grad_worker_fraction = 1
      - HYBRID_OPT: grad_worker_fraction = 0.5
      - MEM-OPT: grad_worker_fraction = 1 / world_size

    See https://arxiv.org/pdf/2107.01739.pdf for more details on distribution
    strategies.
    """

    COMM_OPT = 1
    MEM_OPT = 2
    HYBRID_OPT = 3

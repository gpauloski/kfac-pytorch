"""KFAC enum types."""

from __future__ import annotations

from enum import Enum


class AllreduceMethod(Enum):
    """Allreduce method."""

    ALLREDUCE = 1
    ALLREDUCE_BUCKETED = 2


class AssignmentStrategy(Enum):
    """KFAC Factor Distribution Method.

    KFAC assigns factors for second-order computation using a heuristic-based
    longest-processing time greedy algorithm. AssignmentStrategy.COMPUTE
    uses an estimation of the second-order computation time as the heuristic
    and AssignmentStrategy.MEMORY uses the memory requirements of storing
    the second-order results as the heuristic.
    """

    COMPUTE = 1
    MEMORY = 2


class ComputeMethod(Enum):
    """KFAC Second Order Computation Method.

    Controls if eigen decompositions or inverse of the factors will be used
    to precondition the gradients.
    """

    EIGEN = 1
    INVERSE = 2


class DistributedStrategy(Enum):
    """KFAC Distribution Strategy.

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

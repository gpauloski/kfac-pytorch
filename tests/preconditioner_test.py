"""Unit Tests for kfac/preconditioner.py."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from kfac.enums import AllreduceMethod
from kfac.enums import AssignmentStrategy
from kfac.enums import ComputeMethod
from kfac.enums import DistributedStrategy
from kfac.preconditioner import KFACPreconditioner
from testing.distributed import distributed_test
from testing.models import TinyModel


def test_preconditioner_init_raises() -> None:
    """Test KFACPreconditioner argument validation."""
    with pytest.raises(ValueError):
        KFACPreconditioner(TinyModel(), allreduce_bucket_cap_mb=-1)

    KFACPreconditioner(
        TinyModel(),
        compute_eigenvalue_outer_product=True,
        compute_method=ComputeMethod.INVERSE,
        colocate_factors=False,
    )
    with pytest.raises(ValueError):
        KFACPreconditioner(
            TinyModel(),
            compute_eigenvalue_outer_product=True,
            compute_method=ComputeMethod.EIGEN,
            colocate_factors=False,
        )

    with pytest.raises(ValueError):
        KFACPreconditioner(TinyModel(), grad_worker_fraction=2)

    with pytest.raises(ValueError):
        KFACPreconditioner(TinyModel(), grad_worker_fraction=-1)

    @distributed_test(world_size=8)
    def _f() -> None:
        with pytest.raises(ValueError):
            KFACPreconditioner(TinyModel(), grad_worker_fraction=0.33)

    _f()

    with pytest.warns():
        KFACPreconditioner(
            TinyModel(),
            compute_method=ComputeMethod.INVERSE,
            colocate_factors=False,
            grad_worker_fraction=DistributedStrategy.MEM_OPT,
        )


def test_preconditioner_init() -> None:
    """Test KFACPreconditioner initialization."""
    p1 = KFACPreconditioner(TinyModel(), assignment_strategy='memory')
    p2 = KFACPreconditioner(
        TinyModel(),
        assignment_strategy=AssignmentStrategy.MEMORY,
    )
    assert p1.assignment_strategy == p2.assignment_strategy

    p1 = KFACPreconditioner(TinyModel(), compute_method='inverse')
    p2 = KFACPreconditioner(TinyModel(), compute_method=ComputeMethod.INVERSE)
    assert p1.compute_method == p2.compute_method

    @distributed_test(world_size=4)
    def _f() -> None:
        p1 = KFACPreconditioner(TinyModel(), grad_worker_fraction=1)
        p2 = KFACPreconditioner(
            TinyModel(),
            grad_worker_fraction=DistributedStrategy.COMM_OPT,
        )
        assert p1.distributed_strategy == p2.distributed_strategy
        assert p1.grad_worker_fraction == p2.grad_worker_fraction

        p1 = KFACPreconditioner(
            TinyModel(),
            grad_worker_fraction=DistributedStrategy.HYBRID_OPT,
        )
        assert p1.grad_worker_fraction == 0.5

        p1 = KFACPreconditioner(
            TinyModel(),
            grad_worker_fraction=DistributedStrategy.MEM_OPT,
        )
        assert p1.grad_worker_fraction == 0.25

        p1 = KFACPreconditioner(TinyModel(), grad_worker_fraction=0)
        assert p1.grad_worker_fraction == 0.25
        assert p1.distributed_strategy == DistributedStrategy.MEM_OPT

        p1 = KFACPreconditioner(
            TinyModel(),
            grad_worker_fraction=0.5,
        )
        assert p1.distributed_strategy == DistributedStrategy.HYBRID_OPT

    _f()

    p1 = KFACPreconditioner(TinyModel(), allreduce_bucket_cap_mb=25)
    assert p1.allreduce_method == AllreduceMethod.ALLREDUCE_BUCKETED

    p1 = KFACPreconditioner(TinyModel(), allreduce_bucket_cap_mb=0)
    assert p1.allreduce_method == AllreduceMethod.ALLREDUCE


def test_preconditioner_logging(caplog: Any) -> None:
    """Test KFACPreconditioner logs relevant info."""
    caplog.set_level(logging.INFO)

    KFACPreconditioner(TinyModel(), loglevel=logging.DEBUG)
    assert len(caplog.records) == 0
    caplog.clear()

    KFACPreconditioner(TinyModel(), loglevel=logging.INFO)
    messages = [r.getMessage() for r in caplog.records]
    # Should register two layers in TinyModel and have a record for each
    assert sum('Registered' in msg for msg in messages) == 2
    # Should print KAISAAssignment once
    assert sum('KAISAAssignment' in msg for msg in messages) == 1

"""Unit Tests for kfac/gpt_neox.py."""
from __future__ import annotations

import contextlib
import logging
from typing import Any
from unittest import mock

import pytest
from deepspeed.pipe import PipelineModule

from kfac.gpt_neox import GPTNeoXAssignment
from kfac.gpt_neox import GPTNeoXKFACPreconditioner
from testing.distributed import distributed_test
from testing.models import sequential_model


@pytest.mark.parametrize(
    'work,ranks',
    (
        ({}, [0, 1]),
        ({'l1': {'A': 1, 'G': 1}, 'l2': {'A': 1, 'G': 1}}, [0]),
        ({'l1': {'A': 1, 'G': 1}, 'l2': {'A': 1, 'G': 1}}, [0, 1]),
        ({'l1': {'A': 1, 'G': 1}, 'l2': {'A': 1, 'G': 1}}, [0, 1, 2]),
    ),
)
def test_gpt_neox_assignment(
    work: dict[str, dict[str, float]],
    ranks: list[int],
) -> None:
    """Test GPTNeoXAssignment."""
    with pytest.raises(ValueError, match='member'):
        GPTNeoXAssignment(
            work,
            local_rank=99999,
            data_parallel_ranks=ranks,
            data_parallel_group=ranks,
        )

    assignments = []
    for rank in ranks:
        assignment = GPTNeoXAssignment(
            work,
            local_rank=rank,
            data_parallel_ranks=ranks,
            data_parallel_group=ranks,
        )
        assignments.append((rank, assignment))

    for rank, assignment in assignments:
        # GPTNeoXAssignment uses MEM-OPT so we should always broadcast
        # gradients and never inverses.
        assert assignment.broadcast_gradients()
        assert not assignment.broadcast_inverses()

        assert set(assignment.get_layers()) == set(work.keys())
        for layer, factors in work.items():
            assert set(assignment.get_factors(layer)) == set(factors.keys())

        for layer, factors in work.items():
            inv_workers = [
                assignment.inv_worker(layer, factor) for factor in factors
            ]
            # Check every factor is assigned to same inv worker
            assert inv_workers.count(inv_workers[0]) == len(inv_workers)
            assert inv_workers[0] in ranks

            assert assignment.is_grad_worker(layer) == (rank == inv_workers[0])

        for layer in work:
            with pytest.raises(NotImplementedError):
                assignment.grad_worker_group(layer)

    for layer in work:
        src_grad_workers = [
            assignment.src_grad_worker(layer) for _, assignment in assignments
        ]
        assert src_grad_workers.count(src_grad_workers[0]) == len(
            src_grad_workers,
        )

        groups = [
            assignment.factor_group(layer) for _, assignment in assignments
        ]
        groups += [
            assignment.grad_receiver_group(layer)
            for _, assignment in assignments
        ]
        assert groups.count(groups[0]) == len(groups)


@pytest.mark.parametrize(
    'work,ranks,expected',
    (
        (
            {'l1': {'A': 1, 'G': 1}, 'l2': {'A': 1, 'G': 1}},
            [2],
            {'l1': {'A': 2, 'G': 2}, 'l2': {'A': 2, 'G': 2}},
        ),
        (
            {'l1': {'A': 1, 'G': 1}, 'l2': {'A': 1, 'G': 1}},
            [0, 1],
            {'l1': {'A': 1, 'G': 1}, 'l2': {'A': 0, 'G': 0}},
        ),
        (
            {'l1': {'A': 1, 'G': 1}, 'l2': {'A': 1, 'G': 1}},
            [0, 1, 2],
            {'l1': {'A': 1, 'G': 1}, 'l2': {'A': 0, 'G': 0}},
        ),
        (
            {
                'l1': {'A': 10, 'G': 10},
                'l2': {'A': 1, 'G': 1},
                'l3': {'A': 1, 'G': 1},
            },
            [0, 1],
            {
                'l1': {'A': 0, 'G': 0},
                'l2': {'A': 1, 'G': 1},
                'l3': {'A': 1, 'G': 1},
            },
        ),
    ),
)
def test_gpt_neox_assignment_load_balancing(
    work: dict[str, dict[str, float]],
    ranks: list[int],
    expected: dict[str, dict[str, float]],
) -> None:
    """Test GPTNeoXAssignment load balancing."""
    for rank in ranks:
        assignment = GPTNeoXAssignment(
            work,
            local_rank=rank,
            data_parallel_ranks=ranks,
            data_parallel_group=ranks,
        )

        for layer, factors in expected.items():
            for factor in factors:
                inv_worker = assignment.inv_worker(layer, factor)
                assert inv_worker == factors[factor]
            assert assignment.is_grad_worker(layer) == (inv_worker == rank)
            assert inv_worker == assignment.src_grad_worker(layer)


@pytest.mark.parametrize(
    'num_stages,kwargs',
    (
        (1, {'assignment_strategy': 'memory'}),
        (2, {'compute_method': 'inverse'}),
        (4, {'allreduce_bucket_cap_mb': 0}),
    ),
)
def test_gpt_neox_kfac_preconditioner(
    num_stages: int,
    kwargs: dict[str, Any],
) -> None:
    """Test GPTNeoXKFACPreconditioner."""

    @distributed_test(world_size=num_stages)
    def check() -> None:
        model = sequential_model(layers=10, hidden_dim=32)
        with (
            mock.patch.object(PipelineModule, 'to', mock.MagicMock()),
            contextlib.redirect_stdout(None),
            contextlib.redirect_stderr(None),
        ):
            logging.disable(10000)
            model = PipelineModule(layers=model, num_stages=num_stages)
            GPTNeoXKFACPreconditioner(model, **kwargs)

    check()


def test_input_validation() -> None:
    """Test GPTNeoXKFACPreconditioner input validation."""

    @distributed_test(world_size=1)
    def check() -> None:
        model = sequential_model(1, 1)
        with pytest.raises(ValueError, match='PipelineModule'):
            GPTNeoXKFACPreconditioner(model)

        with (
            mock.patch.object(PipelineModule, 'to', mock.MagicMock()),
            contextlib.redirect_stdout(None),
            contextlib.redirect_stderr(None),
        ):
            logging.disable(10000)
            model = PipelineModule(layers=model, num_stages=1)
        with pytest.raises(ValueError, match='allreduce_bucket_cap_mb'):
            GPTNeoXKFACPreconditioner(model, allreduce_bucket_cap_mb=-1)

    check()


def test_model_parallel_unsupported() -> None:
    """Test DeepSpeed model parallelism unsupported."""

    @distributed_test(world_size=2)
    def check() -> None:
        model = sequential_model(layers=10, hidden_dim=32)
        with (
            mock.patch.object(PipelineModule, 'to', mock.MagicMock()),
            contextlib.redirect_stdout(None),
            contextlib.redirect_stderr(None),
        ):
            logging.disable(10000)
            model = PipelineModule(layers=model, num_stages=1)
            model.mpu().get_data_parallel_world_size = lambda: 1
            with pytest.raises(ValueError):
                GPTNeoXKFACPreconditioner(model)

    check()

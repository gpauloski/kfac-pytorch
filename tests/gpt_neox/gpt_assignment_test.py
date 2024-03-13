"""Unit Tests for kfac/gpt_neox.py."""

from __future__ import annotations

from unittest import mock

import pytest
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology

from kfac.gpt_neox.assignment import GPTNeoXAssignment
from kfac.gpt_neox.mpu import get_group_with_rank


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
    with pytest.raises(TypeError):
        GPTNeoXAssignment(
            work,
            local_rank=99999,
            topology=object(),
            data_parallel_group=None,
            model_parallel_group=None,
        )

    assignments = []
    topology = PipeModelDataParallelTopology(1, len(ranks), 1)
    for rank in ranks:
        assignment = GPTNeoXAssignment(
            work,
            local_rank=rank,
            topology=topology,
            data_parallel_group=None,
            model_parallel_group=None,
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

            model_parallel_peers = get_group_with_rank(
                rank,
                topology.get_axis_comm_lists('model'),
            )
            assert assignment.is_grad_worker(layer) == (
                rank in model_parallel_peers
                and inv_workers[0] in model_parallel_peers
            )

        for layer in work:
            with pytest.raises(NotImplementedError):
                assignment.grad_worker_group(layer)

    for layer in work:
        src_grad_workers = [
            assignment.src_grad_worker(layer) for _, assignment in assignments
        ]

        assert src_grad_workers.count(src_grad_workers[0]) == 1

        factor_workers = set()
        for factor in work[layer]:
            factor_workers.add(assignment.factor_worker(layer, factor))
        assert len(factor_workers) == 1

        groups = [
            assignment.factor_group(layer, 'A')
            for _, assignment in assignments
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
            [0],
            {'l1': {'A': 0, 'G': 0}, 'l2': {'A': 0, 'G': 0}},
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
    topology = PipeModelDataParallelTopology(1, len(ranks), 1)
    for rank in ranks:
        assignment = GPTNeoXAssignment(
            work,
            local_rank=rank,
            topology=topology,
            data_parallel_group=None,
            model_parallel_group=None,
        )

        for layer, factors in expected.items():
            for factor in factors:
                inv_worker = assignment.inv_worker(layer, factor)
                assert inv_worker == factors[factor]

            model_parallel_peers = get_group_with_rank(
                rank,
                topology.get_axis_comm_lists('model'),
            )
            assert assignment.is_grad_worker(layer) == (
                rank in model_parallel_peers
                and inv_worker in model_parallel_peers
            )


def test_reuse_comm_groups() -> None:
    """Test that we reuse existing comm groups when possible."""
    with mock.patch('torch.distributed.new_group', return_value=-1):
        topology = PipeModelDataParallelTopology(2, 1, 2)
        assignment = GPTNeoXAssignment(
            {},
            local_rank=0,
            topology=topology,
            data_parallel_group=-2,  # type: ignore
            model_parallel_group=-3,  # type: ignore
        )
        assert (
            assignment.pipe_parallel_peer_group
            == assignment.data_parallel_group
        )

        topology = PipeModelDataParallelTopology(2, 2, 2)
        assignment = GPTNeoXAssignment(
            {},
            local_rank=0,
            topology=topology,
            data_parallel_group=-2,  # type: ignore
            model_parallel_group=-3,  # type: ignore
        )
        assert (
            assignment.pipe_parallel_peer_group
            != assignment.data_parallel_group
            != assignment.model_parallel_group
        )

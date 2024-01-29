"""Unit Tests for kfac/assignment.py."""

from __future__ import annotations

from typing import Any
from typing import cast
from typing import Sized

import pytest

from kfac.assignment import KAISAAssignment

TEST_WORK = {
    'l1': {'A': 1.0, 'G': 1.0},
    'l2': {'A': 1.0, 'G': 1.0},
    'l3': {'A': 1.0, 'G': 1.0},
    'l4': {'A': 1.0, 'G': 1.0},
    'l5': {'A': 1.0, 'G': 1.0},
    'l6': {'A': 1.0, 'G': 1.0},
    'l7': {'A': 1.0, 'G': 1.0},
    'l8': {'A': 1.0, 'G': 1.0},
    'l9': {'A': 1.0, 'G': 1.0},
    'l10': {'A': 1.0, 'G': 1.0},
    'l11': {'A': 1.0, 'G': 1.0},
    'l12': {'A': 1.0, 'G': 1.0},
    'l13': {'A': 1.0, 'G': 1.0},
    'l14': {'A': 1.0, 'G': 1.0},
    'l15': {'A': 1.0, 'G': 1.0},
    'l16': {'A': 1.0, 'G': 1.0},
}

partition_grad_workers = KAISAAssignment.partition_grad_workers
partition_grad_receivers = KAISAAssignment.partition_grad_receivers


def identity(x: Any) -> Any:
    """Identity function."""
    return x


@pytest.mark.parametrize('world_size,grad_workers', ((4, 8), (4, 3), (0, 2)))
def test_partition_grad_workers_input_check(
    world_size: int,
    grad_workers: int,
) -> None:
    """Test partition_grad_workers raises."""
    with pytest.raises(ValueError):
        partition_grad_workers(world_size, grad_workers)


@pytest.mark.parametrize('world_size,grad_workers', ((4, 8), (4, 3), (0, 2)))
def test_partition_grad_receivers_input_check(
    world_size: int,
    grad_workers: int,
) -> None:
    """Test partition_grad_receivers raises."""
    with pytest.raises(ValueError):
        partition_grad_receivers(world_size, grad_workers)


@pytest.mark.parametrize(
    'world_size,grad_workers,expected',
    (
        (16, 8, [[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]),
        (
            16,
            4,
            [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]],
        ),
        (
            16,
            2,
            [
                [0, 8],
                [1, 9],
                [2, 10],
                [3, 11],
                [4, 12],
                [5, 13],
                [6, 14],
                [7, 15],
            ],
        ),
        (8, 8, [[0, 1, 2, 3, 4, 5, 6, 7]]),
        (8, 4, [[0, 2, 4, 6], [1, 3, 5, 7]]),
        (8, 2, [[0, 4], [1, 5], [2, 6], [3, 7]]),
        (8, 1, [[0], [1], [2], [3], [4], [5], [6], [7]]),
        (2, 1, [[0], [1]]),
    ),
)
def test_partition_grad_workers(
    world_size: int,
    grad_workers: int,
    expected: list[list[int]],
) -> None:
    """Test partition_grad_workers."""
    _expected = {frozenset(ranks) for ranks in expected}
    result = partition_grad_workers(world_size, grad_workers)
    assert result == _expected


@pytest.mark.parametrize(
    'world_size,grad_workers,expected',
    (
        (
            16,
            8,
            [
                [0, 1],
                [2, 3],
                [4, 5],
                [6, 7],
                [8, 9],
                [10, 11],
                [12, 13],
                [14, 15],
            ],
        ),
        (
            16,
            4,
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
        ),
        (16, 2, [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]),
        (8, 8, [[0], [1], [2], [3], [4], [5], [6], [7]]),
        (8, 4, [[0, 1], [2, 3], [4, 5], [6, 7]]),
        (8, 2, [[0, 1, 2, 3], [4, 5, 6, 7]]),
        (8, 1, [[0, 1, 2, 3, 4, 5, 6, 7]]),
        (2, 1, [[0, 1]]),
        (2, 2, [[0], [1]]),
        (1, 1, [[0]]),
    ),
)
def test_partition_grad_receivers(
    world_size: int,
    grad_workers: int,
    expected: list[list[int]],
) -> None:
    """Test partition_grad_receivers."""
    _expected = {frozenset(ranks) for ranks in expected}
    result = partition_grad_receivers(world_size, grad_workers)
    assert result == _expected


@pytest.mark.parametrize(
    'grad_worker_fraction,local_rank,world_size',
    ((2, 0, 1), (-1, 0, 1), (1, 1, 1), (1, -1, 2), (1, 1, -2), (0.33, 0, 8)),
)
def test_kaisa_assignment_input_check(
    grad_worker_fraction: float,
    local_rank: int,
    world_size: int,
) -> None:
    """Test KAISAAssignment raises."""
    with pytest.raises(ValueError):
        KAISAAssignment(
            {},
            local_rank=local_rank,
            world_size=world_size,
            grad_worker_fraction=grad_worker_fraction,
            group_func=identity,
        )


@pytest.mark.parametrize(
    'world_size,grad_worker_fraction,expected_grad_workers',
    (
        (1, 1, 1),
        (1, 0, 1),
        (1, 0.5, 1),
        (4, 1, 4),
        (4, 0, 1),
        (4, 0.5, 2),
        (8, 0.25, 2),
    ),
)
def test_kaisa_assignment_initialize(
    world_size: int,
    grad_worker_fraction: float,
    expected_grad_workers: int,
) -> None:
    """Test KAISAAssignment grad worker group sizes."""
    for i in range(world_size):
        assignment = KAISAAssignment(
            {},
            local_rank=i,
            world_size=world_size,
            grad_worker_fraction=grad_worker_fraction,
            group_func=identity,
        )
        assert assignment.grad_workers == expected_grad_workers


@pytest.mark.parametrize(
    'work,worker_groups,world_size,colocate_factors,expected',
    (
        (
            # Test empty work
            {},
            [[0], [1], [2, 3]],
            4,
            False,
            {},
        ),
        (
            # Test world_size = 1 with same work costs
            {'l1': {'A': 1, 'G': 1}, 'l2': {'A': 1, 'G': 1}},
            [[0]],
            1,
            False,
            {'l1': {'A': 0, 'G': 0}, 'l2': {'A': 0, 'G': 0}},
        ),
        (
            # Test world_size = 1 with different work costs
            {'l1': {'A': 1, 'G': 2}, 'l2': {'A': 3, 'G': 4}},
            [[0]],
            1,
            False,
            {'l1': {'A': 0, 'G': 0}, 'l2': {'A': 0, 'G': 0}},
        ),
        (
            # Test basic assignment with colocate=True
            {'l1': {'A': 1, 'G': 2}, 'l2': {'A': 3, 'G': 4}},
            [[0, 1, 2, 3]],
            4,
            True,
            {'l1': {'A': 1, 'G': 1}, 'l2': {'A': 0, 'G': 0}},
        ),
        (
            # Test basic assignment with colocate=False
            {'l1': {'A': 1, 'G': 2}, 'l2': {'A': 3, 'G': 4}},
            [[0, 1, 2, 3]],
            4,
            False,
            {'l1': {'A': 3, 'G': 2}, 'l2': {'A': 1, 'G': 0}},
        ),
        (
            # Test world_size > work items and work items == 1
            {'l1': {'A': 1}},
            [[0, 1, 2, 3, 4, 5, 6, 7]],
            8,
            False,
            {'l1': {'A': 0}},
        ),
        (
            # Test world_size > work items and work items > 1
            {'l1': {'A': 1, 'G': 2}},
            [[0, 1, 2, 3, 4, 5, 6, 7]],
            8,
            False,
            {'l1': {'A': 1, 'G': 0}},
        ),
        (
            # Test sort by descending cost then descending key
            {'l1': {'A': 1, 'G': 1}},
            [[0, 1, 2, 3, 4, 5, 6, 7]],
            8,
            False,
            {'l1': {'A': 1, 'G': 0}},
        ),
        (
            # Test more complex assignment 1
            {
                'l1': {'A': 1, 'B': 100, 'C': 5, 'D': 2},
                'l2': {'A': 0.01, 'B': 0.01, 'C': 0.01, 'D': 0.01},
            },
            [[0, 1, 2, 3, 4, 5, 6, 7]],
            8,
            False,
            {
                'l1': {'A': 3, 'B': 0, 'C': 1, 'D': 2},
                'l2': {'A': 7, 'B': 6, 'C': 5, 'D': 4},
            },
        ),
        (
            # Test more complex assignment 2
            {
                'l1': {'A': 1, 'B': 100, 'C': 5, 'D': 2},
                'l2': {
                    'A': 0.01,
                    'B': 0.01,
                    'C': 0.01,
                    'D': 0.01,
                    'E': 0.01,
                    'F': 0.01,
                    'G': 0.01,
                    'H': 0.01,
                },
            },
            [[0, 1, 2, 3, 4, 5, 6, 7]],
            8,
            False,
            {
                'l1': {'A': 3, 'B': 0, 'C': 1, 'D': 2},
                'l2': {
                    'A': 7,
                    'B': 6,
                    'C': 5,
                    'D': 4,
                    'E': 7,
                    'F': 6,
                    'G': 5,
                    'H': 4,
                },
            },
        ),
        (
            # Test more complex assignment 3
            {
                'l1': {'A': 1, 'B': 100, 'C': 5, 'D': 2},
                'l2': {
                    'A': 0.01,
                    'B': 0.01,
                    'C': 0.01,
                    'D': 0.01,
                    'E': 0.01,
                    'F': 0.01,
                    'G': 0.01,
                    'H': 0.01,
                },
            },
            [[0, 1]],
            2,
            False,
            {
                'l1': {'A': 1, 'B': 0, 'C': 1, 'D': 1},
                'l2': {
                    'A': 1,
                    'B': 1,
                    'C': 1,
                    'D': 1,
                    'E': 1,
                    'F': 1,
                    'G': 1,
                    'H': 1,
                },
            },
        ),
        (
            # Test 8 workers with 2 worker groups
            {
                'l1': {'A': 1, 'B': 100, 'C': 5, 'D': 2},
                'l2': {'A': 0.01, 'B': 0.01, 'C': 0.01, 'D': 0.01},
            },
            [[0, 2, 4, 6], [1, 3, 5, 7]],
            8,
            False,
            {
                'l1': {'A': 6, 'B': 0, 'C': 2, 'D': 4},
                'l2': {'A': 7, 'B': 5, 'C': 3, 'D': 1},
            },
        ),
        (
            # Test 8 worker with 2 worker groups (more work)
            {
                'l1': {'A': 1, 'B': 100, 'C': 5, 'D': 2},
                'l2': {
                    'A': 0.01,
                    'B': 0.01,
                    'C': 0.01,
                    'D': 0.01,
                    'E': 0.01,
                    'F': 0.01,
                    'G': 0.01,
                    'H': 0.01,
                },
            },
            [[0, 2, 4, 6], [1, 3, 5, 7]],
            8,
            False,
            {
                'l1': {'A': 6, 'B': 0, 'C': 2, 'D': 4},
                'l2': {
                    'A': 7,
                    'B': 5,
                    'C': 3,
                    'D': 1,
                    'E': 7,
                    'F': 5,
                    'G': 3,
                    'H': 1,
                },
            },
        ),
        # The next to tests test that for world size 2, 2 worker groups
        # and colocate=False gives the same assignment as 1 worker group
        # and colocate=True
        (
            {
                'l1': {'A': 1, 'B': 100, 'C': 5, 'D': 2},
                'l2': {'A': 0.01, 'B': 0.01, 'C': 0.01, 'D': 0.01},
            },
            [[0], [1]],
            2,
            False,
            {
                'l1': {'A': 0, 'B': 0, 'C': 0, 'D': 0},
                'l2': {'A': 1, 'B': 1, 'C': 1, 'D': 1},
            },
        ),
        (
            {
                'l1': {'A': 1, 'B': 100, 'C': 5, 'D': 2},
                'l2': {'A': 0.01, 'B': 0.01, 'C': 0.01, 'D': 0.01},
            },
            [[0, 1]],
            2,
            True,
            {
                'l1': {'A': 0, 'B': 0, 'C': 0, 'D': 0},
                'l2': {'A': 1, 'B': 1, 'C': 1, 'D': 1},
            },
        ),
    ),
)
def test_kaisa_assignment_greedy_assignment(
    work: dict[str, dict[str, float]],
    worker_groups: list[list[int]],
    world_size: int,
    colocate_factors: bool,
    expected: dict[str, dict[str, int]],
) -> None:
    """Test KAISAAssignment greedy assignment."""
    assert expected == KAISAAssignment.greedy_assignment(
        work,
        worker_groups,
        world_size,
        colocate_factors,
    )


@pytest.mark.parametrize(
    'world_size,grad_worker_fraction,colocate_factors,grad_worker_group_size,'
    'grad_receiver_group_size',
    (
        (1, 1, True, 1, 1),
        (1, 0, False, 1, 1),
        # MEM-OPT
        (4, 0.25, False, 1, 4),
        (4, 0.25, True, 1, 4),
        # HYBRID-OPT
        (4, 0.5, False, 2, 2),
        (4, 0.5, True, 2, 2),
        # COMM-OPT
        (4, 1, False, 4, 1),
        (4, 1, True, 4, 1),
        # 16 workers, all grad_worker_fractions
        (16, 1 / 16, False, 1, 16),
        (16, 1 / 8, False, 2, 8),
        (16, 1 / 4, False, 4, 4),
        (16, 1 / 2, False, 8, 2),
        (16, 1, False, 16, 1),
    ),
)
def test_kaisa_assignment_group_sizes(
    world_size: int,
    grad_worker_fraction: float,
    colocate_factors: bool,
    grad_worker_group_size: int,
    grad_receiver_group_size: int,
) -> None:
    """Test KAISAAssignment grad_worker_fraction group sizes."""
    assignments = [
        KAISAAssignment(
            TEST_WORK,
            local_rank=rank,
            world_size=world_size,
            grad_worker_fraction=grad_worker_fraction,
            group_func=identity,
            colocate_factors=colocate_factors,
        )
        for rank in range(world_size)
    ]

    layer_count = len(TEST_WORK)
    for assignment in assignments:
        # Check assigned layers and factors are correct
        layers = assignment.get_layers()
        assert len(set(layers)) == layer_count
        for layer in layers:
            assert len(set(assignment.get_factors(layer))) == 2

        # Check repr: one line for each layer + one prefix and postfix line
        assert repr(assignment).count('\n') + 1 == layer_count + 2

        # Check broadcast decisions are correct
        broadcast_gradients = grad_worker_group_size < world_size
        assert assignment.broadcast_gradients() == broadcast_gradients
        broadcast_inverses = grad_worker_group_size > 1
        assert assignment.broadcast_inverses() == broadcast_inverses

    for layer in TEST_WORK:
        # Check all ranks have the same inv worker for each factor
        assert len({a.inv_worker(layer, 'A') for a in assignments}) == 1
        assert len({a.inv_worker(layer, 'G') for a in assignments}) == 1

        # Check number of unique src grad workers == grad worker group size
        assert (
            len({a.src_grad_worker(layer) for a in assignments})
            == grad_worker_group_size
        )

        # Check number of ranks that report they are a grad worker for a layer
        # equals the expected number of grad workers
        assert (
            sum(a.is_grad_worker(layer) for a in assignments)
            == grad_worker_group_size
        )

        for assignment in assignments:
            if colocate_factors:
                assert assignment.inv_worker(
                    layer,
                    'A',
                ) == assignment.inv_worker(layer, 'G')
            assert 0 <= assignment.inv_worker(layer, 'A') < world_size
            assert 0 <= assignment.inv_worker(layer, 'G') < world_size
            assert 0 <= assignment.src_grad_worker(layer) < world_size
            assert (
                len(cast(Sized, assignment.grad_worker_group(layer)))
                == grad_worker_group_size
            )
            assert (
                len(cast(Sized, assignment.grad_receiver_group(layer)))
                == grad_receiver_group_size
            )


def test_kaisa_factor_allreduce_groups() -> None:
    """Test KAISA assignment factor groups is always None."""
    for rank in [0, 1, 2, 3]:
        assignment = KAISAAssignment(
            TEST_WORK,
            local_rank=rank,
            world_size=4,
            grad_worker_fraction=0.5,
            group_func=lambda ranks: None,
        )
        for layer in TEST_WORK:
            assert assignment.factor_group(layer, 'A') is None
            assert assignment.factor_group(layer, 'G') is None

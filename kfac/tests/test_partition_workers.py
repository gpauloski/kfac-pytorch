"""Partition Workers Unit Tests"""
from pytest import raises

from kfac.allocator import WorkerAllocator

partition_grad_workers = WorkerAllocator.partition_grad_workers
partition_grad_receivers = WorkerAllocator.partition_grad_receivers


def test_input_check() -> None:
    with raises(ValueError):
        partition_grad_workers(4, 8)

    with raises(ValueError):
        partition_grad_workers(4, 3)

    with raises(ValueError):
        partition_grad_workers(0, 2)

    with raises(ValueError):
        partition_grad_receivers(4, 8)

    with raises(ValueError):
        partition_grad_receivers(4, 3)

    with raises(ValueError):
        partition_grad_receivers(0, 2)


def test_partition_grad_workers() -> None:
    def _check(world_size, grad_workers, expected):
        expected = {frozenset(ranks) for ranks in expected}
        result = partition_grad_workers(world_size, grad_workers)
        assert result == expected

    _check(16, 8, [[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]])
    _check(
        16, 4, [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]
    )
    _check(
        16,
        2,
        [[0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [5, 13], [6, 14], [7, 15]],
    )
    _check(8, 8, [[0, 1, 2, 3, 4, 5, 6, 7]])
    _check(8, 4, [[0, 2, 4, 6], [1, 3, 5, 7]])
    _check(8, 2, [[0, 4], [1, 5], [2, 6], [3, 7]])
    _check(8, 1, [[0], [1], [2], [3], [4], [5], [6], [7]])
    _check(2, 1, [[0], [1]])
    _check(2, 2, [[0, 1]])
    _check(1, 1, [[0]])


def test_partition_grad_receivers() -> None:
    def _check(world_size, grad_workers, expected):
        expected = {frozenset(ranks) for ranks in expected}
        result = partition_grad_receivers(world_size, grad_workers)
        assert result == expected

    _check(
        16,
        8,
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]],
    )
    _check(
        16, 4, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    )
    _check(16, 2, [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]])
    _check(8, 8, [[0], [1], [2], [3], [4], [5], [6], [7]])
    _check(8, 4, [[0, 1], [2, 3], [4, 5], [6, 7]])
    _check(8, 2, [[0, 1, 2, 3], [4, 5, 6, 7]])
    _check(8, 1, [[0, 1, 2, 3, 4, 5, 6, 7]])
    _check(2, 1, [[0, 1]])
    _check(2, 2, [[0], [1]])
    _check(1, 1, [[0]])

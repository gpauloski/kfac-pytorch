"""Partition Workers Unit Tests"""
import kfac

from pytest import raises


def test_input_check() -> None:
    with raises(ValueError):
        kfac.WorkerAllocator(2, 0, 1)

    with raises(ValueError):
        kfac.WorkerAllocator(-1, 0, 1)

    with raises(ValueError):
        kfac.WorkerAllocator(0.25, 0, 2)

    with raises(ValueError):
        kfac.WorkerAllocator(1, 1, 1)

    with raises(ValueError):
        kfac.WorkerAllocator(1, -1, 2)

    with raises(ValueError):
        kfac.WorkerAllocator(1, 1, -2)


def test_initialize() -> None:
    allocator1 = kfac.WorkerAllocator(1, 0, 2)
    allocator2 = kfac.WorkerAllocator(1, 1, 2)

    # Check worker partitions are same regardless of rank
    assert allocator1.grad_worker_groups == allocator2.grad_worker_groups
    assert allocator1.grad_receiver_groups == allocator2.grad_receiver_groups

    with raises(ValueError):
        # We did not pass group_func to WorkerAllocator so the communication
        # groups should not have been initialized and this should fail.
        allocator1.comm_group({0, 1})


def test_unconstrained_assignment() -> None:
    def _check(world_size, work, assignment):
        allocator1 = kfac.WorkerAllocator(1, 0, world_size)
        assignment1 = allocator1.assign_layer_work(work)
        assert assignment == assignment1
        # Check ranks give identical assignments
        if world_size > 1:
            allocator2 = kfac.WorkerAllocator(1, 1, world_size)
            assignment2 = allocator2.assign_layer_work(work)
            assert assignment1 == assignment2

    work = {
        "layer1": [1, 100, 5, 2],
        "layer2": [0.1, 0.1, 0.1, 0.1],
    }
    assignment = {
        "layer1": [3, 0, 1, 2],
        "layer2": [4, 5, 6, 7],
    }
    _check(8, work, assignment)

    work = {
        "layer1": [1, 100, 5, 2],
        "layer2": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    }
    assignment = {
        "layer1": [3, 0, 1, 2],
        "layer2": [4, 5, 6, 7, 4, 5, 6, 7, 4],
    }
    _check(8, work, assignment)
    assignment = {
        "layer1": [1, 0, 1, 1],
        "layer2": [1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    _check(2, work, assignment)

    _check(8, {"layer1": [1]}, {"layer1": [0]})
    _check(8, {"layer1": [1, 1]}, {"layer1": [0, 1]})
    _check(8, {"layer1": [1, 2]}, {"layer1": [1, 0]})
    _check(8, {"layer1": []}, {"layer1": []})
    _check(1, {"layer1": [1, 1, 1, 1]}, {"layer1": [0, 0, 0, 0]})
    _check(2, {"layer1": [1, 1, 1, 1]}, {"layer1": [0, 1, 0, 1]})


def test_constrained_assignment() -> None:
    def _check(world_size, work, groups, assignment):
        allocator1 = kfac.WorkerAllocator(1, 0, world_size)
        assignment1 = allocator1.assign_layer_work(work, groups)
        assert assignment == assignment1
        # Check ranks give identical assignments
        if world_size > 1:
            allocator2 = kfac.WorkerAllocator(1, 1, world_size)
            assignment2 = allocator2.assign_layer_work(work, groups)
            assert assignment1 == assignment2

    work = {
        "layer1": [1, 100, 5, 2],
        "layer2": [0.1, 0.1, 0.1, 0.1],
    }
    group = [[0, 2, 4, 6], [1, 3, 5, 7]]
    assignment = {
        "layer1": [6, 0, 2, 4],
        "layer2": [1, 3, 5, 7],
    }
    _check(8, work, group, assignment)

    work = {
        "layer1": [1, 100, 5, 2],
        "layer2": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    }
    group = [[0, 2, 4, 6], [1, 3, 5, 7]]
    assignment = {
        "layer1": [6, 0, 2, 4],
        "layer2": [1, 3, 5, 7, 1, 3, 5, 7, 1],
    }
    _check(8, work, group, assignment)
    group = [[0], [1]]
    assignment = {
        "layer1": [0, 0, 0, 0],
        "layer2": [1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    _check(2, work, group, assignment)
    group = [[0, 1]]
    assignment = {
        "layer1": [1, 0, 1, 1],
        "layer2": [1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    _check(2, work, group, assignment)

    _check(8, {"layer1": [1]}, [[0, 1, 2, 3, 4, 5, 6, 7]], {"layer1": [0]})
    _check(
        8, {"layer1": [1, 1]}, [[0, 1, 2, 3, 4, 5, 6, 7]], {"layer1": [0, 1]}
    )
    _check(
        8, {"layer1": [1, 2]}, [[0, 1, 2, 3, 4, 5, 6, 7]], {"layer1": [1, 0]}
    )
    _check(8, {"layer1": []}, [[0, 1, 2, 3, 4, 5, 6, 7]], {"layer1": []})
    _check(1, {"layer1": [1, 1, 1, 1]}, [[0]], {"layer1": [0, 0, 0, 0]})
    _check(2, {"layer1": [1, 1, 1, 1]}, [[0, 1]], {"layer1": [0, 1, 0, 1]})
    _check(2, {"layer1": [1, 1, 1, 1]}, [[0], [1]], {"layer1": [0, 0, 0, 0]})


def test_get_groups_comm_opt() -> None:
    world_size = 4
    grad_worker_frac = 1
    # Pass lambda x: x so communication groups are just the sets of ranks
    # for easy checking
    allocator1 = kfac.WorkerAllocator(
        grad_worker_frac, 0, world_size, lambda x: x
    )
    allocator2 = kfac.WorkerAllocator(
        grad_worker_frac, 1, world_size, lambda x: x
    )

    for rank in range(world_size):
        assert allocator1.get_grad_worker_ranks(rank) == set(range(world_size))
        assert allocator2.get_grad_worker_ranks(rank) == set(range(world_size))
        assert allocator1.get_grad_receiver_ranks() == set([0])
        assert allocator2.get_grad_receiver_ranks() == set([1])

        assert allocator1.get_grad_worker_group(rank) == set(range(world_size))
        assert allocator2.get_grad_worker_group(rank) == set(range(world_size))
        assert allocator1.get_grad_receiver_group() == set([0])
        assert allocator2.get_grad_receiver_group() == set([1])

        assert allocator1.get_grad_src_rank(rank) == 0
        assert allocator2.get_grad_src_rank(rank) == 1


def test_get_groups_mem_opt() -> None:
    world_size = 4
    grad_worker_frac = 0.25
    # Pass lambda x: x so communication groups are just the sets of ranks
    # for easy checking
    allocator1 = kfac.WorkerAllocator(
        grad_worker_frac, 0, world_size, lambda x: x
    )
    allocator2 = kfac.WorkerAllocator(
        grad_worker_frac, 1, world_size, lambda x: x
    )

    for rank in range(world_size):
        assert allocator1.get_grad_worker_ranks(rank) == set([rank])
        assert allocator2.get_grad_worker_ranks(rank) == set([rank])
        assert allocator1.get_grad_receiver_ranks() == set(range(world_size))
        assert allocator2.get_grad_receiver_ranks() == set(range(world_size))

        assert allocator1.get_grad_worker_group(rank) == set([rank])
        assert allocator2.get_grad_worker_group(rank) == set([rank])
        assert allocator1.get_grad_receiver_group() == set(range(world_size))
        assert allocator2.get_grad_receiver_group() == set(range(world_size))

        assert allocator1.get_grad_src_rank(rank) == rank
        assert allocator2.get_grad_src_rank(rank) == rank


def test_get_groups_hybrid_opt() -> None:
    world_size = 4
    grad_worker_frac = 0.5
    # Pass lambda x: x so communication groups are just the sets of ranks
    # for easy checking
    allocator1 = kfac.WorkerAllocator(
        grad_worker_frac, 0, world_size, lambda x: x
    )
    allocator2 = kfac.WorkerAllocator(
        grad_worker_frac, 1, world_size, lambda x: x
    )
    allocator3 = kfac.WorkerAllocator(
        grad_worker_frac, 2, world_size, lambda x: x
    )
    allocator4 = kfac.WorkerAllocator(
        grad_worker_frac, 3, world_size, lambda x: x
    )

    for rank in range(world_size):
        wg = [[0, 2], [1, 3]]
        rg = [[0, 1], [2, 3]]
        assert allocator1.get_grad_worker_ranks(rank) == set(wg[rank % 2])
        assert allocator2.get_grad_worker_ranks(rank) == set(wg[rank % 2])
        assert allocator3.get_grad_worker_ranks(rank) == set(wg[rank % 2])
        assert allocator4.get_grad_worker_ranks(rank) == set(wg[rank % 2])
        assert allocator1.get_grad_receiver_ranks() == set(rg[0])
        assert allocator2.get_grad_receiver_ranks() == set(rg[0])
        assert allocator3.get_grad_receiver_ranks() == set(rg[1])
        assert allocator4.get_grad_receiver_ranks() == set(rg[1])

        assert allocator1.get_grad_worker_group(rank) == set(wg[rank % 2])
        assert allocator2.get_grad_worker_group(rank) == set(wg[rank % 2])
        assert allocator3.get_grad_worker_group(rank) == set(wg[rank % 2])
        assert allocator4.get_grad_worker_group(rank) == set(wg[rank % 2])
        assert allocator1.get_grad_receiver_group() == set(rg[0])
        assert allocator2.get_grad_receiver_group() == set(rg[0])
        assert allocator3.get_grad_receiver_group() == set(rg[1])
        assert allocator4.get_grad_receiver_group() == set(rg[1])

        def _inter(local_rank):
            return list(set(wg[rank % 2]) & set(rg[local_rank // 2]))[0]

        assert allocator1.get_grad_src_rank(rank) == _inter(0)
        assert allocator2.get_grad_src_rank(rank) == _inter(1)
        assert allocator3.get_grad_src_rank(rank) == _inter(2)
        assert allocator4.get_grad_src_rank(rank) == _inter(3)

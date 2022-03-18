from __future__ import annotations

import os
from multiprocessing import Process
from typing import Any
from typing import Callable

import torch
from pytest import raises

from kfac.distributed import AllreduceTensorBucket
from kfac.distributed import fill_triu
from kfac.distributed import Future
from kfac.distributed import get_triu
from kfac.distributed import NonSquareTensorError
from kfac.distributed import TorchDistributedCommunicator


def init_distributed(
    func: Callable[..., None],
    rank: int,
    world_size: int,
) -> Callable[..., None]:
    def run(*args: Any, **kwargs: Any) -> None:
        # Determine backend and initialize default distributed group
        if torch.cuda.is_available() and torch.distributed.is_nccl_available():
            backend = 'nccl'
        else:
            backend = 'gloo'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        torch.distributed.init_process_group(backend)

        result = func(*args, **kwargs)
        torch.distributed.destroy_process_group()
        assert result is None

    return run


def run_parallel(
    workers: int,
    func: Callable[..., None],
    *args: Any,
    **kwargs: Any,
) -> None:
    funcs = []
    for rank in range(workers):
        funcs.append(init_distributed(func, rank, workers))

    processes = []
    for f in funcs:
        p = Process(target=f, args=args, kwargs=kwargs)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        assert not p.exitcode


def test_init_distributed() -> None:
    """Test the distributed testing wrappers"""

    def assert_init() -> None:
        assert torch.distributed.is_initialized()
        assert torch.distributed.get_rank() >= 0

    def assert_false() -> None:
        with raises(AssertionError):
            raise AssertionError

    run_parallel(1, assert_init)
    run_parallel(4, assert_init)
    run_parallel(4, assert_false)


def test_triu() -> None:
    """Test upper triangular support methods"""

    def check(shape: list[int]) -> None:
        t = torch.rand(shape)
        # Make symmetric
        t = t + t.T
        t_tri = get_triu(t)
        t_out = fill_triu(t.size(), t_tri)
        assert torch.equal(t, t_out)

    check([1, 1])
    check([2, 2])
    check([4, 4])


def test_allreduce() -> None:
    """Test allreduce"""

    def simple_allreduce(
        shape: list[int],
        symmetric: bool = False,
        expect_raises: type[BaseException] | None = None,
    ) -> None:
        try:
            world_size = torch.distributed.get_world_size()
            comm = TorchDistributedCommunicator()

            t = torch.ones(shape)
            t_res = comm.allreduce(t, symmetric=symmetric)
            if isinstance(t_res, Future):
                t_res = t_res.wait()
            assert torch.sum(t_res).item() == torch.numel(t_res) * world_size
        except Exception as e:
            if expect_raises is not None and not isinstance(e, expect_raises):
                raise

    run_parallel(1, simple_allreduce, [2, 4])
    run_parallel(4, simple_allreduce, [2, 4])
    run_parallel(4, simple_allreduce, [4, 4], symmetric=True)

    # not square for symmetric comms
    run_parallel(
        4,
        simple_allreduce,
        [4, 2],
        symmetric=True,
        expect_raises=NonSquareTensorError,
    )


def test_broadcast() -> None:
    """Test broadcast"""

    def simple_broadcast(
        shape: list[int],
        symmetric: bool = False,
        expect_raises: type[BaseException] | None = None,
    ) -> None:
        try:
            rank = torch.distributed.get_rank()
            comm = TorchDistributedCommunicator()

            t = rank * torch.ones(shape)
            t_res = comm.broadcast(t, src=0, symmetric=symmetric)
            if isinstance(t_res, Future):
                t_res = t_res.wait()
            # Rank 0 will broadcast and it should be all zeros
            assert torch.sum(t_res).item() == 0
        except Exception as e:
            if expect_raises is not None and not isinstance(e, expect_raises):
                raise

    run_parallel(1, simple_broadcast, [2, 4])
    run_parallel(4, simple_broadcast, [2, 4])
    run_parallel(4, simple_broadcast, [4, 4], symmetric=True)

    # not square for symmetric comms
    run_parallel(
        4,
        simple_broadcast,
        [4, 2],
        symmetric=True,
        expect_raises=NonSquareTensorError,
    )


def test_allreduce_tensor_bucket() -> None:
    """Test AllreduceTensorBucket."""

    def allreduce() -> None:
        bucket = AllreduceTensorBucket()

        # Communication operation can only be called once
        assert not bucket.communicated()
        bucket.allreduce()
        assert bucket.communicated()
        with raises(RuntimeError):
            bucket.allreduce()

        bucket = AllreduceTensorBucket()

        t1 = torch.ones([2, 2], dtype=torch.float32)
        f1 = bucket.add_tensor(t1)
        # 4 elements and each takes 4 bytes
        assert bucket.size == 4 * 4

        t2 = torch.ones([2, 4], dtype=torch.float32)
        f2 = bucket.add_tensor(t2)
        # 4 + 8 elements and each element takes 4 bytes
        assert bucket.size == (4 + 8) * 4

        assert not f1.done()
        assert not f2.done()

        allreduce_future = bucket.allreduce()

        if allreduce_future is not None:
            allreduce_tensor = allreduce_future.wait()

        assert allreduce_tensor.nelement() == 12

        s = torch.distributed.get_world_size()

        f1.wait()
        assert f1.done()
        assert torch.equal(s * t1, f1.value())

        f2.wait()
        assert f2.done()
        assert torch.equal(s * t2, f2.value())

    run_parallel(1, allreduce)
    run_parallel(4, allreduce)


def test_allreduce_bucketed() -> None:
    """Test bucketed allreduce."""

    def allreduce(
        shape: list[int],
        tensor_count: int,
        bucket_cap_mb: float,
        symmetric: bool = False,
    ) -> None:
        world_size = torch.distributed.get_world_size()
        comm = TorchDistributedCommunicator(bucket_cap_mb)

        tensors = []
        for _ in range(tensor_count):
            t = torch.ones(shape, dtype=torch.float32)
            tensors.append(
                comm.allreduce_bucketed(t, symmetric=symmetric),
            )
        comm.flush_allreduce_buckets()

        for tensor in tensors:
            if isinstance(tensor, Future):
                tensor = tensor.wait()
            assert torch.sum(tensor).item() == world_size * torch.numel(tensor)

    # Test sum of all tensors less than bucket
    run_parallel(1, allreduce, [2, 2], 4, 1)
    run_parallel(4, allreduce, [100, 100], 4, 1)

    # Test each tensor larger than bucket
    run_parallel(4, allreduce, [100, 100], 4, 0.001)

    # Test symmetric
    run_parallel(1, allreduce, [4, 4], 4, 1, True)
    run_parallel(4, allreduce, [100, 100], 4, 0.001, True)


def test_allreduce_bucketed_grouped() -> None:
    """Test bucketed allreduce with communication groups."""

    def allreduce(
        shape: list[int],
        tensor_count: int,
        bucket_cap_mb: float,
        symmetric: bool = False,
    ) -> None:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        comm = TorchDistributedCommunicator(bucket_cap_mb)

        if world_size == 1:
            group = None
        else:
            # Exclude rank 0
            group = torch.distributed.new_group(
                [i for i in range(world_size) if i >= 1],
            )
        group_size = torch.distributed.get_world_size(group)

        tensors = []
        for _ in range(tensor_count):
            t = torch.ones(shape, dtype=torch.float32)
            if group is None or rank > 0:
                tensors.append(
                    comm.allreduce_bucketed(
                        t,
                        symmetric=symmetric,
                        group=group,
                    ),
                )
        comm.flush_allreduce_buckets()

        for tensor in tensors:
            if isinstance(tensor, Future):
                tensor = tensor.wait()
            if group is not None and rank == 0:
                assert torch.sum(tensor).item() == torch.numel(tensor)
            else:
                assert torch.sum(tensor).item() == group_size * torch.numel(
                    tensor,
                )

    # Test sum of all tensors less than bucket
    run_parallel(1, allreduce, [2, 2], 4, 1)
    run_parallel(4, allreduce, [100, 100], 4, 1)

    # Test each tensor larger than bucket
    run_parallel(4, allreduce, [100, 100], 4, 0.001)

    # Test symmetric
    run_parallel(1, allreduce, [4, 4], 4, 1, True)
    run_parallel(4, allreduce, [100, 100], 4, 0.001, True)

from __future__ import annotations

import os
from multiprocessing import Process
from typing import Callable

import torch
from pytest import raises

from kfac.distributed import fill_triu
from kfac.distributed import Future
from kfac.distributed import get_triu
from kfac.distributed import NonSquareTensorError
from kfac.distributed import TorchDistributedCommunicator


def init_distributed(func: Callable, rank: int, world_size: int) -> Callable:
    def run(*args: list, **kwargs: dict) -> None:
        # Determine backend and initialize default distributed group
        if torch.cuda.is_available() and torch.distributed.is_nccl_available():
            backend = "nccl"
        else:
            backend = "gloo"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        torch.distributed.init_process_group(backend)

        result = func(*args, **kwargs)
        torch.distributed.destroy_process_group()
        assert result is None

    return run


def run_parallel(
    workers: int,
    func: Callable,
    *args: list,
    **kwargs: dict,
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
            assert False

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
        expect_raises: Exception = None,
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
            if not isinstance(e, expect_raises):
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
        expect_raises: Exception = None,
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
            if not isinstance(e, expect_raises):
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


def test_broadcast_bucketing() -> None:
    """Test bucketed broadcast"""

    def broadcast(
        shape: list[int],
        tensor_count: int,
        bucket_cap_mb: float,
        symmetric: bool = False,
    ) -> None:
        rank = torch.distributed.get_rank()
        comm = TorchDistributedCommunicator(bucket_cap_mb)

        tensors = []
        for i in range(tensor_count):
            if rank == 0:
                t = i * torch.ones(shape, dtype=torch.float32)
            else:
                t = torch.zeros(shape, dtype=torch.float32)
            tensors.append(
                comm.broadcast_bucketed(t, src=0, symmetric=symmetric),
            )
        comm.flush_broadcast_buckets()

        for i, tensor in enumerate(tensors):
            if isinstance(tensor, Future):
                tensor = tensor.wait()
            assert torch.sum(tensor).item() == i * torch.numel(tensor)

    # Test sum of all tensors less than bucket
    run_parallel(1, broadcast, [2, 2], 4, 1)
    run_parallel(4, broadcast, [100, 100], 4, 1)

    # Test each tensor larger than bucket
    run_parallel(4, broadcast, [100, 100], 4, 0.001)

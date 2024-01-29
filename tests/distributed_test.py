"""Unit tests for kfac/distributed.py."""

from __future__ import annotations

import pytest
import torch

from kfac.distributed import AllreduceTensorBucket
from kfac.distributed import fill_triu
from kfac.distributed import Future
from kfac.distributed import get_rank
from kfac.distributed import get_triu
from kfac.distributed import get_world_size
from kfac.distributed import NonSquareTensorError
from kfac.distributed import TorchDistributedCommunicator
from testing.distributed import distributed_test


def test_distributed_not_initialized() -> None:
    """Test rank/world_size functions when not using distributed."""
    assert get_rank() == 0
    assert get_world_size() == 1


def test_triu() -> None:
    """Test upper triangular support methods."""

    def check(shape: list[int]) -> None:
        """Check get triangle and restore."""
        t = torch.rand(shape)
        # Make symmetric
        t = t + t.T
        t_tri = get_triu(t)
        t_out = fill_triu(t.size(), t_tri)
        assert torch.equal(t, t_out)

    check([1, 1])
    check([2, 2])
    check([4, 4])

    with pytest.raises(ValueError):
        get_triu(torch.rand([4, 2]))
    with pytest.raises(ValueError):
        get_triu(torch.rand([2, 2, 2]))
    with pytest.raises(ValueError):
        x = torch.rand([4, 4])
        t = get_triu(x)
        fill_triu((2, 2, 2), t)


@pytest.mark.parametrize(
    'world_size,shape,symmetric,expect_raises',
    [
        (1, [2, 4], False, None),
        (4, [2, 4], False, None),
        (4, [4, 4], True, None),
        (4, [4, 2], True, NonSquareTensorError),
    ],
)
def test_allreduce(
    world_size: int,
    shape: list[int],
    symmetric: bool,
    expect_raises: type[BaseException] | None,
) -> None:
    """Test allreduce."""

    @distributed_test(world_size)
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
            assert isinstance(t_res, torch.Tensor)
            assert torch.sum(t_res).item() == torch.numel(t_res) * world_size
        except Exception as e:
            if expect_raises is not None and not isinstance(e, expect_raises):
                raise

    simple_allreduce(shape, symmetric, expect_raises)


@pytest.mark.parametrize(
    'world_size,shape,symmetric,expect_raises',
    [
        (1, [2, 4], False, None),
        (4, [2, 4], False, None),
        (4, [4, 4], True, None),
        (4, [4, 2], True, NonSquareTensorError),
    ],
)
def test_broadcast(
    world_size: int,
    shape: list[int],
    symmetric: bool,
    expect_raises: type[BaseException] | None,
) -> None:
    """Test broadcast."""

    @distributed_test(world_size)
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
            assert isinstance(t_res, torch.Tensor)
            # Rank 0 will broadcast and it should be all zeros
            assert torch.sum(t_res).item() == 0
        except Exception as e:
            if expect_raises is not None and not isinstance(e, expect_raises):
                raise

    simple_broadcast(shape, symmetric, expect_raises)


@pytest.mark.parametrize('world_size', [1, 4])
def test_allreduce_tensor_bucket(world_size: int) -> None:
    """Test AllreduceTensorBucket."""

    @distributed_test(world_size)
    def allreduce() -> None:
        """Run allreduce example in distributed environment."""
        bucket = AllreduceTensorBucket()

        # Communication operation can only be called once
        assert not bucket.communicated()
        res = bucket.allreduce()
        assert res is None
        assert bucket.communicated()
        with pytest.raises(RuntimeError):
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

        assert allreduce_future is not None
        allreduce_tensor = allreduce_future.wait()

        assert allreduce_tensor.nelement() == 12

        s = torch.distributed.get_world_size()

        f1.wait()
        assert f1.done()
        assert torch.equal(s * t1, f1.value())

        f2.wait()
        assert f2.done()
        assert torch.equal(s * t2, f2.value())

    allreduce()


@pytest.mark.parametrize(
    'world_size,shape,tensor_count,bucket_cap_mb,symmetric,expect_raises',
    [
        # Test sum of all tensors less than bucket
        (1, [2, 2], 4, 1, False, None),
        (4, [100, 100], 4, 1, False, None),
        # Test each tensor larger than bucket
        (4, [100, 100], 4, 0.001, False, None),
        # Test symmetric
        (1, [4, 4], 4, 1, True, None),
        (4, [100, 100], 4, 0.001, True, None),
        (4, [100, 4], 4, 1, True, NonSquareTensorError),
    ],
)
def test_allreduce_bucketed(
    world_size: int,
    shape: list[int],
    tensor_count: int,
    bucket_cap_mb: float,
    symmetric: bool,
    expect_raises: type[BaseException] | None,
) -> None:
    """Test bucketed allreduce."""

    @distributed_test(world_size)
    def allreduce(
        shape: list[int],
        tensor_count: int,
        bucket_cap_mb: float,
        symmetric: bool = False,
        expect_raises: type[BaseException] | None = None,
    ) -> None:
        """Test allreduce in distributed environment."""
        try:
            world_size = torch.distributed.get_world_size()
            comm = TorchDistributedCommunicator(bucket_cap_mb)

            tensors = []
            for _ in range(tensor_count):
                t = torch.ones(shape, dtype=torch.float32)
                tensors.append(
                    comm.allreduce_bucketed(t, symmetric=symmetric),
                )
            if world_size > 1:
                with pytest.raises(RuntimeError):
                    comm._new_allreduce_bucket(None)
            comm.flush_allreduce_buckets()

            for tensor in tensors:
                if isinstance(tensor, Future):
                    tensor = tensor.wait()
                assert isinstance(tensor, torch.Tensor)
                assert torch.sum(tensor).item() == world_size * torch.numel(
                    tensor,
                )
        except Exception as e:
            if expect_raises is not None and not isinstance(e, expect_raises):
                raise

    allreduce(shape, tensor_count, bucket_cap_mb, symmetric)


@pytest.mark.parametrize(
    'world_size,shape,tensor_count,bucket_cap_mb,symmetric',
    [
        # Test sum of all tensors less than bucket
        (1, [2, 2], 4, 1, False),
        (4, [100, 100], 4, 1, False),
        # Test each tensor larger than bucket
        (4, [100, 100], 4, 0.001, False),
        # Test symmetric
        (1, [4, 4], 4, 1, True),
        (4, [100, 100], 4, 0.001, True),
    ],
)
def test_allreduce_bucketed_grouped(
    world_size: int,
    shape: list[int],
    tensor_count: int,
    bucket_cap_mb: float,
    symmetric: bool,
) -> None:
    """Test bucketed allreduce with communication groups."""

    @distributed_test(world_size)
    def allreduce(
        shape: list[int],
        tensor_count: int,
        bucket_cap_mb: float,
        symmetric: bool = False,
    ) -> None:
        """Test allreduce in distributed environment."""
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
        # All buckets should be removed now so calling again shouldn't be
        # an issue
        comm.flush_allreduce_buckets()

        for tensor in tensors:
            if isinstance(tensor, Future):
                tensor = tensor.wait()
            assert isinstance(tensor, torch.Tensor)
            assert torch.sum(tensor).item() == group_size * torch.numel(
                tensor,
            )

    allreduce(shape, tensor_count, bucket_cap_mb, symmetric)

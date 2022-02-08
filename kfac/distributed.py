from __future__ import annotations

import warnings

import torch
import torch.distributed as dist

try:
    import apex_C

    flatten = apex_C.flatten
    unflatten = apex_C.unflatten
except ImportError:
    warnings.warn(
        "apex is not installed or was not installed wit --cpp_ext. "
        "Falling back to PyTorch flatten and unflatten.",
    )
    flatten = torch._utils._flatten_dense_tensors
    unflatten = torch._utils._unflatten_dense_tensors


Future = (torch._C.Future, torch.futures.Future)


class NonSquareTensorError(Exception):
    pass


class AllreduceTensorBucket:
    def __init__(self) -> None:
        self._tensors: list[torch.Tensor] = []
        self._futures: list[Future] = []
        self._size: int = 0
        self._communicated: bool = False

    @property
    def size(self) -> int:
        """Get current size of tensors in bucket."""
        return self._size

    def communicated(self) -> bool:
        """Check if communication for the bucket has been initiated."""
        return self._communicated

    def add_tensor(self, tensor: torch.Tensor) -> Future:
        """Add tensor to bucket.

        Args:
            tensor (torch.Tensor): tensor to add to bucket

        Returns:
            future that can be waited on with `future.wait()` to get the
            allreduced tensor.
        """
        future = torch.futures.Future()
        self._tensors.append(tensor)
        self._futures.append(future)
        assert len(self._tensors) == len(self._futures)
        self._size += tensor.element_size() * tensor.nelement()
        return future

    def allreduce(self) -> Future:
        """Initiate the allreduce for the bucket.

        Returns:
            future that will return the flattened and reduced tensor for
            of all tensors that were added to the bucket.

        Raises:
            RuntimeError:
                if allreduce is called twice on the same bucket.
        """
        if self.communicated():
            raise RuntimeError(
                "Communication for this bucket has already been performed. "
                "Ensure allreduce() or broadcast() is only called once for a "
                "given TensorBucket.",
            )
        self._communicated = True
        if len(self._tensors) == 0:
            return
        tensor = flatten(self._tensors)
        future = dist.all_reduce(tensor, async_op=True).get_future()
        # future.value is a list of one tensor so unpack it to be cleaner
        future = future.then(lambda f: f.value()[0])

        def _callback(future: Future) -> torch.Tensor:
            tensors = unflatten(future.value(), self._tensors)
            for sub_tensor, sub_future in zip(tensors, self._futures):
                sub_future.set_result(sub_tensor)
            # No longer need to hold strong references to items in these lists
            self._tensors.clear()
            self._futures.clear()

        future.add_done_callback(_callback)

        return future


class TorchDistributedCommunicator:
    """Interface to collective communication with PyTorch."""

    def __init__(
        self,
        bucket_cap_mb: float = 25.0,
    ) -> None:
        """Init TorchDistributedCommunicator.

        Args:
            bucket_cap_mb (float): maximum size for bucketed communication
                operations in megabytes (default: 25).
        """
        self._bucket_cap_mb = bucket_cap_mb
        self._allreduce_bucket: AllreduceTensorBucket | None = None

    @property
    def bucket_cap_bytes(self) -> int:
        """Get bucket cap in bytes."""
        return int(self._bucket_cap_mb * 1000 * 1000)

    def _get_allreduce_bucket(self) -> AllreduceTensorBucket | None:
        """Get current allreduce bucket.

        Returns:
            Current AllreduceTensorBucket if one has been created else None.
        """
        return self._allreduce_bucket

    def _new_allreduce_bucket(self) -> AllreduceTensorBucket:
        """Create a new allreduce bucket.

        Returns:
            TensorBucket

        Raises:
            RuntimeError:
                if the current bucket has not been communicated yet because
                this method will replace the current bucket with a new empty
                bucket.
        """
        bucket = self._get_allreduce_bucket()
        if bucket is not None and not bucket.communicated():
            raise RuntimeError(
                "Current bucket is being replaced without having been "
                "communicated.",
            )
        self._allreduce_bucket = AllreduceTensorBucket()
        return self._get_allreduce_bucket()

    def allreduce(
        self,
        tensor: torch.Tensor,
        group: dist.ProcessGroup = None,
        symmetric=False,
    ) -> torch._C.Future | torch.futures.Future | torch.Tensor:
        """Allreduce tensor asynchronously

        Args:
            tensor (torch.Tensor): tensor to allreduce
            group (torch.distributed.ProcessGroup): optional process group
                to perform communication within
            symmetric (bool): communicate symmetric tensor using upper triangle

        Returns:
            Future to tensor. Tensor can be retrieved with `future.wait()`.
            The returned tensor buffer may be different from the input buffer
            depending on the bucketing configuration.

            If group size is 1, no communication is performed and the tensor
            is returned.

        Raises:
            NonSquareTensorError:
                if symmetric is True and tensor is not a 2D square tensor.
        """
        if dist.get_world_size(group) == 1:
            return tensor
        shape = tensor.size()
        if symmetric:
            if len(shape) != 2 or shape[0] != shape[1]:
                raise NonSquareTensorError(
                    "Symmetric communication can only be done with a 2D "
                    f"square tensor. Got tensor with shape {shape}.",
                )
            tensor = get_triu(tensor)
        tensor = tensor.contiguous()
        future = dist.all_reduce(
            tensor,
            group=group,
            async_op=True,
        ).get_future()
        if symmetric:
            future = future.then(
                lambda fut: fill_triu(shape, fut.value()[0]),
            )
        else:
            future = future.then(lambda fut: fut.value()[0])
        return future

    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int,
        group: dist.ProcessGroup = None,
        symmetric=False,
    ) -> torch._C.Future | torch.futures.Future | torch.Tensor:
        """Broadcast tensor from src to all other workers asynchronously

        Args:
            tensor (torch.Tensor): tensor for broadcast
            src (int): rank of worker with src tensor
            group (torch.distributed.ProcessGroup): optional process group
                to perform communication within
            symmetric (bool): communicate symmetric tensor using upper triangle

        Returns:
            Future to tensor. Tensor can be retrieved with `future.wait()`.
            The returned tensor buffer may be different from the input buffer
            depending on the bucketing configuration.

            If group size is 1, no communication is performed and the tensor
            is returned.

        Raises:
            NonSquareTensorError:
                if symmetric is True and tensor is not a 2D square tensor.
        """
        if dist.get_world_size(group) == 1:
            return tensor
        shape = tensor.size()
        if symmetric:
            if len(shape) != 2 or shape[0] != shape[1]:
                raise NonSquareTensorError(
                    "Symmetric communication can only be done with a 2D "
                    f"square tensor. Got tensor with shape {shape}.",
                )
            tensor = get_triu(tensor)
        tensor = tensor.contiguous()
        future = dist.broadcast(
            tensor,
            src=src,
            group=group,
            async_op=True,
        ).get_future()
        if symmetric:
            future = future.then(lambda fut: fill_triu(shape, fut.value()[0]))
        else:
            future = future.then(lambda fut: fut.value()[0])
        return future

    def allreduce_bucketed(
        self,
        tensor: torch.Tensor,
        group: dist.ProcessGroup = None,
        symmetric=False,
    ) -> torch._C.Future | torch.futures.Future | torch.Tensor:
        """Allreduce tensor asynchronously with bucketing

        Warning:
            Allreduces are only performed once a bucket fills up. As a result,
            flush_allreduce_bucket() must be called to perform the allreduce
            on the last bucket which will only be partially filled.

        Note:
            The size of a bucket will not exceed bucket_cap_mb unless the
            bucket only contains a single tensor which is larger than
            bucket_cap_mp. In other words, tensors larger than the bucket cap
            will not be broken up.

        Args:
            tensor (torch.Tensor): tensor to allreduce
            group (torch.distributed.ProcessGroup): optional process group
                to perform communication within
            symmetric (bool): communicate symmetric tensor using upper triangle

        Returns:
            Future to tensor. Tensor can be retrieved with `future.wait()`.
            The returned tensor buffer may be different from the input buffer
            depending on the bucketing configuration.

            If group size is 1, no communication is performed and the tensor
            is returned.

        Raises:
            NonSquareTensorError:
                if symmetric is True and tensor is not a 2D square tensor.
            ValueError:
                if group is not None (groups are not supported yet).
        """
        if dist.get_world_size(group) == 1:
            return tensor
        shape = tensor.size()
        if symmetric:
            if len(shape) != 2 or shape[0] != shape[1]:
                raise NonSquareTensorError(
                    "Symmetric communication can only be done with a 2D "
                    f"square tensor. Got tensor with shape {shape}.",
                )
            tensor = get_triu(tensor)
        if group is not None:
            raise ValueError(
                "Bucketed allreduce does not support custom groups currently",
            )
        tensor_size = tensor.element_size() * tensor.nelement()
        bucket = self._get_allreduce_bucket()
        if bucket is None:
            bucket = self._new_allreduce_bucket()
        if bucket.size + tensor_size > self.bucket_cap_bytes:
            bucket.allreduce()
            bucket = self._new_allreduce_bucket()
        future = bucket.add_tensor(tensor)
        if symmetric:
            future = future.then(lambda fut: fill_triu(shape, fut.value()))
        return future

    def flush_allreduce_bucket(self) -> None:
        """Initiate the communication for the current allreduce bucket."""
        bucket = self._get_allreduce_bucket()
        if bucket is not None:
            bucket.allreduce()
            self._allreduce_bucket = None


def get_triu(tensor):
    """Returns flattened upper triangle of 2D tensor"""
    if len(tensor.shape) != 2:
        raise ValueError("triu(tensor) requires tensor to be 2 dimensional")
    if tensor.shape[0] > tensor.shape[1]:
        raise ValueError("tensor cannot have more rows than columns")
    idxs = torch.triu_indices(
        tensor.shape[0],
        tensor.shape[1],
        device=tensor.device,
    )
    return tensor[idxs[0], idxs[1]]


def fill_triu(shape, triu_tensor):
    """Reconstruct symmetric 2D tensor from flattened upper triangle

    Usage:
      >>> x = tensor.new_empty([10, 10])
      >>> triu_x = get_triu(x)
      >>> x_new = fill_triu([10, 10], triu_tensor)
      >>> assert torch.equal(x, x_new)  # true

    Args:
      shape (tuple): tuple(rows, cols) of size of output tensor
      triu_tensor (tensor): flattened upper triangle of the tensor returned by
          get_triu()

    Returns:
      Symmetric tensor with `shape` where the upper/lower triangles are filled
          with the data in `triu_tensor`
    """
    if len(shape) != 2:
        raise ValueError("shape must be 2 dimensional")
    rows, cols = shape
    dst_tensor = triu_tensor.new_empty(shape)
    idxs = torch.triu_indices(rows, cols, device=triu_tensor.device)
    dst_tensor[idxs[0], idxs[1]] = triu_tensor
    idxs = torch.triu_indices(rows, rows, 1, device=dst_tensor.device)
    dst_tensor.transpose(0, 1)[idxs[0], idxs[1]] = dst_tensor[idxs[0], idxs[1]]
    return dst_tensor

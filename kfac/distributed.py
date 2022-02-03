from typing import Union

import torch
import torch.distributed as dist

Future = (torch._C.Future, torch.futures.Future)


class NonSquareTensorError(Exception):
    pass


class TorchDistributedCommunicator:
    __slots__ = [
        "allreduce_bucketing",
        "bucket_cap_mb",
        "_allreduce_futures",
        "_allreduce_buffer",
    ]

    def __init__(
        self,
        allreduce_bucketing: bool = False,
        bucket_cap_mb: int = 25,
    ) -> None:
        self.allreduce_bucketing = allreduce_bucketing
        self.bucket_cap_mb = bucket_cap_mb
        self._allreduce_buffer = None
        self._allreduce_futures = []

    def allreduce(
        self,
        tensor: torch.Tensor,
        group: dist.ProcessGroup = None,
        symmetric=False,
    ) -> Union[torch._C.Future, torch.futures.Future, torch.Tensor]:
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
    ) -> Union[torch._C.Future, torch.futures.Future, torch.Tensor]:
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

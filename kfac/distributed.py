from typing import Union

import torch
import torch.distributed as dist

Future = (torch._C.Future, torch.futures.Future)


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
        upper_tri=False,
    ) -> Union[torch._C.Future, torch.futures.Future, torch.Tensor]:
        """Allreduce tensor asynchronously

        Args:
            tensor (torch.Tensor): tensor to allreduce
            group (torch.distributed.ProcessGroup): optional process group
                to perform communication within
            upper_tri (bool): communicate only upper triangle

        Returns:
            Future to tensor. Tensor can be retrieved with `future.wait()`.
            The returned tensor buffer may be different from the input buffer
            depending on the bucketing configuration.

            If group size is 1, no communication is performed and the tensor
            is returned.
        """
        if dist.get_world_size(group) == 1:
            return tensor
        shape = tensor.size()
        length = torch.numel(tensor)
        if upper_tri:
            tensor = get_triu(tensor)
        if not self.allreduce_bucketing:
            tensor = tensor.contiguous()
            future = dist.all_reduce(
                tensor,
                group=group,
                async_op=True,
            ).get_future()
            if upper_tri:
                future = future.then(
                    lambda fut: fill_triu(shape, fut.value()[0]),
                )
            else:
                future = future.then(lambda fut: fut.value()[0])
            return future
        else:
            if self._allreduce_buffer is None:
                index = 0
                self._allreduce_buffer = tensor.flatten()
            else:
                index = torch.numel(self._allreduce_buffer)
                self._allreduce_buffer = torch.cat(
                    [self._allreduce_buffer, tensor.flatten()],
                )

            def extract_and_shape(fut):
                t = fut.value()[0]
                t = fut[index : index + length].view(shape)
                if upper_tri:
                    t = fill_triu(shape, t)
                return t

            future = torch.future.Future().then(extract_and_shape)
            self._allreduce_futures.append(future)

            mbs = torch.numel(self._allreduce_buffer) * torch.element_size(
                self._allreduce_buffer,
            )
            if mbs > self.bucket_cap_mb:
                self._flush_allreduce_bucket()
                self._allreduce_futures = None
                self._allreduce_buffer = None

            return future

    def _flush_allreduce_bucket(self):
        mini_futures = self._allreduce_futures

        def set_all_futures(fut):
            t = fut.value()[0]
            for mini_fut in mini_futures:
                mini_fut.set_result(t)

        # TODO(gpauloski)
        # big_future = (
        #     dist.all_reduce(
        #         self._allreduce_buffer, group=group, async_op=True
        #      ).get_future().then(mini_futures)
        # )

    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int,
        group: dist.ProcessGroup = None,
        upper_tri=False,
    ) -> Union[torch._C.Future, torch.futures.Future, torch.Tensor]:
        """Broadcast tensor from src to all other workers asynchronously

        Args:
            tensor (torch.Tensor): tensor for broadcast
            src (int): rank of worker with src tensor
            group (torch.distributed.ProcessGroup): optional process group
                to perform communication within
            upper_tri (bool): communicate only upper triangle

        Returns:
            Future to tensor. Tensor can be retrieved with `future.wait()`.
            The returned tensor buffer may be different from the input buffer
            depending on the bucketing configuration.

            If group size is 1, no communication is performed and the tensor
            is returned.
        """
        if dist.get_world_size(group) == 1:
            return tensor
        shape = tensor.size()
        if upper_tri:
            tensor = tensor[torch.triu_indices(shape[0], shape[1])]
        tensor = tensor.contiguous()
        future = dist.broadcast(
            tensor,
            src=src,
            group=group,
            async_op=True,
        ).get_future()
        if upper_tri:
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

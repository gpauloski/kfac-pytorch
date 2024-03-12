"""Extensions of MPU functions."""

from __future__ import annotations

import torch
import torch.distributed as dist


def gather_from_model_parallel_region(
    tensor: torch.Tensor,
    dst: int,
    model_parallel_group: dist.ProcessGroup | None,
    fp32_allreduce: bool = False,
    dim: int = -1,
) -> torch.Tensor | None:
    """Gather model parallel partitions into single tensor.

    Note:
        This is a true `gather` where as mpu.gather_from_model_parallel_region
        is an `all gather`.

    Note:
        The concatenation is done along the last axis. I.e., this is the
        inverse operation of mpu.scatter_to_model_parallel_region().

    Args:
        tensor (torch.Tensor): tensor partition to gather.
        dst (rank): destination rank to gather full tensor on.
        model_parallel_group (ProcessGroup): model parallel process group.
            If None, model parallel region will be assumed to have size 1.
        fp32_allreduce (bool): if True and tensor is bf16, the tensor will
            be cast to float before communication. Note: this is to match
            the functionality of megatron's
            gather_from_model_parallel_region().
        dim (int): dimension along which to concatenate tensors.

    Returns:
        Gathered tensor on rank `dst` else None.
    """
    world_size = (
        1
        if model_parallel_group is None
        else dist.get_world_size(model_parallel_group)
    )
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return tensor

    # Bf16 convert
    dt = tensor.dtype
    if dt == torch.bfloat16 and fp32_allreduce:
        tensor = tensor.float()

    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]

    # TODO(gpauloski): PyTorch>=1.11 supports gather directly
    # which will be much faster
    torch.distributed.all_gather(
        tensor_list,
        tensor,
        group=model_parallel_group,
    )

    if dist.get_rank() == dst:
        # Note: torch.cat already creates a contiguous tensor.
        output = torch.cat(tensor_list, dim=dim).contiguous()

        # Bf16 convert
        if dt == torch.bfloat16 and fp32_allreduce:
            output = output.bfloat16()

        return output
    else:
        return None


def get_group_with_rank(rank: int, groups: list[list[int]]) -> list[int]:
    """Returns first group from list of groups containing rank.

    Args:
        rank (int): rank to search for.
        groups (list[list[int]]): list of groups where each group is a list
            of ranks.

    Returns:
        group (list of ranks) containing rank.

    Raises:
        ValueError:
            if a matching group is not found.
    """
    for group in groups:
        if rank in group:
            return group
    raise ValueError(f'Rank {rank} was not in any of the groups.')


def split_tensor_along_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    dim: int,
    contiguous_split_chunks: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Split a tensor along its last dimension.

    Source: https://github.com/EleutherAI/gpt-neox/blob/d7af1e7a8e3a816610b7d169456f81ca62d34ff7/megatron/mpu/utils.py

    Args:
        tensor (torch.Tensor): input tensor
        num_partitions (int): number of partitions to split the tensor
        dim (int): dimension along which to split the tensor.
        contiguous_split_chunks (bool): If True, make each chunk contiguous
            in memory.

    Returns:
        tuple of tensors
    """  # noqa: E501
    dim_size = tensor.size()[dim]

    if dim_size % num_partitions != 0:
        raise ValueError(
            f'Tensor dim {dim} (size={dim_size}) is not divisible '
            f'into {num_partitions} parts.',
        )

    dim_size = dim_size // num_partitions
    tensor_list = torch.split(tensor, dim_size, dim=dim)

    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tuple(tensor_list)

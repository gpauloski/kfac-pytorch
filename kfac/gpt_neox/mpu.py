"""Extensions of MPU functions."""
from __future__ import annotations

import torch
import torch.distributed as dist


def gather_from_model_parallel_region(
    tensor: torch.Tensor,
    dst: int,
    model_parallel_group: dist.ProcessGroup,
    fp32_allreduce: bool = False,
) -> torch.Tensor | None:
    """Gather model parallel partitions into single tensor.

    Note:
        This is a true `gather` where as mpu.gather_from_model_parallel_region
        is an `all gather`.

    Note:
        The concatentation is done along the last axis. I.e., this is the
        inverse operation of mpu.scatter_to_model_parallel_region().

    Args:
        tensor (torch.Tensor): tensor parition to gather.
        dst (rank): destination rank to gather full tensor on.
        model_parallel_group (ProcessGroup): model parallel process group.
        fp32_allreduce (bool): if True and tensor is bf16, the tensor will
            be cast to float before communication.

    Returns:
        Gathered tensor on rank `dst` else None.
    """
    world_size = dist.get_world_size(model_parallel_group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return tensor

    # Bf16 convert
    dt = tensor.dtype
    if dt == torch.bfloat16 and fp32_allreduce:
        tensor = tensor.float()

    # Size and dimension.
    last_dim = tensor.dim() - 1

    if dst == dist.get_rank():
        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    else:
        tensor_list = None

    torch.distributed.gather(
        tensor,
        tensor_list,
        dst=dst,
        group=model_parallel_group,
    )

    if tensor_list is not None:
        # Note: torch.cat already creates a contiguous tensor.
        output = torch.cat(tensor_list, dim=last_dim).contiguous()

        # Bf16 convert
        if dt == torch.bfloat16 and fp32_allreduce:
            output = output.bfloat16()

        return output
    else:
        return None

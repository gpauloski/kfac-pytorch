"""Helper wrappers for supported PyTorch modules."""
from __future__ import annotations

import torch
import torch.distributed as dist

from kfac.layers.modules import LinearModuleHelper


class GPTNeoXLinearModuleHelper(LinearModuleHelper):
    """ModuleHelper for GPTNeoX layers."""

    def __init__(
        self,
        module: torch.nn.Module,
        model_parallel_group: dist.ProcessGroup,
    ):
        """Init ModuleHelper.

        Args:
            module (torch.nn.Module): module in model to wrap.
            model_parallel_group (ProcessGroup): model parallel distributed
                process group this rank belongs to.
        """
        self.module = module
        self.model_parallel_group = model_parallel_group
        self.model_parallel_world_size = dist.get_world_size(
            self.model_parallel_group,
        )

    @property
    def a_factor_shape(self) -> tuple[int, int]:
        """Get shape of A factor.

        A shape = (in_features + int(has_bias), in_features + int(has_bias))
        """
        x = (
            self.module.weight.shape[1] * self.model_parallel_world_size
        ) + int(self.has_bias())
        return (x, x)

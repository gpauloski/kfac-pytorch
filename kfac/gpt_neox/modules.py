"""Helper wrappers for supported PyTorch modules."""
from __future__ import annotations

import sys

if sys.version_info >= (3, 9):  # pragma: >=3.9 cover
    from typing import Literal
else:  # pragma: <3.9 cover
    from typing_extensions import Literal

import torch
import torch.distributed as dist

from kfac.layers.modules import LinearModuleHelper


class GPTNeoXLinearModuleHelper(LinearModuleHelper):
    """ModuleHelper for GPTNeoX layers."""

    def __init__(
        self,
        module: torch.nn.Module,
        model_parallel_group: dist.ProcessGroup,
        parallelism: Literal['input', 'output'],
    ):
        """Init ModuleHelper.

        Args:
            module (torch.nn.Module): module in model to wrap.
            model_parallel_group (ProcessGroup): model parallel distributed
                process group this rank belongs to.
            parallelism (str): "input" if the layer is split on the input or
                "output" if split on the output.
        """
        self.module = module
        self.model_parallel_group = model_parallel_group
        self.model_parallel_world_size = dist.get_world_size(
            self.model_parallel_group,
        )
        self.parallelism = parallelism

    @property
    def a_factor_shape(self) -> tuple[int, int]:
        """Get shape of A factor."""
        if self.parallelism == 'input':
            x = (
                self.module.weight.shape[1] * self.model_parallel_world_size
            ) + int(self.has_bias())
        else:
            x = self.module.weight.shape[1] + int(self.has_bias())
        return (x, x)

    @property
    def g_factor_shape(self) -> tuple[int, int]:
        """Get shape of G factor."""
        if self.parallelism == 'output':
            x = self.module.weight.shape[0] * self.model_parallel_world_size
        else:
            x = self.module.weight.shape[0]
        return (x, x)

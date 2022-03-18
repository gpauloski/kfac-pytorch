from __future__ import annotations

from typing import Any
from typing import cast
from typing import Type

import torch

from kfac.layers.base import KFACBaseLayer
from kfac.layers.eigen import KFACEigenLayer
from kfac.layers.inverse import KFACInverseLayer
from kfac.layers.modules import Conv2dModuleHelper
from kfac.layers.modules import LinearModuleHelper
from kfac.layers.modules import ModuleHelper
from kfac.preconditioner import ComputeMethod

try:
    from megatron.mpu.layers import ColumnParallelLinear  # type: ignore
    from megatron.mpu.layers import RowParallelLinear

    megatron = True
except ImportError:
    megatron = False

__all__ = ['KNOWN_MODULES', 'get_kfac_layers', 'module_requires_grad']

KNOWN_MODULES = {'linear', 'conv2d'}
LINEAR_TYPES: tuple[type[torch.nn.Module], ...] = (torch.nn.Linear,)
CONV2D_TYPES: tuple[type[torch.nn.Module], ...] = (torch.nn.Conv2d,)

if megatron:
    KNOWN_MODULES |= {'columnparallellinear', 'rowparallellinear'}
    LINEAR_TYPES = LINEAR_TYPES + (
        cast(Type[torch.nn.Module], ColumnParallelLinear),
        cast(Type[torch.nn.Module], RowParallelLinear),
    )


def get_kfac_layers(
    module: torch.nn.Module,
    method: ComputeMethod,
    **kwargs: Any,
) -> list[tuple[torch.nn.Module, KFACBaseLayer]]:
    """Instantiates KFACLayer(s) for module

    Args:
      module: module to register
      method: type of KFAC layer to use
      **kwargs: parameters to pass to KFACLayer

    Returns:
      list of tuples where each tuple is (module, KFACLayer)
    """
    helper: type[ModuleHelper]
    if isinstance(module, LINEAR_TYPES):
        helper = LinearModuleHelper
    elif isinstance(module, CONV2D_TYPES):
        helper = Conv2dModuleHelper
    else:
        raise NotImplementedError(
            f'KFAC does not support layer {module.__class__.__name__}',
        )

    layer: KFACBaseLayer
    if method == ComputeMethod.EIGEN:
        layer = KFACEigenLayer(module, module_helper=helper(module), **kwargs)
    elif method == ComputeMethod.INVERSE:
        layer = KFACInverseLayer(
            module,
            module_helper=helper(module),
            **kwargs,
        )
    else:
        raise ValueError(f'Unknown KFAC method type: {method}')

    return [(module, layer)]


def module_requires_grad(module: torch.nn.Module) -> bool:
    """Returns False if any module param has .requires_grad=False"""
    return all([p.requires_grad for p in module.parameters()])

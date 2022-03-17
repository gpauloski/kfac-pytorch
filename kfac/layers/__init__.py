from enum import Enum

import torch.nn as nn

import kfac
from kfac.layers.eigen import KFACEigenLayer
from kfac.layers.inverse import KFACInverseLayer
from kfac.layers.modules import Conv2dModuleHelper
from kfac.layers.modules import LinearModuleHelper

try:
    from megatron.mpu.layers import ColumnParallelLinear
    from megatron.mpu.layers import RowParallelLinear

    megatron = True
except ImportError:
    megatron = False

__all__ = ["KNOWN_MODULES", "get_kfac_layers", "module_requires_grad"]

KNOWN_MODULES = {"linear", "conv2d"}
LINEAR_TYPES = (nn.Linear,)
CONV2D_TYPES = (nn.Conv2d,)

if megatron:
    KNOWN_MODULES |= {"columnparallellinear", "rowparallellinear"}
    LINEAR_TYPES = LINEAR_TYPES + (ColumnParallelLinear, RowParallelLinear)


def get_kfac_layers(module, method, **kwargs):
    """Instantiates KFACLayer(s) for module

    Args:
      module: module to register
      method: type of KFAC layer to use
      **kwargs: parameters to pass to KFACLayer

    Returns:
      list of tuples where each tuple is (module, KFACLayer)
    """
    if isinstance(module, LINEAR_TYPES):
        Helper = LinearModuleHelper
    elif isinstance(module, CONV2D_TYPES):
        Helper = Conv2dModuleHelper
    else:
        raise NotImplementedError(
            f"KFAC does not support layer {module.__class__.__name__}",
        )

    if method == kfac.ComputeMethod.EIGEN:
        layer = KFACEigenLayer(module, Helper(module), **kwargs)
    elif method == kfac.ComputeMethod.INVERSE:
        layer = KFACInverseLayer(module, Helper(module), **kwargs)
    else:
        raise ValueError(f"Unknown KFAC method type: {method}")

    return [(module, layer)]


def module_requires_grad(module):
    """Returns False if any module param has .requires_grad=False"""
    return all([p.requires_grad for p in module.parameters()])

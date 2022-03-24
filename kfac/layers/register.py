"""Utilities for registering PyTorch modules to KFAC layers."""
from __future__ import annotations

from typing import Any
from typing import cast
from typing import Type

import torch

from kfac.layers.base import KFACBaseLayer
from kfac.layers.modules import Conv2dModuleHelper
from kfac.layers.modules import LinearModuleHelper
from kfac.layers.modules import ModuleHelper

try:  # pragma: no cover
    from megatron.mpu.layers import ColumnParallelLinear  # type: ignore
    from megatron.mpu.layers import RowParallelLinear

    megatron = True
except ImportError:
    megatron = False

KNOWN_MODULES = {'linear', 'conv2d'}
LINEAR_TYPES: tuple[type[torch.nn.Module], ...] = (torch.nn.Linear,)
CONV2D_TYPES: tuple[type[torch.nn.Module], ...] = (torch.nn.Conv2d,)

if megatron:  # pragma: no cover
    KNOWN_MODULES |= {'columnparallellinear', 'rowparallellinear'}
    LINEAR_TYPES = LINEAR_TYPES + (
        cast(Type[torch.nn.Module], ColumnParallelLinear),
        cast(Type[torch.nn.Module], RowParallelLinear),
    )


def get_flattened_modules(
    root: torch.nn.Module,
) -> list[tuple[str, torch.nn.Module]]:
    """Returns flattened view of leaves of module tree."""
    return [
        (name, module)
        for name, module in root.named_modules()
        if len(list(module.children())) == 0
    ]


def requires_grad(module: torch.nn.Module) -> bool:
    """Return False if any module param has requires_grad=False."""
    return all([p.requires_grad for p in module.parameters()])


def get_module_helper(module: torch.nn.Module) -> ModuleHelper | None:
    """Return KFAC module helper that wraps a PyTorch module."""
    if isinstance(module, LINEAR_TYPES):
        return LinearModuleHelper(module)
    elif isinstance(module, CONV2D_TYPES):
        return Conv2dModuleHelper(module)
    else:
        return None


def register_modules(
    model: torch.nn.Module,
    kfac_layer_type: type[KFACBaseLayer],
    skip_layers: list[str],
    **layer_kwargs: Any,
) -> dict[torch.nn.Module, tuple[str, KFACBaseLayer]]:
    """Register supported modules in model with a KFACLayer.

    Args:
        model (torch.nn.Module): model to scan for modules to register.
        kfac_layer_type (type[KFACBaseLayer]): type of subclass of
            KFACBaseLayer to use.
        skip_layers (list[str]): names of layers to skip registering. Names
            can either by the name of the attribute or the name of the
            class of the layer. Matches are case insensitive.
        **layer_kwargs (dict[str, Any]): optional keyword arguments to
            pass to the kfac_layer_type constructor.
    """
    modules = get_flattened_modules(model)
    skip_layers = [s.lower() for s in skip_layers]

    kfac_layers: dict[torch.nn.Module, tuple[str, KFACBaseLayer]] = {}
    for name, module in modules:
        if (
            name.lower() not in skip_layers
            and module.__class__.__name__.lower() not in skip_layers
            and requires_grad(module)
        ):
            module_helper = get_module_helper(module)
            if module_helper is None:
                continue

            kfac_layer = kfac_layer_type(module_helper, **layer_kwargs)

            # get_flattened_modules() should never give us modules with the
            # same name
            assert module not in kfac_layers
            kfac_layers[module] = (name, kfac_layer)

    return kfac_layers

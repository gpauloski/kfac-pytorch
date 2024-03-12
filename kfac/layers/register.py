"""Utilities for registering PyTorch modules to KFAC layers."""

from __future__ import annotations

import re
from typing import Any

import torch

from kfac.layers.base import KFACBaseLayer
from kfac.layers.modules import Conv2dModuleHelper
from kfac.layers.modules import LinearModuleHelper
from kfac.layers.modules import ModuleHelper

KNOWN_MODULES = {'linear', 'conv2d'}
LINEAR_TYPES: tuple[type[torch.nn.Module], ...] = (torch.nn.Linear,)
CONV2D_TYPES: tuple[type[torch.nn.Module], ...] = (torch.nn.Conv2d,)


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
        return Conv2dModuleHelper(module)  # type: ignore
    else:
        return None


def any_match(query: str, patterns: list[str]) -> bool:
    """Check if a query string matches any pattern in a list.

    Note:
        `search()` is used rather than `match()` so True will be returned
        if there is a match anywhere in the query string.
    """
    regexes = [re.compile(p) for p in patterns]
    return any(regex.search(query) for regex in regexes)


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
        skip_layers (list[str]): regex patterns that if matched, will cause
            the layer to not be registered. The patterns will be applied
            against the layer's name and class name.
        **layer_kwargs (dict[str, Any]): optional keyword arguments to
            pass to the kfac_layer_type constructor.
    """
    modules = get_flattened_modules(model)

    kfac_layers: dict[torch.nn.Module, tuple[str, KFACBaseLayer]] = {}
    for name, module in modules:
        if (
            not any_match(name, skip_layers)
            and not any_match(module.__class__.__name__, skip_layers)
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

"""Utilities for registering PyTorch modules to KFAC layers."""
from __future__ import annotations

from typing import Callable
from typing import cast
from typing import Type

import torch

from kfac.distributed import TorchDistributedCommunicator
from kfac.enums import AllreduceMethod
from kfac.enums import ComputeMethod
from kfac.layers.base import KFACBaseLayer
from kfac.layers.eigen import KFACEigenLayer
from kfac.layers.inverse import KFACInverseLayer
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
    compute_method: ComputeMethod,
    *,
    allreduce_method: AllreduceMethod,
    compute_eigenvalue_outer_product: bool,
    grad_scaler: torch.cuda.amp.GradScaler | Callable[[], float] | None,
    factor_dtype: torch.dtype | None,
    inv_dtype: torch.dtype,
    skip_layers: list[str],
    symmetry_aware: bool,
    tdc: TorchDistributedCommunicator,
) -> dict[torch.nn.Module, tuple[str, KFACBaseLayer]]:
    """Register supported modules in model with a KFACLayer.

    Args:
        model (torch.nn.Module): model to scan for modules to register.
        compute_method (ComputeMethod): compute method to use for gradient
            preconditioner (inverse or eigen).
        allreduce_method (AllreduceMethod): allreduce method (default:
            AllreduceMethod.ALLREDUCE).
        compute_eigenvalue_outer_product (bool): precompute the outerproduct of
            eigen values on the worker that eigen decomposes G. This reduces
            the cost of the preconditioning stage but uses more memory
            (default: False).
        grad_scaler (optional): optional GradScaler or callable that
            returns the scale factor used in AMP training (default: None).
        factor_dtype (torch.dtype): data format to store factors in. If
            None, factors are stored in the format used in training
            (default: None).
        inv_dtype (torch.dtype): data format to store inverses in.
            Inverses (or eigen decompositions) may be unstable in half-
            precision (default: torch.float32).
        skip_layers (list[str]): names of layers to skip registering. Names
            can either by the name of the attribute or the name of the
            class of the layer. Matches are case insensitive.
        symmetry_aware (bool): use symmetry aware communication method.
            This is typically more helpful when the factors are very
            large (default: False).
        tdc (TorchDistributedCommunicator): communicator object. Typically
            the communicator object should be shared by all KFACBaseLayers.
    """
    # TODO(gpauloski): maybe this function should take **kwargs that get
    # passed to the KFAC layer.
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

            kfac_layer: KFACBaseLayer
            if compute_method == ComputeMethod.EIGEN:
                kfac_layer = KFACEigenLayer(
                    module_helper,
                    allreduce_method=allreduce_method,
                    factor_dtype=factor_dtype,
                    grad_scaler=grad_scaler,
                    inv_dtype=inv_dtype,
                    prediv_eigenvalues=compute_eigenvalue_outer_product,
                    symmetry_aware=symmetry_aware,
                    tdc=tdc,
                )
            elif compute_method == ComputeMethod.INVERSE:
                kfac_layer = KFACInverseLayer(
                    module_helper,
                    allreduce_method=allreduce_method,
                    factor_dtype=factor_dtype,
                    grad_scaler=grad_scaler,
                    inv_dtype=inv_dtype,
                    symmetry_aware=symmetry_aware,
                    tdc=tdc,
                )
            else:
                raise AssertionError(f'Unknown {compute_method=}')

            # get_flattened_modules() should never give us modules with the
            # same name
            assert module not in kfac_layers
            kfac_layers[module] = (name, kfac_layer)

    return kfac_layers

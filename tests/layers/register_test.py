from __future__ import annotations

import pytest
import torch

from kfac.distributed import TorchDistributedCommunicator
from kfac.enums import AllreduceMethod
from kfac.enums import ComputeMethod
from kfac.layers.modules import Conv2dModuleHelper
from kfac.layers.modules import LinearModuleHelper
from kfac.layers.modules import ModuleHelper
from kfac.layers.register import get_flattened_modules
from kfac.layers.register import get_module_helper
from kfac.layers.register import register_modules
from kfac.layers.register import requires_grad
from testing.models import LeNet
from testing.models import TinyModel


class NestedTinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.tiny1 = TinyModel()
        self.tiny2 = TinyModel()
        self.tiny3 = TinyModel()


@pytest.mark.parametrize(
    'module,expected',
    (
        (
            TinyModel(),
            [
                ('linear1', torch.nn.Linear),
                ('activation', torch.nn.ReLU),
                ('linear2', torch.nn.Linear),
                ('softmax', torch.nn.Softmax),
            ],
        ),
        (
            NestedTinyModel(),
            [
                ('tiny1.linear1', torch.nn.Linear),
                ('tiny1.activation', torch.nn.ReLU),
                ('tiny1.linear2', torch.nn.Linear),
                ('tiny1.softmax', torch.nn.Softmax),
                ('tiny2.linear1', torch.nn.Linear),
                ('tiny2.activation', torch.nn.ReLU),
                ('tiny2.linear2', torch.nn.Linear),
                ('tiny2.softmax', torch.nn.Softmax),
                ('tiny3.linear1', torch.nn.Linear),
                ('tiny3.activation', torch.nn.ReLU),
                ('tiny3.linear2', torch.nn.Linear),
                ('tiny3.softmax', torch.nn.Softmax),
            ],
        ),
        (
            LeNet(),
            [
                ('conv1', torch.nn.Conv2d),
                ('conv2', torch.nn.Conv2d),
                ('fc1', torch.nn.Linear),
                ('fc2', torch.nn.Linear),
                ('fc3', torch.nn.Linear),
            ],
        ),
    ),
)
def test_get_flattened_modules(
    module: torch.nn.Module,
    expected: list[tuple[str, type[torch.nn.Module]]],
) -> None:
    modules = get_flattened_modules(module)
    for (name, module), (exp_name, exp_type) in zip(modules, expected):
        assert name == exp_name
        assert isinstance(module, exp_type)


def test_requires_grad() -> None:
    linear = torch.nn.Linear(1, 1)
    assert requires_grad(linear)
    linear.bias.requires_grad = False
    assert not requires_grad(linear)


@pytest.mark.parametrize(
    'module,expected',
    (
        (torch.nn.Linear(1, 1), LinearModuleHelper),
        (torch.nn.Conv2d(1, 1, 1), Conv2dModuleHelper),
        (torch.nn.Conv3d(1, 1, 1), type(None)),
    ),
)
def test_get_module_helper(
    module: torch.nn.Module,
    expected: type[ModuleHelper | None],
) -> None:
    assert isinstance(get_module_helper(module), expected)


@pytest.mark.parametrize(
    'model,compute_method,skip_layers,expected_count',
    (
        (TinyModel(), ComputeMethod.EIGEN, [], 2),
        (TinyModel(), ComputeMethod.INVERSE, [], 2),
        (NestedTinyModel(), ComputeMethod.EIGEN, [], 6),
        (NestedTinyModel(), ComputeMethod.INVERSE, [], 6),
        (LeNet(), ComputeMethod.EIGEN, [], 5),
        (LeNet(), ComputeMethod.INVERSE, [], 5),
        (torch.nn.Conv3d(1, 1, 1), ComputeMethod.EIGEN, [], 0),
        # Test skip_layers: both by name or class and case invariant
        (LeNet(), ComputeMethod.EIGEN, ['fc1'], 4),
        (LeNet(), ComputeMethod.EIGEN, ['FC1'], 4),
        (LeNet(), ComputeMethod.EIGEN, ['Conv2d'], 3),
        (LeNet(), ComputeMethod.EIGEN, ['conv2d'], 3),
        (LeNet(), ComputeMethod.EIGEN, ['Conv2d', 'Linear'], 0),
    ),
)
def test_register_modules(
    model: torch.nn.Module,
    compute_method: ComputeMethod,
    skip_layers: list[str],
    expected_count: int,
) -> None:
    kfac_layers = register_modules(
        model,
        compute_method,
        allreduce_method=AllreduceMethod.ALLREDUCE,
        compute_eigenvalue_outer_product=False,
        grad_scaler=None,
        factor_dtype=None,
        inv_dtype=torch.float32,
        skip_layers=skip_layers,
        symmetry_aware=False,
        tdc=TorchDistributedCommunicator(),
    )
    assert len(kfac_layers) == expected_count

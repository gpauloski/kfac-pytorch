"""Unit tests for kfac/layers/register.py."""

from __future__ import annotations

import pytest
import torch

from kfac.distributed import TorchDistributedCommunicator
from kfac.enums import AllreduceMethod
from kfac.layers.base import KFACBaseLayer
from kfac.layers.eigen import KFACEigenLayer
from kfac.layers.inverse import KFACInverseLayer
from kfac.layers.modules import Conv2dModuleHelper
from kfac.layers.modules import LinearModuleHelper
from kfac.layers.modules import ModuleHelper
from kfac.layers.register import any_match
from kfac.layers.register import get_flattened_modules
from kfac.layers.register import get_module_helper
from kfac.layers.register import register_modules
from kfac.layers.register import requires_grad
from testing.models import LeNet
from testing.models import TinyModel


class NestedTinyModel(torch.nn.Module):
    """Nested model for testing recursive module discovery."""

    def __init__(self) -> None:
        """Init NestedTinyModel."""
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
    """Test get_flattened_modules."""
    modules = get_flattened_modules(module)
    for (name, module), (exp_name, exp_type) in zip(modules, expected):
        assert name == exp_name
        assert isinstance(module, exp_type)


def test_requires_grad() -> None:
    """Test requires_grad."""
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
    """Test get_module_helper."""
    assert isinstance(get_module_helper(module), expected)


@pytest.mark.parametrize(
    'model,layer_type,skip_layers,expected_count',
    (
        (TinyModel(), KFACEigenLayer, [], 2),
        (TinyModel(), KFACInverseLayer, [], 2),
        (NestedTinyModel(), KFACEigenLayer, [], 6),
        (NestedTinyModel(), KFACInverseLayer, [], 6),
        (LeNet(), KFACEigenLayer, [], 5),
        (LeNet(), KFACInverseLayer, [], 5),
        (torch.nn.Conv3d(1, 1, 1), KFACEigenLayer, [], 0),
        # Test skip_layers: both by name or class and case invariant
        (LeNet(), KFACEigenLayer, ['fc1'], 4),
        (LeNet(), KFACEigenLayer, ['Conv2d'], 3),
        (LeNet(), KFACEigenLayer, ['Conv2d', 'Linear'], 0),
    ),
)
def test_register_modules(
    model: torch.nn.Module,
    layer_type: type[KFACBaseLayer],
    skip_layers: list[str],
    expected_count: int,
) -> None:
    """Test register_modules."""
    kwargs = dict(
        allreduce_method=AllreduceMethod.ALLREDUCE,
        grad_scaler=None,
        factor_dtype=None,
        inv_dtype=torch.float32,
        symmetry_aware=False,
        tdc=TorchDistributedCommunicator(),
    )
    kfac_layers = register_modules(
        model,
        layer_type,
        skip_layers=skip_layers,
        **kwargs,
    )
    assert len(kfac_layers) == expected_count


@pytest.mark.parametrize(
    'query,patterns,match',
    (
        ('mystring', [], False),
        ('mystring', ['yourstring'], False),
        ('mystring', ['mystring'], True),
        ('mystring', ['string'], True),
        ('mystring', ['^string'], False),
        ('mystring', ['^string', '^my'], True),
        (
            '2.attention.query_key_value',
            ['attention', 'query_key_value'],
            True,
        ),
    ),
)
def test_any_match(query: str, patterns: list[str], match: bool) -> None:
    """Test any_match()."""
    assert any_match(query, patterns) == match

"""Unit Tests for kfac/base_preconditioner.py."""

from __future__ import annotations

from collections import defaultdict
from typing import Any
from typing import cast

import pytest
import torch

from kfac.base_preconditioner import BaseKFACPreconditioner
from kfac.distributed import TorchDistributedCommunicator
from kfac.enums import AllreduceMethod
from kfac.layers.base import KFACBaseLayer
from kfac.layers.inverse import KFACInverseLayer
from kfac.layers.register import register_modules
from testing.assignment import LazyAssignment
from testing.distributed import distributed_test
from testing.models import LeNet
from testing.models import TinyModel


def example_layers() -> dict[torch.nn.Module, tuple[str, KFACBaseLayer]]:
    """Return register layers of LeNet with KFAC."""
    return register_modules(
        LeNet(),
        kfac_layer_type=KFACInverseLayer,
        allreduce_method=AllreduceMethod.ALLREDUCE,
        grad_scaler=None,
        factor_dtype=None,
        inv_dtype=torch.float32,
        skip_layers=[],
        symmetry_aware=False,
        tdc=TorchDistributedCommunicator(),
    )


def test_base_preconditioner_init_raises() -> None:
    """Test BaseKFACPreconditioner raises."""
    with pytest.raises(ValueError):
        BaseKFACPreconditioner(
            example_layers(),
            assignment=LazyAssignment(),
            tdc=TorchDistributedCommunicator(),
            factor_update_steps=-1,
        )

    with pytest.raises(ValueError):
        BaseKFACPreconditioner(
            example_layers(),
            assignment=LazyAssignment(),
            tdc=TorchDistributedCommunicator(),
            inv_update_steps=-1,
        )

    with pytest.raises(ValueError):
        BaseKFACPreconditioner(
            example_layers(),
            assignment=LazyAssignment(),
            tdc=TorchDistributedCommunicator(),
            damping=-1,
        )

    with pytest.raises(ValueError):
        BaseKFACPreconditioner(
            example_layers(),
            assignment=LazyAssignment(),
            tdc=TorchDistributedCommunicator(),
            factor_decay=-1,
        )

    with pytest.raises(ValueError):
        BaseKFACPreconditioner(
            example_layers(),
            assignment=LazyAssignment(),
            tdc=TorchDistributedCommunicator(),
            factor_decay=2,
        )

    with pytest.raises(ValueError):
        BaseKFACPreconditioner(
            example_layers(),
            assignment=LazyAssignment(),
            tdc=TorchDistributedCommunicator(),
            kl_clip=-1,
        )

    with pytest.raises(ValueError):
        BaseKFACPreconditioner(
            example_layers(),
            assignment=LazyAssignment(),
            tdc=TorchDistributedCommunicator(),
            lr=-1,
        )

    with pytest.raises(ValueError):
        BaseKFACPreconditioner(
            example_layers(),
            assignment=LazyAssignment(),
            tdc=TorchDistributedCommunicator(),
            accumulation_steps=-1,
        )

    with pytest.warns():
        BaseKFACPreconditioner(
            example_layers(),
            assignment=LazyAssignment(),
            tdc=TorchDistributedCommunicator(),
            factor_update_steps=3,
            inv_update_steps=2,
        )


def test_base_preconditioner_init() -> None:
    """Test BaseKFACPreconditioner initialize."""
    factor_update_steps = 1
    inv_update_steps = 2
    damping = 0.003
    factor_decay = 0.95
    kl_clip = 0.001
    lr = 0.1
    accumulation_steps = 1

    preconditioner = BaseKFACPreconditioner(
        layers=example_layers(),
        assignment=LazyAssignment(),
        tdc=TorchDistributedCommunicator(),
        factor_update_steps=factor_update_steps,
        inv_update_steps=inv_update_steps,
        damping=damping,
        factor_decay=factor_decay,
        kl_clip=kl_clip,
        lr=lr,
        accumulation_steps=accumulation_steps,
    )
    assert preconditioner.damping == damping
    assert preconditioner.factor_decay == factor_decay
    assert preconditioner.kl_clip == kl_clip
    assert preconditioner.lr == lr
    assert preconditioner.factor_update_steps == factor_update_steps
    assert preconditioner.inv_update_steps == inv_update_steps
    assert preconditioner.steps == 0

    preconditioner = BaseKFACPreconditioner(
        layers=example_layers(),
        assignment=LazyAssignment(),
        tdc=TorchDistributedCommunicator(),
        damping=lambda x: damping,
        factor_decay=lambda x: factor_decay,
        kl_clip=lambda x: kl_clip,
        lr=lambda x: lr,
    )
    assert preconditioner.damping == damping
    assert preconditioner.factor_decay == factor_decay
    assert preconditioner.kl_clip == kl_clip
    assert preconditioner.lr == lr

    defaults = {'default1': None, 'default2': None}
    preconditioner2 = BaseKFACPreconditioner(
        layers=example_layers(),
        assignment=LazyAssignment(),
        tdc=TorchDistributedCommunicator(),
        defaults=defaults,
    )

    assert repr(preconditioner) != repr(preconditioner2)
    # repr() should list all parameters including those passed in with the
    # defaults parameter
    assert 'default1' in repr(preconditioner2)
    assert 'default2' in repr(preconditioner2)


def test_empty_state_dict() -> None:
    """Test state dict functionality with no factors."""
    p1 = BaseKFACPreconditioner(
        layers=example_layers(),
        assignment=LazyAssignment(),
        tdc=TorchDistributedCommunicator(),
        factor_update_steps=1,
        inv_update_steps=3,
        damping=5,
        factor_decay=0.7,
        kl_clip=11,
        lr=13,
        accumulation_steps=17,
        update_factors_in_hook=False,
        defaults={'default1': 19},
    )
    p1._steps = 99
    state_dict = p1.state_dict(include_factors=False)
    # include_factors=True should add entries for the factors even though
    # they are None at this point
    assert state_dict != p1.state_dict(include_factors=True)

    # We filled p1 with non-default values so we can load the
    # state_dict of p1 into p2 and see what is loaded
    p2 = BaseKFACPreconditioner(
        layers=example_layers(),
        assignment=LazyAssignment(),
        tdc=TorchDistributedCommunicator(),
    )
    p2.load_state_dict(state_dict, compute_inverses=False)

    assert p1.factor_update_steps == p2.factor_update_steps
    assert p1.inv_update_steps == p2.inv_update_steps
    assert p1.damping == p2.damping
    assert p1.factor_decay == p2.factor_decay
    assert p1.kl_clip == p2.kl_clip
    assert p1.lr == p2.lr

    # We only load the hyperparameters and training state
    assert p1._accumulation_steps != p2._accumulation_steps
    assert p1._update_factors_in_hook != p2._update_factors_in_hook
    assert p1._defaults != p2._defaults

    # Steps should be loaded
    assert p1._steps == p2._steps

    p3 = BaseKFACPreconditioner(
        layers=example_layers(),
        assignment=LazyAssignment(),
        tdc=TorchDistributedCommunicator(),
        factor_update_steps=lambda x: 1,
        inv_update_steps=lambda x: 3,
        damping=lambda x: 5,
        factor_decay=lambda x: 0.7,
        kl_clip=lambda x: 11,
        lr=lambda x: 13,
    )
    state_dict = p3.state_dict()
    assert 'factor_update_steps' not in state_dict
    assert 'inv_update_steps' not in state_dict
    assert 'damping' not in state_dict
    assert 'factor_decay' not in state_dict
    assert 'kl_clip' not in state_dict
    assert 'lr' not in state_dict

    p3.load_state_dict(state_dict, compute_inverses=False)
    assert p3.factor_update_steps == 1
    assert p3.inv_update_steps == 3
    assert p3.damping == 5
    assert p3.factor_decay == 0.7
    assert p3.kl_clip == 11
    assert p3.lr == 13

    # Check warns user if they set compute_inverses but the state dict was
    # created with include_factors=False
    with pytest.warns():
        p3.load_state_dict({'steps': 0}, compute_inverses=True)

    # Should cause no problems... but doesn't do much but set steps!
    p3.load_state_dict({'steps': 0}, compute_inverses=False)

    # Check mismatch in registered layers
    del state_dict['layers'][list(state_dict['layers'].keys()).pop()]
    with pytest.raises(ValueError):
        p3.load_state_dict(state_dict, compute_inverses=False)


@pytest.mark.parametrize(
    'accumulation_steps,broadcast,kfac_args',
    (
        (1, False, {'factor_update_steps': 2, 'inv_update_steps': 4}),
        (2, True, {'factor_update_steps': 4, 'inv_update_steps': 4}),
        (
            2,
            False,
            {
                'factor_update_steps': 4,
                'inv_update_steps': 4,
                'update_factors_in_hook': False,
            },
        ),
    ),
)
def test_base_preconditioner_e2e(
    accumulation_steps: int,
    broadcast: bool,
    kfac_args: dict[str, Any],
) -> None:
    """Run small e2e training example."""

    @distributed_test(world_size=1)
    def e2e() -> None:
        """Helper to run training in simulated distributed environment."""
        batch_size = 2
        model = TinyModel()
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        tdc = TorchDistributedCommunicator()
        layers = register_modules(
            model,
            KFACInverseLayer,
            allreduce_method=AllreduceMethod.ALLREDUCE,
            grad_scaler=None,
            factor_dtype=None,
            inv_dtype=torch.float32,
            skip_layers=[],
            symmetry_aware=False,
            tdc=tdc,
        )
        preconditioner = BaseKFACPreconditioner(
            layers=layers,
            assignment=LazyAssignment(broadcast=broadcast),
            tdc=tdc,
            accumulation_steps=accumulation_steps,
            **kfac_args,
        )

        for i in range(1, 10):
            x = torch.rand(batch_size, 10)
            y = torch.rand(batch_size, 10)
            y_pred = model(x)
            if i % accumulation_steps == 0:
                loss = criterion(y_pred, y)
                loss.backward()
                grad_weight_linear2 = model.linear2.weight.grad
                grad_bias_linear2 = model.linear2.bias.grad
                preconditioner.step()

                assert grad_weight_linear2 is not None
                assert model.linear2.weight.grad is not None
                assert grad_bias_linear2 is not None
                assert model.linear2.bias.grad is not None

                # Verify gradient was preconditioned
                assert not torch.equal(
                    grad_weight_linear2,
                    model.linear2.weight.grad,
                )
                assert not torch.equal(
                    grad_bias_linear2,
                    model.linear2.bias.grad,
                )
                optimizer.step()
                optimizer.zero_grad()

        # Test state dict computes inverses
        state_dict = preconditioner.state_dict()
        for _, layer in preconditioner._layers.values():
            layer = cast(KFACInverseLayer, layer)
            layer.a_factor = None
            layer.g_factor = None
            layer.a_inv = None
            layer.g_inv = None
        preconditioner.load_state_dict(state_dict)
        for _, layer in preconditioner._layers.values():
            layer = cast(KFACInverseLayer, layer)
            assert isinstance(layer.a_inv, torch.Tensor)
            assert isinstance(layer.g_inv, torch.Tensor)

        # Test grad hook supports tensor input rather than tuple
        preconditioner._save_grad_output(
            model.linear1,
            torch.rand(batch_size, 10),
            torch.rand(batch_size, 20),
        )

        # Test hook additional functionality
        if preconditioner._update_factors_in_hook:
            # Reset preconditioner to ensure hooks trigger
            preconditioner._steps = 0
            preconditioner._mini_steps = defaultdict(int)
            preconditioner._accumulation_steps = 100

            # Do forward/backward pass to verify hooks trigger and we
            # have temp factors for batch
            x = torch.rand(batch_size, 10)
            y = torch.rand(batch_size, 10)
            loss = criterion(model(x), y)
            loss.backward()
            mem_usage = preconditioner.memory_usage()
            for mem in mem_usage.values():
                assert mem > 0
            preconditioner.reset_batch()

            # Make sure hooks do not trigger when model is not in training mode
            model.eval()
            x = torch.rand(batch_size, 10)
            y = torch.rand(batch_size, 10)
            loss = criterion(model(x), y)
            loss.backward()
            mem_usage = preconditioner.memory_usage()
            for key, mem in mem_usage.items():
                if 'batch' in key:
                    assert mem == 0

    e2e()


def test_base_preconditioner_callable_hyperparams() -> None:
    """Test BaseKFACPreconditioner supports callable hyperparams."""
    p = BaseKFACPreconditioner(
        example_layers(),
        assignment=LazyAssignment(),
        tdc=TorchDistributedCommunicator(),
        factor_update_steps=lambda x: x * 2,
        inv_update_steps=lambda x: x * 3,
        damping=lambda x: x * 5,
        factor_decay=lambda x: x * 7,
        kl_clip=lambda x: x * 9,
    )

    for x in range(0, 10):
        p._steps = x
        assert p.factor_update_steps == x * 2
        assert p.inv_update_steps == x * 3
        assert p.damping == x * 5
        assert p.factor_decay == x * 7
        assert p.kl_clip == x * 9

    p = BaseKFACPreconditioner(
        example_layers(),
        assignment=LazyAssignment(),
        tdc=TorchDistributedCommunicator(),
        factor_update_steps=lambda x: 2,
        inv_update_steps=lambda x: 3,
        damping=lambda x: 5,
        factor_decay=lambda x: 7,
        kl_clip=lambda x: 9,
    )

    for x in range(0, 10):
        p._steps = x
        assert p.factor_update_steps == 2
        assert p.inv_update_steps == 3
        assert p.damping == 5
        assert p.factor_decay == 7
        assert p.kl_clip == 9


def test_grad_scale_no_layers() -> None:
    """Test computing grad scale with no layers has no divide by 0 error."""
    p = BaseKFACPreconditioner(
        layers=example_layers(),
        assignment=LazyAssignment(),
        tdc=TorchDistributedCommunicator(),
    )
    p._layers = {}
    assert p._compute_grad_scale() == 1.0

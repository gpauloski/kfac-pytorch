from __future__ import annotations

from typing import Any
from typing import Literal
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from kfac.distributed import TorchDistributedCommunicator
from kfac.enums import AllreduceMethod
from kfac.enums import DistributedStrategy
from kfac.layers.base import KFACBaseLayer
from kfac.layers.eigen import KFACEigenLayer
from kfac.layers.inverse import KFACInverseLayer
from kfac.layers.modules import LinearModuleHelper
from testing.distributed import distributed_test


@pytest.mark.parametrize(
    'kfac_layer,world_size,strategy,kwargs',
    [
        (KFACEigenLayer, 1, DistributedStrategy.MEM_OPT, {}),
        (KFACEigenLayer, 1, DistributedStrategy.COMM_OPT, {}),
        (KFACInverseLayer, 1, DistributedStrategy.MEM_OPT, {}),
        (KFACInverseLayer, 1, DistributedStrategy.COMM_OPT, {}),
        (KFACEigenLayer, 4, DistributedStrategy.MEM_OPT, {}),
        (KFACEigenLayer, 4, DistributedStrategy.COMM_OPT, {}),
        (
            KFACInverseLayer,
            4,
            DistributedStrategy.MEM_OPT,
            {'symmetry_aware': True},
        ),
        (
            KFACInverseLayer,
            4,
            DistributedStrategy.COMM_OPT,
            {'symmetry_aware': True},
        ),
        (
            KFACEigenLayer,
            4,
            DistributedStrategy.MEM_OPT,
            {
                'allreduce_method': AllreduceMethod.ALLREDUCE_BUCKETED,
                'grad_scaler': torch.cuda.amp.GradScaler(),
                'symmetry_aware': True,
                'prediv_eigenvalues': True,
            },
        ),
        (
            KFACEigenLayer,
            4,
            DistributedStrategy.COMM_OPT,
            {
                'allreduce_method': AllreduceMethod.ALLREDUCE_BUCKETED,
                'grad_scaler': torch.cuda.amp.GradScaler(),
                'symmetry_aware': True,
                'prediv_eigenvalues': True,
            },
        ),
    ],
)
def test_preconditioning_step(
    kfac_layer: type[KFACBaseLayer],
    world_size: int,
    strategy: Literal[
        DistributedStrategy.COMM_OPT | DistributedStrategy.MEM_OPT
    ],
    kwargs: dict[str, Any],
) -> None:
    """Test full preconditioning step for a layer.

    Warning:
        This test does not do an exhaustive check of allocation strategies.
    """

    @distributed_test(world_size)
    def precondition() -> None:
        in_features = 10
        out_features = 5
        batch_size = 2
        module = torch.nn.Linear(in_features, out_features)
        module_helper = LinearModuleHelper(module)

        layer = kfac_layer(
            module=module_helper,
            tdc=TorchDistributedCommunicator(),
            **kwargs,
        )

        # Compute gradient
        x = torch.rand([batch_size, in_features])
        y = torch.rand([batch_size, out_features])
        loss = (module(x) - y).sum()
        loss.backward()

        # Stage 1: save intermediate variables
        layer.save_layer_input([x])
        layer.save_layer_input([x])
        layer.save_layer_grad_output((y,))
        layer.save_layer_grad_output((y,))

        # Stage 2: compute factors
        layer.update_a_factor()
        layer.update_g_factor()

        # Stage 3: reduce factors
        layer.reduce_a_factor()
        layer.reduce_g_factor()
        if layer.allreduce_method == AllreduceMethod.ALLREDUCE_BUCKETED:
            layer.tdc.flush_allreduce_buckets()

        # Stage 4: compute second-order info
        if dist.get_rank() == 0:
            layer.compute_a_inv()
            layer.compute_g_inv()

        # Stage 5: communicate second-order info
        if world_size > 1 and strategy == DistributedStrategy.COMM_OPT:
            layer.broadcast_a_inv(src=0)
            layer.broadcast_g_inv(src=0)

        # Stage 6: compute and communicate preconditioned gradient
        if strategy == DistributedStrategy.COMM_OPT or (
            strategy == DistributedStrategy.MEM_OPT and dist.get_rank() == 0
        ):
            layer.preconditioned_grad()
        if strategy == DistributedStrategy.MEM_OPT:
            layer.broadcast_grad(src=0)

        # Stage 7: update gradient
        layer.update_grad()

    precondition()


@pytest.mark.parametrize('layer_type', (KFACInverseLayer, KFACEigenLayer))
def test_inv_layer(layer_type: type[KFACBaseLayer]) -> None:
    batch_size, in_features, out_features = 2, 5, 5
    module = torch.nn.Linear(in_features, out_features)
    x = torch.rand([batch_size, in_features])
    y = torch.rand([batch_size, out_features])
    loss = (module(x) - y).sum()
    loss.backward()

    module_helper = LinearModuleHelper(module)
    layer = layer_type(
        module=module_helper,
        tdc=TorchDistributedCommunicator(),
    )

    assert 'LinearModuleHelper' in repr(layer)
    assert layer_type.__name__ in repr(layer)

    # Cannot reduce factors, update gradient, or compute inverses
    with pytest.raises(RuntimeError):
        layer.reduce_a_factor()
    with pytest.raises(RuntimeError):
        layer.reduce_g_factor()
    with pytest.raises(RuntimeError):
        layer.update_grad()
    with pytest.raises(RuntimeError):
        layer.compute_a_inv()
    with pytest.raises(RuntimeError):
        layer.compute_g_inv()
    with pytest.raises(RuntimeError):
        layer.preconditioned_grad()

    # Broadcasts should fail because src rank has not computed the data
    with mock.patch('torch.distributed.get_rank', return_value=0):
        with pytest.raises(RuntimeError):
            layer.broadcast_grad(src=0)
        with pytest.raises(RuntimeError):
            layer.broadcast_a_inv(src=0)
        with pytest.raises(RuntimeError):
            layer.broadcast_g_inv(src=0)

    state_dict = layer.state_dict()
    assert 'A' in state_dict and state_dict['A'] is None
    assert 'G' in state_dict and state_dict['G'] is None
    with pytest.raises(KeyError):
        # state_dict must have A and G keys
        layer.load_state_dict({})
    layer.load_state_dict(state_dict)

    mem_usage = layer.memory_usage()
    for key in mem_usage:
        assert mem_usage[key] == 0

    layer.save_layer_input([x])
    layer.save_layer_grad_output((y,))

    # layer memory usage should reflect temp factors for current batch
    mem_usage = layer.memory_usage()
    assert mem_usage['a_batch'] > 0
    assert mem_usage['g_batch'] > 0

    # Check clear current batch
    layer.reset_batch()
    mem_usage = layer.memory_usage()
    assert mem_usage['a_batch'] == 0
    assert mem_usage['g_batch'] == 0
    # Should not raise an error no batch data has been accumulated
    layer.update_a_factor()
    layer.update_g_factor()

    # Repeat twice: once initializes the factors, the second will add to
    # the factors
    layer.save_layer_input([x])
    layer.save_layer_grad_output((y,))
    layer.update_a_factor()
    layer.update_g_factor()
    layer.save_layer_input([x])
    layer.save_layer_grad_output((y,))
    layer.update_a_factor()
    layer.update_g_factor()

    # flushed current batch so factors should have size but there is no
    # temp factors anymore
    mem_usage = layer.memory_usage()
    assert mem_usage['a_factors'] > 0
    assert mem_usage['g_factors'] > 0
    assert mem_usage['a_batch'] == 0
    assert mem_usage['g_batch'] == 0

    state_dict = layer.state_dict()
    assert isinstance(state_dict['A'], torch.Tensor)
    assert isinstance(state_dict['G'], torch.Tensor)
    layer.load_state_dict(state_dict)
    assert torch.equal(layer.a_factor, state_dict['A'])
    assert torch.equal(layer.g_factor, state_dict['G'])

    # Check gradient scaling. We haven't computed the preconditioned gradient
    # so just fake one
    grad = module_helper.get_grad()
    layer.grad = grad
    layer.update_grad(scale=10)
    assert torch.equal(10 * grad, module_helper.get_grad())


def test_nonsymmetric_eigen() -> None:
    batch_size, in_features, out_features = 2, 5, 5
    module = torch.nn.Linear(in_features, out_features)
    x = torch.rand([batch_size, in_features])
    y = torch.rand([batch_size, out_features])
    loss = (module(x) - y).sum()
    loss.backward()

    with mock.patch.object(
        LinearModuleHelper,
        'has_symmetric_factors',
        return_value=False,
    ):
        module_helper = LinearModuleHelper(module)
        layer = KFACEigenLayer(
            module=module_helper,
            tdc=TorchDistributedCommunicator(),
        )
    assert not layer.symmetric_factors

    layer.save_layer_input([x])
    layer.save_layer_grad_output((y,))
    layer.update_a_factor()
    layer.update_g_factor()
    layer.compute_a_inv()
    layer.compute_g_inv()
    layer.preconditioned_grad()
    layer.update_grad()

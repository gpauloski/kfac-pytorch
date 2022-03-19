from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.distributed as dist

from kfac.allocator import WorkerAllocator
from kfac.distributed import TorchDistributedCommunicator
from kfac.layers.base import KFACBaseLayer
from kfac.layers.eigen import KFACEigenLayer
from kfac.layers.inverse import KFACInverseLayer
from kfac.layers.modules import LinearModuleHelper
from testing.distributed import distributed_test


@pytest.mark.parametrize(
    'kfac_layer,world_size,grad_worker_frac,kwargs',
    [
        (KFACEigenLayer, 1, 1, {}),
        (KFACInverseLayer, 1, 1, {}),
        # MEM-OPT
        (KFACEigenLayer, 4, 0.25, {}),
        # HYBRID-OPT
        # TODO(gpauloski): fix this case
        # (KFACEigenLayer, 4, 0.5, {}),
        (KFACEigenLayer, 2, 0.5, {}),
        # COMM-OPT
        (KFACEigenLayer, 4, 1, {}),
    ],
)
def test_preconditioning_step(
    kfac_layer: type[KFACBaseLayer],
    world_size: int,
    grad_worker_frac: float,
    kwargs: dict[str, Any],
) -> None:
    @distributed_test(world_size)
    def precondition() -> None:
        in_features = 10
        out_features = 5
        batch_size = 2
        module = torch.nn.Linear(in_features, out_features)
        module_helper = LinearModuleHelper(module)

        layer = kfac_layer(
            module,
            module_helper=module_helper,
            tdc=TorchDistributedCommunicator(),
            **kwargs,
        )

        allocator = WorkerAllocator(
            grad_worker_frac,
            dist.get_rank(),
            dist.get_world_size(),
            dist.new_group,
        )
        layer.assign_workers(
            a_inv_worker=0,
            g_inv_worker=0,
            grad_src_worker=allocator.get_grad_src_rank(0),
            grad_worker_ranks=allocator.get_grad_worker_ranks(0),
            grad_worker_group=allocator.get_grad_worker_group(0),
            grad_receiver_ranks=allocator.get_grad_receiver_ranks(),
            grad_receiver_group=allocator.get_grad_receiver_group(),
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

        # Stage 4: compute second-order info
        layer.compute_a_inv()
        layer.compute_g_inv()

        # Stage 5: communicate second-order info
        layer.broadcast_a_inv()
        layer.broadcast_g_inv()

        # Stage 6: compute and communicate preconditioned gradient
        layer.preconditioned_grad()
        layer.broadcast_grad()

        # Stage 7: update gradient
        layer.update_grad()

    precondition()

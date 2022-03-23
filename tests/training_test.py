"""End-to-end training test for KFACPreconditoner."""
from __future__ import annotations

from multiprocessing import Process

import pytest
import torch

from kfac.preconditioner import KFACPreconditioner
from testing.distributed import distributed_test
from testing.models import TinyModel


def train() -> None:
    """Train TinyModel with KFAC on random data."""
    batch_size = 4
    in_features = 10
    out_features = 10
    epochs = 20

    x = torch.rand(batch_size, in_features)
    y = torch.rand(batch_size, out_features)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(x)
        torch.distributed.all_reduce(y)

    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    preconditioner = KFACPreconditioner(
        model,
        factor_update_steps=5,
        inv_update_steps=10,
        allreduce_bucket_cap_mb=0,
        update_factors_in_hook=False,
    )
    criterion = torch.nn.MSELoss(reduction='sum')

    losses = []
    for _ in range(epochs):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        losses.append(loss.item())
        loss.backward()
        preconditioner.step()
        optimizer.step()
        optimizer.zero_grad()

    assert losses[0] > losses[-1]


@pytest.mark.parametrize(
    'distributed,world_size',
    ((False, 1), (True, 1), (True, 2), (True, 4)),
)
def test_training(distributed: bool, world_size: int) -> None:
    """Test end-to-end training with KFACPreconditioner."""
    if not distributed:
        # Note: torch does not allow forking if autograd has been used
        # in the parent process. So we perform the training is a separate
        # process to keep this parent process "clean". See
        # https://github.com/pytorch/pytorch/issues/69839#issuecomment-993686048
        p = Process(target=train)
        p.start()
        p.join()
    else:
        _train = distributed_test(world_size=world_size)(train)
        _train()

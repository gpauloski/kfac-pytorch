"""Unit Tests for kfac/gpt_neox.py."""
from __future__ import annotations

import contextlib
import logging
from typing import Any
from unittest import mock

import pytest
import torch
from deepspeed.pipe import PipelineModule

from kfac.gpt_neox.preconditioner import GPTNeoXKFACPreconditioner
from testing.distributed import distributed_test
from testing.gpt_neox import RowParallelLinear
from testing.gpt_neox import sequential_model


@pytest.mark.parametrize(
    'num_stages,kwargs',
    (
        (1, {'assignment_strategy': 'memory'}),
        # (2, {'compute_method': 'eigen'}),
        (4, {'allreduce_bucket_cap_mb': 0}),
    ),
)
def test_gpt_neox_kfac_preconditioner(
    num_stages: int,
    kwargs: dict[str, Any],
) -> None:
    """Test GPTNeoXKFACPreconditioner."""

    @distributed_test(world_size=num_stages)
    def check() -> None:
        num_layers = 6
        model = sequential_model(layers=num_layers, hidden_dim=32)
        # This one should not be registered because it is not
        # a Column/RowParallelLinear
        model.append(torch.nn.Linear(32, 32))
        # This one should not be registered because it does not require grad
        module = RowParallelLinear(32, 32)
        module.requires_grad_(False)
        model.append(module)

        with (
            mock.patch.object(PipelineModule, 'to', mock.MagicMock()),
            # Trashing stdout/stderr because PipelineModule prints stuff
            contextlib.redirect_stdout(None),
            contextlib.redirect_stderr(None),
        ):
            logging.disable(10000)
            model = PipelineModule(layers=model, num_stages=num_stages)
            p = GPTNeoXKFACPreconditioner(model, **kwargs)

            # Check only 10 layers are registered (not the linear one)
            layers_per_rank = [
                0 for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather_object(
                layers_per_rank,
                len(p._layers),
            )
            assert sum(layers_per_rank) == num_layers

    check()


def test_input_validation() -> None:
    """Test GPTNeoXKFACPreconditioner input validation."""

    @distributed_test(world_size=1)
    def check() -> None:
        model = sequential_model(1, 1)

        with (
            mock.patch.object(PipelineModule, 'to', mock.MagicMock()),
            # Trashing stdout/stderr because PipelineModule prints stuff
            contextlib.redirect_stdout(None),
            contextlib.redirect_stderr(None),
        ):
            logging.disable(10000)
            model_ = PipelineModule(model, num_stages=1)
        with pytest.raises(ValueError, match='Inverse'):
            GPTNeoXKFACPreconditioner(model_, compute_method='inverse')

        with pytest.raises(ValueError, match='PipelineModule'):
            GPTNeoXKFACPreconditioner(model)

        with (
            mock.patch.object(PipelineModule, 'to', mock.MagicMock()),
            contextlib.redirect_stdout(None),
            contextlib.redirect_stderr(None),
        ):
            logging.disable(10000)
            model_ = PipelineModule(layers=model, num_stages=1)
        with pytest.raises(ValueError, match='allreduce_bucket_cap_mb'):
            GPTNeoXKFACPreconditioner(model_, allreduce_bucket_cap_mb=-1)

    check()


def test_model_parallel_unsupported() -> None:
    """Test DeepSpeed model parallelism unsupported."""

    @distributed_test(world_size=2)
    def check() -> None:
        model = sequential_model(layers=10, hidden_dim=32)
        with (
            mock.patch.object(PipelineModule, 'to', mock.MagicMock()),
            contextlib.redirect_stdout(None),
            contextlib.redirect_stderr(None),
        ):
            logging.disable(10000)
            model = PipelineModule(layers=model, num_stages=1)
            model.mpu().get_data_parallel_world_size = lambda: 1
            with pytest.raises(ValueError):
                GPTNeoXKFACPreconditioner(model)

    check()

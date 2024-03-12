"""Unit Tests for kfac/gpt_neox.py."""

from __future__ import annotations

import logging
import os
import pathlib
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from typing import Any
from unittest import mock

import deepspeed
import pytest
import torch

from kfac.gpt_neox.preconditioner import GPTNeoXKFACPreconditioner
from testing.distributed import distributed_test
from testing.gpt_neox import get_pipeline_module
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

        deepspeed.init_distributed()

        # This one should not be registered because it is not
        # a Column/RowParallelLinear
        model.append(torch.nn.Linear(32, 32))
        # This one should not be registered because it does not require grad
        module = RowParallelLinear(32, 32)
        module.requires_grad_(False)
        model.append(module)

        # Trashing stdout/stderr because get_pipeline_module prints stuff
        with redirect_stdout(None), redirect_stderr(None):
            logging.disable(10000)
            model = get_pipeline_module(layers=model, num_stages=num_stages)
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

        deepspeed.init_distributed()

        # Trashing stdout/stderr because get_pipeline_module prints stuff
        with redirect_stdout(None), redirect_stderr(None):
            logging.disable(10000)
            model_ = get_pipeline_module(model, num_stages=1)
        with pytest.raises(ValueError, match='Inverse'):
            GPTNeoXKFACPreconditioner(model_, compute_method='inverse')

        with pytest.raises(ValueError, match='PipelineModule'):
            GPTNeoXKFACPreconditioner(model)

        # Trashing stdout/stderr because get_pipeline_module prints stuff
        with redirect_stdout(None), redirect_stderr(None):
            logging.disable(10000)
            model_ = get_pipeline_module(layers=model, num_stages=1)
        with pytest.raises(ValueError, match='allreduce_bucket_cap_mb'):
            GPTNeoXKFACPreconditioner(model_, allreduce_bucket_cap_mb=-1)

    check()


def test_state_dict() -> None:
    """Test GPTNeoXKFACPreconditioner state dict."""
    world_size = 2

    @distributed_test(world_size=world_size)
    def check() -> None:
        num_layers = 6
        model = sequential_model(layers=num_layers, hidden_dim=32)

        deepspeed.init_distributed()

        # Trashing stdout/stderr because get_pipeline_module prints stuff
        with redirect_stdout(None), redirect_stderr(None):
            logging.disable(10000)
            model = get_pipeline_module(layers=model, num_stages=1)

        p = GPTNeoXKFACPreconditioner(model)

        state_dict = p.state_dict(include_factors=False)
        assert 'layers' not in state_dict
        p.load_state_dict(state_dict)

        p._assignment.inv_worker = mock.MagicMock(  # type: ignore
            return_value=1,
        )

        for name, layer in p._layers.values():
            if torch.distributed.get_rank() == p._assignment.inv_worker(
                name,
                'A',
            ):
                layer.a_factor = torch.rand([5, 5])
                layer.g_factor = torch.rand([5, 5])

        state_dict = p.state_dict(include_factors=True)
        assert 'layers' in state_dict
        assert len(state_dict['layers']) == num_layers
        p.load_state_dict(state_dict.copy(), compute_inverses=False)

        for _, layer in p._layers.values():
            layer.compute_a_inv = mock.MagicMock()  # type: ignore
            layer.compute_g_inv = mock.MagicMock()  # type: ignore
        p.load_state_dict(state_dict)
        for _, layer in p._layers.values():
            assert layer.compute_a_inv.called  # type: ignore
            assert layer.compute_g_inv.called  # type: ignore

    check()


def test_state_dict_save_factor_to_file_error() -> None:
    """Test param validation for saving factors to disk."""

    @distributed_test(world_size=1)
    def check() -> None:
        model = sequential_model(layers=1, hidden_dim=32)

        deepspeed.init_distributed()

        # Trashing stdout/stderr because get_pipeline_module prints stuff
        with redirect_stdout(None), redirect_stderr(None):
            logging.disable(10000)
            model = get_pipeline_module(layers=model, num_stages=1)

        p = GPTNeoXKFACPreconditioner(model)

        with pytest.raises(ValueError, match='factor_checkpoint_dir'):
            p.save_factors_to_dir()

        with pytest.raises(ValueError, match='factor_checkpoint_dir'):
            p.load_factors_from_dir()

    check()


def test_load_factors_from_dir_warning(tmp_path: pathlib.Path) -> None:
    """Test warning if checkpoint dir does not exist."""

    @distributed_test(world_size=1)
    def check() -> None:
        model = sequential_model(layers=1, hidden_dim=32)

        deepspeed.init_distributed()

        # Trashing stdout/stderr because get_pipeline_module prints stuff
        with redirect_stdout(None), redirect_stderr(None):
            logging.disable(10000)
            model = get_pipeline_module(layers=model, num_stages=1)

        path = str(tmp_path / 'checkpoint')
        p = GPTNeoXKFACPreconditioner(model, factor_checkpoint_dir=path)

        with pytest.warns(UserWarning, match='not a directory'):
            p.load_factors_from_dir()

    check()


def test_state_dict_save_factors_to_file(tmp_path: pathlib.Path) -> None:
    """Test GPTNeoXKFACPreconditioner state dict."""
    world_size = 2

    @distributed_test(world_size=world_size)
    def check() -> None:
        num_layers = 6
        model = sequential_model(layers=num_layers, hidden_dim=32)

        deepspeed.init_distributed()

        # Trashing stdout/stderr because get_pipeline_module prints stuff
        with redirect_stdout(None), redirect_stderr(None):
            logging.disable(10000)
            model = get_pipeline_module(layers=model, num_stages=1)

        p = GPTNeoXKFACPreconditioner(
            model,
            factor_checkpoint_dir=str(tmp_path),
        )

        # Force the second rank to be the inverse worker for everything
        # such that the first rank should not save anything
        p._assignment.inv_worker = mock.MagicMock(  # type: ignore
            return_value=1,
        )
        p._assignment.factor_worker = mock.MagicMock(  # type: ignore
            return_value=1,
        )

        for name, layer in p._layers.values():
            if torch.distributed.get_rank() == p._assignment.inv_worker(
                name,
                'A',
            ):
                layer.a_factor = torch.rand([5, 5])
                layer.g_factor = torch.rand([5, 5])

        state_dict = p.state_dict(include_factors=True)
        torch.distributed.barrier()
        p.load_state_dict(state_dict, compute_inverses=False)

        assert 'layers' not in state_dict
        assert os.path.isdir(tmp_path)
        files = [
            f
            for f in os.listdir(tmp_path)
            if os.path.isfile(os.path.join(tmp_path, f))
        ]
        assert len(files) == num_layers

        torch.distributed.barrier()

        if torch.distributed.get_rank() == 0:
            # Delete file to check we only load files that exist
            os.remove(os.path.join(tmp_path, files[-1]))

        torch.distributed.barrier()

        for _, layer in p._layers.values():
            layer.compute_a_inv = mock.MagicMock()  # type: ignore
            layer.compute_g_inv = mock.MagicMock()  # type: ignore
        p.load_state_dict(state_dict)
        a_inv_called = 0
        g_inv_called = 0
        for _, layer in p._layers.values():
            a_inv_called += int(layer.compute_a_inv.called)  # type: ignore
            g_inv_called += int(layer.compute_g_inv.called)  # type: ignore

        if torch.distributed.get_rank() == 1:
            # We remove one layer checkpoint so one less of each inverse should
            # be computed
            assert a_inv_called == num_layers - 1
            assert g_inv_called == num_layers - 1

    check()

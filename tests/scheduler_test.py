"""Unit tests for kfac/scheduler.py."""

from __future__ import annotations

from typing import Any
from typing import Callable

import pytest

from kfac.base_preconditioner import BaseKFACPreconditioner
from kfac.preconditioner import KFACPreconditioner
from kfac.scheduler import LambdaParamScheduler
from testing.models import TinyModel


def factor_func(scale: int, constant: bool = True) -> Callable[..., int]:
    """Get function which returns scale given step."""

    def factor(step: int = 1) -> int:
        """Scale function."""
        return scale if constant else scale * step

    return factor


@pytest.mark.parametrize(
    'preconditioner_type,preconditioner_kwargs',
    ((KFACPreconditioner, {'model': TinyModel()}),),
)
def test_input_check(
    preconditioner_type: type[BaseKFACPreconditioner],
    preconditioner_kwargs: dict[str, Any],
) -> None:
    """Test raises ValueError if preconditioner was already passed lambda."""
    preconditioner = preconditioner_type(
        **preconditioner_kwargs,
        factor_update_steps=factor_func(1),
    )
    with pytest.raises(ValueError):
        LambdaParamScheduler(
            preconditioner,
            factor_update_steps_lambda=factor_func(1),
        )

    preconditioner = KFACPreconditioner(
        TinyModel(),
        inv_update_steps=factor_func(1),
    )
    with pytest.raises(ValueError):
        LambdaParamScheduler(
            preconditioner,
            inv_update_steps_lambda=factor_func(1),
        )

    preconditioner = KFACPreconditioner(TinyModel(), damping=factor_func(1))
    with pytest.raises(ValueError):
        LambdaParamScheduler(preconditioner, damping_lambda=factor_func(1))

    preconditioner = KFACPreconditioner(
        TinyModel(),
        factor_decay=factor_func(1),
    )
    with pytest.raises(ValueError):
        LambdaParamScheduler(
            preconditioner,
            factor_decay_lambda=factor_func(1),
        )

    preconditioner = KFACPreconditioner(TinyModel(), kl_clip=factor_func(1))
    with pytest.raises(ValueError):
        LambdaParamScheduler(preconditioner, kl_clip_lambda=factor_func(1))

    preconditioner = KFACPreconditioner(TinyModel(), lr=factor_func(1))
    with pytest.raises(ValueError):
        LambdaParamScheduler(preconditioner, lr_lambda=factor_func(1))


@pytest.mark.parametrize(
    'preconditioner_type,preconditioner_kwargs',
    ((KFACPreconditioner, {'model': TinyModel()}),),
)
def test_scheduler(
    preconditioner_type: type[BaseKFACPreconditioner],
    preconditioner_kwargs: dict[str, Any],
) -> None:
    """Test param scheduler."""
    preconditioner = preconditioner_type(
        **preconditioner_kwargs,
        factor_update_steps=1,
        inv_update_steps=1,
        damping=1,
        factor_decay=1,
        kl_clip=1,
        lr=1,
    )
    scheduler = LambdaParamScheduler(
        preconditioner,
        factor_update_steps_lambda=factor_func(2),
        inv_update_steps_lambda=factor_func(3),
        damping_lambda=factor_func(5),
        factor_decay_lambda=factor_func(7),
        kl_clip_lambda=factor_func(9),
        lr_lambda=factor_func(11),
    )

    for steps in range(1, 10):
        preconditioner._steps = steps
        scheduler.step()
        assert preconditioner.factor_update_steps == 2**steps
        assert preconditioner.inv_update_steps == 3**steps
        assert preconditioner.damping == 5**steps
        assert preconditioner.factor_decay == 7**steps
        assert preconditioner.kl_clip == 9**steps
        assert preconditioner.lr == 11**steps

    preconditioner = preconditioner_type(
        **preconditioner_kwargs,
        factor_update_steps=1,
        inv_update_steps=1,
        damping=1,
        factor_decay=1,
        kl_clip=1,
        lr=1,
    )
    scheduler = LambdaParamScheduler(
        preconditioner,
        factor_update_steps_lambda=factor_func(2, False),
        inv_update_steps_lambda=factor_func(3, False),
        damping_lambda=factor_func(5, False),
        factor_decay_lambda=factor_func(7, False),
        kl_clip_lambda=factor_func(9, False),
        lr_lambda=factor_func(11, False),
    )
    for steps in range(1, 10):
        preconditioner._steps = steps
        scheduler.step(step=0)
        assert preconditioner.factor_update_steps == 0
        assert preconditioner.inv_update_steps == 0
        assert preconditioner.damping == 0
        assert preconditioner.factor_decay == 0
        assert preconditioner.kl_clip == 0
        assert preconditioner.lr == 0

    scheduler = LambdaParamScheduler(preconditioner)
    scheduler.step()

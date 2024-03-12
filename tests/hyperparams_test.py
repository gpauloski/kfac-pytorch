"""Unit tests for kfac/hyperparams.py."""

from __future__ import annotations

import pytest

from kfac.hyperparams import exp_decay_factor_averaging


def test_exp_decay_factor_averaging_types() -> None:
    """Test types and exceptions of exp_decay_factor_averaging()."""
    assert callable(exp_decay_factor_averaging())
    assert isinstance(exp_decay_factor_averaging()(1), float)
    with pytest.raises(ValueError):
        exp_decay_factor_averaging(0)
    with pytest.raises(ValueError):
        exp_decay_factor_averaging(-1)
    with pytest.raises(ValueError):
        exp_decay_factor_averaging()(-1)


def test_exp_decay_factor_averaging_non_decreasing() -> None:
    """Test exp_decay_factor_averaging() produces non decreasing values."""
    func = exp_decay_factor_averaging()
    values = [func(step) for step in range(1000)]
    assert all(a <= b for a, b in zip(values, values[1:]))


@pytest.mark.parametrize(
    'min_value,values',
    (
        (
            0.95,
            [(0, 0), (1, 0), (5, 0.8), (10, 0.9), (100, 0.95), (1000, 0.95)],
        ),
        (0.1, [(1, 0), (10, 0.1), (100, 0.1), (1000, 0.1)]),
        (1, [(1, 0), (10, 0.9), (100, 0.99)]),
    ),
)
def test_exp_decay_factor_averaging_values(
    min_value: float,
    values: list[tuple[int, float]],
) -> None:
    """Test exp_decay_factor_averaging() input/outputs."""
    func = exp_decay_factor_averaging(min_value)
    for step, expected_value in values:
        assert func(step) == expected_value

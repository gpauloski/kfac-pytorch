"""Common hyperparameter schedules."""

from __future__ import annotations

from typing import Callable


def exp_decay_factor_averaging(
    min_value: float = 0.95,
) -> Callable[[int], float]:
    """Exponentially decaying factor averaging schedule.

    Implements the running average estimate strategy for the Kronecker factors
    A and G from "Optimizing Neural Networks with Kronecker-factored
    Approximate Curvature" (Martens et al., 2015).

    The running average weight e at K-FAC step k is min(1 - 1/k, min_value)
    where the min_value is 0.95 by default.

    Args:
        min_value (float): minimum value for the running average weight.

    Returns:
        callable that takes an integer value for the current K-FAC step and
        returns a float value for the running average weight. This callable
        can be passed as the value of `factor_decay` to instances of
        `kfac.base_preconditioner.BaseKFACPreconditioner`. Note: that if the
        current step is 0, 1 / k is undefined so k = 1 will be used,
        and if the current step is negative, a ValueError will be raised.

    Raises:
        ValueError:
            if `min_value` is less than or equal to zero.
    """
    if min_value <= 0:
        raise ValueError('min_value must be greater than 0')

    def _factor_weight(step: int) -> float:
        if step < 0:
            raise ValueError(
                f'step value cannot be negative. Got step={step}.',
            )
        if step == 0:
            step = 1
        return min(1 - (1 / step), min_value)

    return _factor_weight

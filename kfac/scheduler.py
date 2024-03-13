"""KFAC preconditioner parameter scheduler."""

from __future__ import annotations

from typing import Callable

from kfac.base_preconditioner import BaseKFACPreconditioner


class LambdaParamScheduler:
    """Lambda param scheduler for KFAC preconditioner.

    Note:
        The lambda functions take as input the step value of the
        preconditioner. This step value is not necessarily the global number
        of optimization steps but rather the number of times
        preconditioner.step() has been called. This can be overridden by
        passing the step value to scheduler.step(step).

    Warning:
        KFACBasePreconditioner can take callables for the parameters instead
        of constant values. Passing callables for the parameters and using
        the LambdaParamScheduler at the same time is not possible.
    """

    def __init__(
        self,
        preconditioner: BaseKFACPreconditioner,
        *,
        factor_update_steps_lambda: Callable[[int], float] | None = None,
        inv_update_steps_lambda: Callable[[int], float] | None = None,
        damping_lambda: Callable[[int], float] | None = None,
        factor_decay_lambda: Callable[[int], float] | None = None,
        kl_clip_lambda: Callable[[int], float] | None = None,
        lr_lambda: Callable[[int], float] | None = None,
    ):
        """Init LambdaParamScheduler.

        Args:
            preconditioner (KFACBasePreconditioner): preconditioner to
                update parameters for.
            factor_update_steps_lambda (callable, optional): function which
                computes a multiplicative factor for the factor_update_steps
                given an integer value of the number of steps from the KFAC
                preconditioner. The result will be cast to an int
                (default: None).
            inv_update_steps_lambda (callable, optional): function which
                computes a multiplicative factor for the inv_update_steps
                given an integer value of the number of steps from the KFAC
                preconditioner. The result will be cast to an int
                (default: None).
            damping_lambda (callable, optional): function which
                computes a multiplicative factor for the damping
                given an integer value of the number of steps from the KFAC
                preconditioner (default: None).
            factor_decay_lambda (callable, optional): function which
                computes a multiplicative factor for the factor_decay
                given an integer value of the number of steps from the KFAC
                preconditioner (default: None).
            kl_clip_lambda (callable, optional): function which
                computes a multiplicative factor for the kl_clip
                given an integer value of the number of steps from the KFAC
                preconditioner (default: None).
            lr_lambda (callable, optional): function which
                computes a multiplicative factor for the lr
                given an integer value of the number of steps from the KFAC
                preconditioner (default: None).

        Raises:
            ValueError:
                if a lambda is passed for a parameter but the parameter in the
                preconditioner is already a callable.
        """
        self._preconditioner = preconditioner
        self._factor_update_steps_lambda = factor_update_steps_lambda
        self._inv_update_steps_lambda = inv_update_steps_lambda
        self._damping_lambda = damping_lambda
        self._factor_decay_lambda = factor_decay_lambda
        self._kl_clip_lambda = kl_clip_lambda
        self._lr_lambda = lr_lambda

        if self._factor_update_steps_lambda is not None:
            if callable(self._preconditioner._factor_update_steps):
                raise ValueError(
                    'preconditioner.factor_update_steps is already a callable '
                    'and cannot be updated by the lambdaparamscheduler.',
                )
        if self._inv_update_steps_lambda is not None:
            if callable(self._preconditioner._inv_update_steps):
                raise ValueError(
                    'preconditioner.inv_update_steps is already a callable '
                    'and cannot be updated by the lambdaparamscheduler.',
                )
        if self._damping_lambda is not None:
            if callable(self._preconditioner._damping):
                raise ValueError(
                    'preconditioner.damping is already a callable '
                    'and cannot be updated by the lambdaparamscheduler.',
                )
        if self._factor_decay_lambda is not None:
            if callable(self._preconditioner._factor_decay):
                raise ValueError(
                    'preconditioner.factor_decay is already a callable '
                    'and cannot be updated by the lambdaparamscheduler.',
                )
        if self._kl_clip_lambda is not None:
            if callable(self._preconditioner._kl_clip):
                raise ValueError(
                    'preconditioner.kl_clip is already a callable '
                    'and cannot be updated by the lambdaparamscheduler.',
                )
        if self._lr_lambda is not None:
            if callable(self._preconditioner._lr):
                raise ValueError(
                    'preconditioner.lr is already a callable '
                    'and cannot be updated by the lambdaparamscheduler.',
                )

    def step(self, step: int | None = None) -> None:
        """Update KFAC preconditioner params.

        Note:
            This should be called after preconditioner.step().

        Args:
            step (int, optional): optionally override the current step.
        """
        if self._factor_update_steps_lambda is not None:
            factor = self._factor_update_steps_lambda(
                step if step is not None else self._preconditioner.steps,
            )
            assert not callable(self._preconditioner._factor_update_steps)
            self._preconditioner._factor_update_steps = int(
                self._preconditioner._factor_update_steps * factor,
            )
        if self._inv_update_steps_lambda is not None:
            factor = self._inv_update_steps_lambda(
                step if step is not None else self._preconditioner.steps,
            )
            assert not callable(self._preconditioner._inv_update_steps)
            self._preconditioner._inv_update_steps = int(
                self._preconditioner._inv_update_steps * factor,
            )
        if self._damping_lambda is not None:
            factor = self._damping_lambda(
                step if step is not None else self._preconditioner.steps,
            )
            assert not callable(self._preconditioner._damping)
            self._preconditioner._damping *= factor
        if self._factor_decay_lambda is not None:
            factor = self._factor_decay_lambda(
                step if step is not None else self._preconditioner.steps,
            )
            assert not callable(self._preconditioner._factor_decay)
            self._preconditioner._factor_decay *= factor
        if self._kl_clip_lambda is not None:
            factor = self._kl_clip_lambda(
                step if step is not None else self._preconditioner.steps,
            )
            assert not callable(self._preconditioner._kl_clip)
            self._preconditioner._kl_clip *= factor
        if self._lr_lambda is not None:
            factor = self._lr_lambda(
                step if step is not None else self._preconditioner.steps,
            )
            assert not callable(self._preconditioner._lr)
            self._preconditioner._lr *= factor

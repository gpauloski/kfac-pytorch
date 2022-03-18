from __future__ import annotations

from typing import Any
from typing import Callable

from kfac.preconditioner import KFACPreconditioner


class KFACParamScheduler:
    """Updates KFAC parameters according to the epoch

    Similar to `torch.optim.lr_scheduler.StepLR()`

    Usage:
      Call KFACParamScheduler.step() each epoch to compute new parameter
      values.

    Args:
      kfac (KFAC): wrapped KFAC preconditioner
      damping_alpha (float, optional): multiplicative factor of the damping
          (default: 1)
      damping_schedule (list, optional): list of steps to update the damping
          by `damping_alpha` (default: None)
      update_freq_alpha (float, optional): multiplicative factor of the KFAC
          update freq (default: 1)
      update_freq_schedule (list, optional): list of steps to update the KFAC
          update freq by `update_freq_alpha` (default: None)
      start_step (int, optional): starting step, for use if resuming training
          from checkpoint (default: 0)
    """

    def __init__(
        self,
        preconditioner: KFACPreconditioner,
        damping_alpha: float = 1,
        damping_schedule: list[int] | None = None,
        update_freq_alpha: float = 1,
        update_freq_schedule: list[int] | None = None,
        start_step: int = 0,
    ) -> None:

        self.preconditioner = preconditioner
        params = self.preconditioner.param_groups[0]

        if damping_schedule is not None and callable(params['damping']):
            raise ValueError(
                'Cannot use a damping schedule when KFAC was passed a '
                'callable for damping ',
            )

        self.damping_base = params['damping']
        self.damping_alpha = damping_alpha
        self.damping_schedule = damping_schedule
        self.damping_factor_func = self._get_factor_func(
            self.damping_schedule,
            self.damping_alpha,
        )

        self.factor_update_freq_base = params['factor_update_steps']
        self.inv_update_freq_base = params['inv_update_steps']
        self.update_freq_alpha = update_freq_alpha
        self.update_freq_schedule = update_freq_schedule
        self.update_freq_factor_func = self._get_factor_func(
            self.update_freq_schedule,
            self.update_freq_alpha,
        )

        self._step = start_step

    def state_dict(self) -> dict[str, Any]:
        """Returns the state of the scheduler as a dict."""
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != 'kfac' and 'func' not in key
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Loads the schedulers state.

        Arguments:
          state_dict (dict): scheduler state. Should be an object returned
              from a call to state_dict().
        """
        self.__dict__.update(state_dict)

    def _get_factor_func(
        self,
        schedule: list[int] | None,
        alpha: float,
    ) -> Callable[[int], float]:
        """Returns a function to compute an update factor using the epoch"""
        if schedule is not None:
            schedule.sort(reverse=True)

        def factor_func(step: int) -> float:
            factor = 1.0
            if schedule is not None:
                for t in schedule:
                    if step >= t:
                        factor *= alpha
            return factor

        return factor_func

    def step(self, step: int | None = None) -> None:
        """Update KFAC parameters"""
        if step is not None:
            self._step = step
        else:
            self._step += 1

        params = self.preconditioner.param_groups[0]

        params['damping'] = self.damping_base * self.damping_factor_func(
            self._step,
        )

        factor = self.update_freq_factor_func(self._step)
        params['factor_update_steps'] = int(
            self.factor_update_freq_base * factor,
        )
        params['inv_update_steps'] = int(self.inv_update_freq_base * factor)

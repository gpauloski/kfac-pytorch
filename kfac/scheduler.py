class KFACParamScheduler():
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
    def __init__(self,
                 kfac,
                 damping_alpha=1,
                 damping_schedule=None,
                 update_freq_alpha=1,
                 update_freq_schedule=None,
                 start_step=0):

        self.kfac = kfac
        params = self.kfac.param_groups[0]

        self.damping_base = params['damping']
        self.damping_alpha = damping_alpha
        self.damping_schedule = damping_schedule
        self.damping_factor_func = \
                self._get_factor_func(self.damping_schedule,
                                     self.damping_alpha)

        self.factor_update_freq_base = params['factor_update_freq']
        self.inv_update_freq_base = params['inv_update_freq']
        self.update_freq_alpha = update_freq_alpha
        self.update_freq_schedule = update_freq_schedule
        self.update_freq_factor_func = \
                self._get_factor_func(self.update_freq_schedule,
                                     self.update_freq_alpha)

        self._step = start_step

    def state_dict(self):
        """Returns the state of the scheduler as a dict."""
        return {key: value for key, value in self.__dict__.items() 
                if key != 'kfac' and 'func' not in key}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
          state_dict (dict): scheduler state. Should be an object returned
              from a call to state_dict().
        """
        self.__dict__.update(state_dict)

    def _get_factor_func(self, schedule, alpha):
        """Returns a function to compute an update factor using the epoch"""
        if schedule is not None:
            schedule.sort(reverse=True)
        else:
            schedule = []

        def factor_func(step):
            factor = 1.
            for t in schedule:
                if step >= t:
                    factor *= alpha
            return factor

        return factor_func

    def step(self, step=None):
        """Update KFAC parameters"""
        if step is not None:
            self._step = step
        else:
            self._step += 1

        params = self.kfac.param_groups[0]

        params['damping'] = self.damping_base * self.damping_factor_func(self._step)

        factor = self.update_freq_factor_func(self._step)
        params['factor_update_freq'] = int(self.factor_update_freq_base * factor)
        params['inv_update_freq'] = int(self.inv_update_freq_base * factor)

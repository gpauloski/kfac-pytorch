import torch.optim as optim

def KFac(optim.Optimizer):
    """Distributed KFac Gradient Preconditioner.

    This is largely inspired by other KFac implementations:
        https://github.com/Thrandis/EKFAC-pytorch
        https://github.com/tensorflow/kfac
    """
    def __init__(self, model, damping=0.01, update_freq=10, alpha=1.0,
                 norm_constraint=None, distribute_inverse=True):
        """Initialize a KFac Gradient Preconditioner.

        Args:
          model (torch.nn.Module, required): Torch model to apply KFac updates.
          damping (float): Tikhonov regularization parameter to prevent ill-
              conditioned matrices when inverting. The curavature matrix (FIM),
              with have the identity matrix * damping added to it before
              inverting. (Default: 0.01)
          update_freq (int): Number of steps between computing the inverses
              of the FIM factors and applying the KFac update. Periodic KFac
              updates reduces computation without losing performance for
              reasonable intervals. (Default: 10)
          alpha (float): Running average for covariance statistics. If alpha=1,
              no running average is computed. (Default: 1.0)
          norm_constraint (float): Constrain the norm of the KFac update to be
              this value if not None. (Default: None)
          distribute_inverse (bool): If true, when computing the inverse of the
              Fisher factors per layer, the layers are computed on different
              workers in a round robin fashion and then communicated in with 
              Horovod. (Default: True)
        """
        if not 0 < update_freq:
            raise ValueError("Invalid update freqency: {}.".format(
                             update_freq) + " Expected value > 0")
        if not 0 < alpha <= 1.0:
            raise ValueError("Invalid alpha: {}.".format(alpha) + 
                             " Expected 0 < alpha <= 1.0")
        if not norm_constraint is None and not 0 < norm_constraint: 
            raise ValueError("Invalid norm constraint: {}.".format(
                             norm_constraint) + " Expected > 0 or None.")

        defaults = dict(damping=damping, update_freq=update_freq, alpha=alpha,
                        norm_constraint=norm_constraint, 
                        distribute_inverse=distribute_inverse)

        self.params = model.parameters()

        super(KFac, self).__init__(self.params, defaults)

    def step(self):
        """Apply one KFac update step.

        Computes the new covariance statistics and if iter % update_freq == 0,
        computes the FIM and applys the update the model gradients.
        """
        return

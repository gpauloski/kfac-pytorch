import torch
from torch.optim.optimizer import Optimizer

SUPPORTED_LAYERS = ["Linear", "Conv2d"]

class KFAC(Optimizer):
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

        # Save references to hooks in case we want to delete them
        self.layer_input_hooks = []
        self.layer_grad_output_hooks = []

        #params = self._register_modules(model)

        defaults = dict(damping=damping, update_freq=update_freq, alpha=alpha,
                        norm_constraint=norm_constraint, 
                        distribute_inverse=distribute_inverse)
 
        super(KFAC, self).__init__(model.parameters(), defaults)

    def step(self):
        """Apply one KFac update step.

        Computes the new covariance statistics and if iter % update_freq == 0,
        computes the FIM and applys the update the model gradients.
        """

        for group in self.param_groups:
            #print("GROUP", group)
            for p in group["params"]:
                #print("p", p)
                p.grad *= 0
        return

        with torch.no_grad():
            for group in self.param_groups:
                if len(group["params"]) == 2:
                    weight, bias = group["params"]
                else:
                    weight, bias = group["params"][0], None
                weight.grad.mul_(0.0)
                if bias is not None:
                    bias.grad.mul_(0.0)

                if "input" in self.state[group["module"]]:
                    del self.state[group["module"]]["input"]
                if "grad_output" in self.state[group["module"]]:
                    del self.state[group["module"]]["grad_output"]

    def ___________register_modules(self, model):
        """Register hooks for all sub-modules.

        Args:
          model (torch.nn.Module): Parent module to register all supported
              sub modules for.
        
        Returns:
          params dictionary of module to update in step()
        """
        params = []
        for module in model.modules():
            if module.__class__.__name__ in SUPPORTED_LAYERS:
                handle = module.register_forward_pre_hook(self._save_input)
                self.layer_input_hooks.append(handle)
                handle = module.register_backward_hook(self._save_grad_output)
                self.layer_grad_output_hooks.append(handle)

                p = [module.weight]
                if module.bias is not None:
                    p.append(module.bias)
                param = {"params": p, "module": module}
                params.append(param)
        return params

    def _save_input(self, module, layer_input):
        """Hook for saving module inputs."""
        if module.training:
            self.state[module]["input"] = layer_input

    def _save_grad_output(self, module, grad_input, grad_output):
        """Hook for saving the gradient for the layer output.

        TODO(gpauloski) Is grad_output actually the gradient we want?
        """ 
        if module.training:
            self.state[module]["grad_output"] = grad_output

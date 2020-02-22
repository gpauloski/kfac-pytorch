import torch
from torch.optim.optimizer import Optimizer
import horovod.torch as hvd

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

        self.damping = damping
        self.update_freq = update_freq
        self.alpha = alpha
        self.norm_constraint = norm_constraint
        self.distribute_inverse = distribute_inverse

        self.param_to_module = {}
        # Save references to hooks in case we want to delete them
        self.layer_input_hooks = []
        self.layer_grad_output_hooks = []

        self._register_modules(model)

        # TODO(gpauloski) register defaults
        super(KFAC, self).__init__(model.parameters(), {})

    def step(self):
        """Apply one KFac update step.

        Computes the new covariance statistics and if iter % update_freq == 0,
        computes the FIM and applys the update the model gradients.
        """
        # TODO(gpauloski) Do we need no_grad()?
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    
                    module = self.param_to_module[p]
                    
                    p.grad *= 0
                
                    if "input" in self.state[module]:
                        del self.state[module]["input"]
                    if "grad_output" in self.state[module]:
                        del self.state[module]["grad_output"]

    def _register_modules(self, model):
        """Register hooks for all sub-modules.

        Args:
          model (torch.nn.Module): Parent module to register all supported
              sub modules for.
        """
        for module in model.modules():
            # TODO(gpauloski) should we only register param->module if it is
            # a layer we are interested in?
            for param in module.parameters():
                self.param_to_module[param] = module

            if module.__class__.__name__ in SUPPORTED_LAYERS:
                handle = module.register_forward_pre_hook(self._save_input)
                self.layer_input_hooks.append(handle)
                handle = module.register_backward_hook(self._save_grad_output)
                self.layer_grad_output_hooks.append(handle)

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



import torch
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer


class KFAC2(Optimizer):

    def __init__(self, net, eps, sua=False, pi=False, update_freq=1,
                 alpha=1.0, constraint_norm=False):
        """ K-FAC Preconditionner for Linear and Conv2d layers.

        Computes the K-FAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.

        Args:
            net (torch.nn.Module): Network to precondition.
            eps (float): Tikhonov regularization parameter for the inverses.
            sua (bool): Applies SUA approximation.
            pi (bool): Computes pi correction for Tikhonov regularization.
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): Running average parameter (if == 1, no r. ave.).
            constraint_norm (bool): Scale the gradients by the squared
                fisher norm.
        """
        self.eps = eps
        self.sua = sua
        self.pi = pi
        self.update_freq = update_freq
        self.alpha = alpha
        self.constraint_norm = constraint_norm
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self._iteration_counter = 0
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear', 'Conv2d']:
                handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                handle = mod.register_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                d = {'params': params, 'mod': mod, 'layer_type': mod_class}
                self.params.append(d)
        super(KFAC2, self).__init__(self.params, {})

    def step(self, update_stats=True, update_params=True):
        """Performs one step of preconditioning."""
        fisher_norm = 0.
        for group in self.param_groups:
            # Getting parameters
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]
            # Update convariances and inverses
            if update_stats:
                if self._iteration_counter % self.update_freq == 0:
                    self._compute_covs(group, state)
                    ixxt, iggt = self._inv_covs(state['xxt'], state['ggt'],
                                                state['num_locations'])
                    state['ixxt'] = ixxt
                    state['iggt'] = iggt
                else:
                    if self.alpha != 1:
                        self._compute_covs(group, state)
            if update_params:
                # Preconditionning
                gw, gb = self._precond(weight, bias, group, state)
                # Updating gradients
                if self.constraint_norm:
                    fisher_norm += (weight.grad * gw).sum()
                weight.grad = gw.clone()
                if bias is not None:
                    if self.constraint_norm:
                        fisher_norm += (bias.grad * gb).sum()
                    bias.grad = gb.clone()
            # Cleaning
            if 'x' in self.state[group['mod']]:
                del self.state[group['mod']]['x']
            if 'gy' in self.state[group['mod']]:
                del self.state[group['mod']]['gy']
        # Eventually scale the norm of the gradients
        if update_params and self.constraint_norm:
            scale = (1. / fisher_norm) ** 0.5
            for group in self.param_groups:
                for param in group['params']:
                    param.grad.data *= scale
        if update_stats:
            self._iteration_counter += 1

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        if mod.training:
            self.state[mod]['x'] = i[0]

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        if mod.training:
            self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)

    def _precond(self, weight, bias, group, state):
        """Applies preconditioning."""
        if group['layer_type'] == 'Conv2d' and self.sua:
            return self._precond_sua(weight, bias, group, state)
        ixxt = state['ixxt']
        iggt = state['iggt']
        g = weight.grad.clone()
        s = g.shape
        if group['layer_type'] == 'Conv2d':
            g = g.contiguous().reshape(s[0], s[1]*s[2]*s[3])
        if bias is not None:
            gb = bias.grad.clone()
            g = torch.cat([g, gb.reshape(gb.shape[0], 1)], dim=1)
        g = torch.mm(torch.mm(iggt, g), ixxt)
        if group['layer_type'] == 'Conv2d':
            g /= state['num_locations']
        if bias is not None:
            gb = g[:, -1].contiguous().reshape(*bias.shape)
            g = g[:, :-1]
        else:
            gb = None
        g = g.contiguous().reshape(*s)
        return g, gb

    def _precond_sua(self, weight, bias, group, state):
        """Preconditioning for KFAC SUA."""
        ixxt = state['ixxt']
        iggt = state['iggt']
        g = weight.grad.data
        s = g.shape
        mod = group['mod']
        g = g.permute(1, 0, 2, 3).contiguous()
        if bias is not None:
            gb = bias.grad.view(1, -1, 1, 1).expand(1, -1, s[2], s[3])
            g = torch.cat([g, gb], dim=0)
        g = torch.mm(ixxt, g.contiguous().view(-1, s[0]*s[2]*s[3]))
        g = g.view(-1, s[0], s[2], s[3]).permute(1, 0, 2, 3).contiguous()
        g = torch.mm(iggt, g.view(s[0], -1)).view(s[0], -1, s[2], s[3])
        g /= state['num_locations']
        if bias is not None:
            gb = g[:, -1, s[2]//2, s[3]//2]
            g = g[:, :-1]
        else:
            gb = None
        return g, gb

    def _compute_covs(self, group, state):
        """Computes the covariances."""
        mod = group['mod']
        x = self.state[group['mod']]['x']
        gy = self.state[group['mod']]['gy']
        # Computation of xxt
        if group['layer_type'] == 'Conv2d':
            if not self.sua:
                x = F.unfold(x, mod.kernel_size, padding=mod.padding,
                             stride=mod.stride)
            else:
                x = x.view(x.shape[0], x.shape[1], -1)
            x = x.data.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
        else:
            x = x.data.t()
        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)
        if self._iteration_counter == 0:
            state['xxt'] = torch.mm(x, x.t()) / float(x.shape[1])
        else:
            state['xxt'].addmm_(mat1=x, mat2=x.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / float(x.shape[1]))
        # Computation of ggt
        if group['layer_type'] == 'Conv2d':
            gy = gy.data.permute(1, 0, 2, 3)
            state['num_locations'] = gy.shape[2] * gy.shape[3]
            gy = gy.contiguous().view(gy.shape[0], -1)
        else:
            gy = gy.data.t()
            state['num_locations'] = 1
        if self._iteration_counter == 0:
            state['ggt'] = torch.mm(gy, gy.t()) / float(gy.shape[1])
        else:
            state['ggt'].addmm_(mat1=gy, mat2=gy.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / float(gy.shape[1]))

    def _inv_covs(self, xxt, ggt, num_locations):
        """Inverses the covariances."""
        # Computes pi
        pi = 1.0
        if self.pi:
            tx = torch.trace(xxt) * ggt.shape[0]
            tg = torch.trace(ggt) * xxt.shape[0]
            pi = (tx / tg)
        # Regularizes and inverse
        eps = self.eps / num_locations
        diag_xxt = xxt.new(xxt.shape[0]).fill_((eps * pi) ** 0.5)
        diag_ggt = ggt.new(ggt.shape[0]).fill_((eps / pi) ** 0.5)
        ixxt = (xxt + torch.diag(diag_xxt)).inverse()
        iggt = (ggt + torch.diag(diag_ggt)).inverse()
        return ixxt, iggt
    
    def __del__(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()

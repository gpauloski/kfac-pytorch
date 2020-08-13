import math
import torch

from . import utils


class KFACLayer(object):
    def __init__(self,
                 module,
                 use_eigen_decomp=True,
                 batch_first=True,
                 accumulate_data=True,
                 A_rank=None,
                 G_rank=None):
        self.module = module
        self.use_eigen_decomp=use_eigen_decomp
        self.batch_first = batch_first
        self.accumulate_data = accumulate_data
        self.eps = 1e-10
        self.A_rank = A_rank
        self.G_rank = G_rank

        # Should be overridden by implementing class
        self.has_bias = False
        self.factors_are_symmetric = True

        self.a_inputs  = []  # inputs accumulated from module hooks
        self.g_outputs = []  # outputs accumulated from module hooks
        self.A_factor  = None
        self.G_factor  = None
        self.A_inv     = None
        self.G_inv     = None
        self.preconditioned_gradient = None

    def __repr__(self):
        return 'KFAC {}({})'.format(self.__class__.__name__, repr(self.module))

    def state_dict(self, include_inverses=False):
        """Returns the state of the KFACLayer as a dictionary.

        Used by kfac.KFAC for state saving/loading. Note by default only the
        factors are saved because the inverses can be recomputed from the
        factors, however, the `include_inverses` flag can override this.
        """
        state = {'A_factor': self.A_factor, 'G_factor': self.G_factor}
        if include_inverses:
            state['A_inv'] = self.A_inv
            state['G_inv'] = self.G_inv
        return state

    def load_state_dict(self, state_dict):
        """Loads the KFACLayer state."""
        device = next(self.module.parameters()).device
        try:
            self._init_A_buffers(state_dict['A_factor'].to(device))
            self._init_G_buffers(state_dict['G_factor'].to(device))
            if 'A_inv' in state_dict:
                self.A_inv = state_dict['A_inv']
            if 'G_inv' in state_dict:
                self.G_inv = state_dict['G_inv'] 
        except KeyError: 
            raise KeyError('KFACLayer state_dict must contain keys: '
                           '["A_factor", "G_factor"].')

    def compute_A_inv(self, rank, damping=0.001):
        """Compute A inverse on specified ranks

        Note: all ranks will enter this function but only the ranks assigned
        to this layer will continue to actually compute the inverses.
        All other ranks will simply zero out their inverse buffers for. This 
        is done so we can sum the inverses across all ranks to communicate the
        results of locally computed inverses.

        Args:
          rank (int): rank of worker entering function
          damping (float): damping value to condition inverse
        """
        if self.A_rank is None:
            raise ValueError('Workers have not been assigned to layer yet.')
        if rank == self.A_rank:
            self._compute_factor_inverse(self.A_factor, self.A_inv, damping)

    def compute_G_inv(self, rank, damping=0.001):
        """Compute G inverse on specified ranks

        See `compute_A_inv` for more info`
        """
        if self.G_rank is None:
            raise ValueError('Workers have not been assigned to layer yet.')
        if rank == self.G_rank:
            self._compute_factor_inverse(self.G_factor, self.G_inv, damping)

    def get_gradient(self):
        """Get formated gradients (weight and bias) of module

        Returns:
          gradient of shape [ouput_dim, input_dim]. If bias != None, concats bias
        """
        g = self._get_weight_grad()
        if self.has_bias:
            g = torch.cat([g, self._get_bias_grad().data.view(-1, 1)], 1)
        return g

    def get_factors(self):
        """Returns list of all factors in layer"""
        return [self.A_factor, self.G_factor]

    def get_inverses(self, return_ranks=False):
        """Returns list of all inv factors in layer

        Args:
          return_ranks (bool): If True, return list indicating rank that 
              inverse tensor is on
        """
        if return_ranks:
            return [self.A_inv, self.G_inv], [self.A_rank, self.G_rank] 
        return [self.A_inv, self.G_inv]

    def compute_preconditioned_gradient(self, damping=0.001):
        """Compute precondition gradient of each weight in module

        Produces a list of preconditioned weight gradient and bias gradient
        (if there is a bias) where each has shape [output_dim, input_dim].
        Preconditioned gradients can be applied to the actual gradients with 
        `update_gradient()`.
        """
        # Compute preconditioned gradient using specified inverse method
        if self.use_eigen_decomp:
            grad = self._get_precondition_gradient_eigen(damping)
        else:
            grad = self._get_precondition_gradient_inv()
        # Reshape appropriately
        if self.has_bias:
            grad = [grad[:, :-1], grad[:, -1:]]
            grad[0] = grad[0].view(self._get_weight_grad().data.size())
            grad[1] = grad[1].view(self._get_bias_grad().data.size())
        else:
            grad = [grad.view(self._get_weight_grad().data.size())]
        self.preconditioned_gradient = grad

    def save_inputs(self, input):
        """Save inputs locally"""
        if self.accumulate_data:
            self.a_inputs.append(input[0].data)
        else:
            self.a_inputs = [input[0].data]

    def save_grad_outputs(self, grad_output):
        """Save grad w.r.t outputs locally"""
        if self.accumulate_data:
            self.g_outputs.append(grad_output[0].data)
        else:
            self.g_outputs = [grad_output[0].data]

    def update_A_factor(self, alpha=0.95):
        """Compute factor A and add to running averages"""
        A_new = self._get_A_factor(self.a_inputs).to(torch.float32)
        del self.a_inputs[:]  # clear accumulated inputs
        if self.A_factor is None:
            self._init_A_buffers(A_new)
        utils.update_running_avg(A_new, self.A_factor, alpha=alpha)

    def update_G_factor(self, alpha=0.95):
        """Compute factor G and add to running averages"""
        G_new = self._get_G_factor(self.g_outputs).to(torch.float32)
        del self.g_outputs[:]  # clear accumulated outputs
        if self.G_factor is None:
            self._init_G_buffers(G_new)
        utils.update_running_avg(G_new, self.G_factor, alpha=alpha)

    def update_gradient(self, scale=None):
        """Updates gradients of module with computed precondition gradients"""
        if self.preconditioned_gradient is None:
            raise RuntimeError('self.compute_preconditioned_gradient() should'
                    ' be called before update_gradient()')
        if scale is not None:
            v = [scale * x for x in self.preconditioned_gradient]
        else:
            v = self.preconditioned_gradient
        self._get_weight_grad().data.copy_(v[0])
        if self.has_bias:
            self._get_bias_grad().data.copy_(v[1])

    def _compute_factor_inverse(self, factor, inverse, damping=0.001):
        """Computes inverse/eigendecomp of factor and saves result to inverse"""
        if self.use_eigen_decomp:
            inverse.data.copy_(
                utils.get_eigendecomp(factor, symmetric=self.factors_are_symmetric)
            )
        else:
            inverse.data.copy_(
                utils.get_inverse(factor, damping=damping, 
                        symmetric=self.factors_are_symmetric)
            )
 
    def _get_A_factor(self, a_inputs):
        """Compute A factor. Returns A."""
        raise NotImplementedError

    def _get_G_factor(self, g_inputs):
        """Compute G factor. Returns G."""
        raise NotImplementedError

    def _get_bias_grad(self):
        """Get bias.grad tensor of module"""
        return self.module.bias.grad

    def _get_weight_grad(self):
        """Get weight.grad tensor of module"""
        return self.module.weight.grad
    
    def _get_precondition_gradient_eigen(self, damping=0.001):
        """Compute preconditioned gradient for eigendecomp method"""
        grad = self.get_gradient()
        QA, QG = self.A_inv[:,:-1], self.G_inv[:,:-1]
        dA, dG = self.A_inv[:,-1], self.G_inv[:,-1]
        v1 = QG.t() @ grad @ QA
        v2 = v1 / (dG.unsqueeze(1) * dA.unsqueeze(0) + damping)
        return QG @ v2 @ QA.t()

    def _get_precondition_gradient_inv(self):
        """Compute preconditioned gradient for inverse method"""
        grad = self.get_gradient() 
        return self.G_inv @ grad @ self.A_inv

    def _init_A_buffers(self, factor):
        """Create buffers for factor A and its inverse"""
        assert self.A_factor is None, ('A buffers have already been '
                'initialized. Was _init_A_buffers() called more than once?')
        self.A_factor = torch.diag(factor.new(factor.shape[0]).fill_(1))
        if self.use_eigen_decomp:
            # add one axtra column for eigenvalues
            shape = (factor.shape[0], factor.shape[0] + 1) 
        else:
            shape = factor.shape
        self.A_inv = factor.new_zeros(shape) 

    def _init_G_buffers(self, factor):
        """Create buffers for factor G and its inverse"""
        assert self.G_factor is None, ('G buffers have already been '
                'initialized. Was _init_G_buffers() called more than once?')
        self.G_factor = torch.diag(factor.new(factor.shape[0]).fill_(1))
        if self.use_eigen_decomp:
            # add one axtra column for eigenvalues
            shape = (factor.shape[0], factor.shape[0] + 1)
        else:
            shape = factor.shape
        self.G_inv = factor.new_zeros(shape) 


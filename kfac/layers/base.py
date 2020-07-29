import math
import torch


class KFACLayer(object):
    def __init__(self,
                 module,
                 use_eigen_decomp=True,
                 damping = 0.001,
                 factor_decay=0.95,
                 batch_averaged=True,
                 A_rank=None,
                 G_rank=None):
        self.module = module
        self.use_eigen_decomp=use_eigen_decomp
        self.damping = damping
        self.factor_decay = factor_decay
        self.batch_averaged = batch_averaged
        self.eps = 1e-10
        self.A_rank = A_rank
        self.G_rank = G_rank

        # Should be set by implementing class
        self.has_bias = None

        # Note: each of the following is a list of tensors b/c
        # torch modules can have multiple inputs/outputs and weights
        self.a_inputs  = None
        self.g_outputs = None
        self.A_factor  = None
        self.G_factor  = None
        self.A_inv     = None
        self.G_inv     = None
        self.preconditioned_gradient = None

    def __repr__(self):
        return 'KFAC {}({})'.format(self.__class__.__name__, repr(self.module))

    def compute_A_inv(self, rank):
        """Compute A inverse on specified ranks

        Note: all ranks will enter this function but only the ranks assigned
        to this layer will continue to actually compute the inverses.
        All other ranks will simply zero out their inverse buffers for. This 
        is done so we can sum the inverses across all ranks to communicate the
        results of locally computed inverses.

        Args:
          rank (int): rank of worker entering function

        TODO(gpauloski): refactor this code and compute_G_invs to helper func
        """
        if self.A_rank is None:
            raise ValueError('Workers have not been assigned to layer yet.')
        if rank == self.A_rank:
            if self.use_eigen_decomp:
                self.A_inv.data.copy_(self._get_block_eigen(self.A_factor))
            else:
                self.A_inv.data.copy_(self._get_block_inv(self.A_factor))

    def compute_G_inv(self, rank):
        """Compute G inverse on specified ranks

        See `compute_A_inv` for more info`
        """
        if self.G_rank is None:
            raise ValueError('Workers have not been assigned to layer yet.')
        if rank == self.G_rank:
            if self.use_eigen_decomp:
                self.G_inv.data.copy_(self._get_block_eigen(self.G_factor))
            else:
                self.G_inv.data.copy_(self._get_block_inv(self.G_factor))

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

    def compute_preconditioned_gradient(self):
        """Compute precondition gradient of each weight in module

        Produces a list of preconditioned weight gradient and bias gradient
        (if there is a bias) where each has shape [output_dim, input_dim].
        Preconditioned gradients can be applied to the actual gradients with 
        `update_gradient()`.
        """
        # Compute preconditioned gradient using specified inverse method
        if self.use_eigen_decomp:
            grad = self._precondition_gradient_eigen()
        else:
            grad = self._precondition_gradient_inv()
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
        self.a_inputs = [a.data for a in input if a is not None]

    def save_grad_outputs(self, grad_output):
        """Save grad w.r.t outputs locally"""
        self.g_outputs = [g.data for g in grad_output if g is not None]

    def update_A_factor(self):
        """Compute factor A and add to running averages"""
        A_new = self._compute_A_factor()
        if self.A_factor is None:
            self._init_A_buffers(A_new)
        self._update_running_avg(A_new, self.A_factor)

    def update_G_factor(self):
        """Compute factor G and add to running averages"""
        G_new = self._compute_G_factor()
        if self.G_factor is None:
            self._init_G_buffers(G_new)
        self._update_running_avg(G_new, self.G_factor)

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
    
    def _compute_A_factor(self):
        """Compute A factor. Returns A."""
        raise NotImplementedError

    def _compute_G_factor(self):
        """Compute G factor. Returns G."""
        raise NotImplementedError

    def _get_block_eigen(self, block):
        """Compute eigendecomposition of a block.

        Args:
          block (tensor): block of shape (x, x) to eigendecompose

        Returns:
          Tensor of shape (x, x+1) where (0:x, 0:x) are the eigenvectors
          and (:, -1) is the eigenvalues
        """
        d, Q = torch.symeig(block, eigenvectors=True)
        d = torch.mul(d, (d > self.eps).float())
        d = d.unsqueeze(1)
        return torch.cat([Q, d], 1)

    def _get_block_inv(self, block):
        """Compute inverse of a block

        Adds a damping factor  `self.damping` to diagonal to prevent
        ill-conditioned matrix inversion.

        Args:
          block (tensor): block of shape (x, x) to invert

        Returns:
          Tensor of shape (x, x), the inverse of block
        """
        diag = block.new(block.shape[0]).fill_(self.damping)
        cholesky = torch.cholesky(block + torch.diag(diag))
        return torch.cholesky_inverse(cholesky)

    def _get_bias_grad(self):
        """Get bias.grad tensor of module"""
        return self.module.bias.grad

    def _get_weight_grad(self):
        """Get weight.grad tensor of module"""
        return self.module.weight.grad

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
    
    def _precondition_gradient_eigen(self):
        """Compute preconditioned gradient for eigendecomp method"""
        grad = self.get_gradient()
        QA, QG = self.A_inv[:,:-1], self.G_inv[:,:-1]
        dA, dG = self.A_inv[:,-1], self.G_inv[:,-1]
        v1 = QG.t() @ grad @ QA
        v2 = v1 / (dG.unsqueeze(1) * dA.unsqueeze(0) + self.damping)
        return QG @ v2 @ QA.t()

    def _precondition_gradient_inv(self):
        """Compute preconditioned gradient for inverse method"""
        grad = self.get_gradient() 
        return self.G_inv @ grad @ self.A_inv

    def _update_running_avg(self, new, current):
        """Computes in-place running average

        current = factor_decay*current + (1-factor_decay)*new

        Args:
          new (tensor): tensor to add to current average
          current (tensor): tensor containing current average. Result will be
              saved in place to this tensor.
        """
        if self.factor_decay != 1:
            current *= self.factor_decay / (1 - self.factor_decay)
            current += new
            current *= (1 - self.factor_decay)


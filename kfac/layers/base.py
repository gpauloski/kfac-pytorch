import math
import torch
import horovod.torch as hvd

from kfac.utils import get_block_boundary


class KFACLayer(object):
    def __init__(self,
                 module,
                 use_eigen_decomp=True,
                 damping = 0.001,
                 factor_decay=0.95,
                 batch_averaged=True):
        self.module = module
        self.use_eigen_decomp=use_eigen_decomp
        self.damping = damping
        self.factor_decay = factor_decay
        self.batch_averaged = batch_averaged
        self.eps = 1e-10

        # Should be set by implementing class
        self.has_bias = None
        self.num_weights = None

        # Note: each of the following is a list of tensors b/c
        # torch modules can have multiple inputs/outputs and weights
        self.a_inputs  = None
        self.g_outputs = None
        self.A_factors = None
        self.G_factors = None
        self.A_invs    = None
        self.G_invs    = None

    def clear_inverses(self):
        """Clear inverse buffers

        Useful for when switching between `diag_blocks=1` and `diag-blocks>1`
        because eigendecompositions saved in place and the off-diagonals must
        be cleared.
        """
        if self.A_invs is not None:
            for A_inv in self.A_invs:
                A_inv.fill_(0)
        if self.G_invs is not None:
            for G_inv in self.G_invs:
                G_inv.fill_(0)

    def compute_A_invs(self, ranks):
        """Compute A inverses on specified ranks

        Note: all ranks will enter this function but only the ranks specified
        in `ranks` will continue to actually compute the inverses.
        All other ranks will simply zero out their inverse buffers for. This 
        is done so we can sum the inverses across all ranks to communicate the
        results of locally computed inverses.

        TODO(gpauloski): refactor this code and compute_G_invs to helper func

        Args:
          ranks: list of horovod ranks (i.e. workers) to use.
        """
        if hvd.rank() in ranks:
            for factor, inv in zip(self.A_factors, self.A_invs):
                self._distributed_factor_inv(factor, inv, ranks)
        else:
            for inv in self.A_invs:
                inv.fill_(0)

    def compute_G_invs(self, ranks):
        """Compute G inverses on specified ranks

        See `compute_A_inv` for more info`
        """
        if hvd.rank() in ranks:
            for factor, inv in zip(self.G_factors, self.G_invs):
                self._distributed_factor_inv(factor, inv, ranks)
        else:
            for inv in self.G_invs:
                inv.fill_(0)

    def get_diag_blocks(self, diag_blocks):
        """Helper method for determining number of diag_blocks to use

        Overrides `diag_blocks` if the `module` does not support
        `diag_blocks>1`.
        """
        return 1

    def get_gradients(self):
        """Get formated gradients of each weight in module

        Returns:
          List of formatted gradients for each weight where each gradient has
          shape [ouput_dim, input_dim].
        """
        grads = []
        for i in range(self.num_weights):
            g = self._get_weight(i).grad.data
            if self.has_bias:
                g = torch.cat([g, self._get_bias(i).grad.data.view(-1, 1)], 1)
            grads.append(g)
        return grads

    def get_factor_handles(self, op=hvd.Average):
        """Get list of handles to call to average factors"""
        return ([hvd.allreduce_async_(a, op=op) for a in self.A_factors] +
                [hvd.allreduce_async_(g, op=op) for g in self.G_factors])

    def get_inverse_handles(self, op=hvd.Sum):
        """Get list of handles to call to sum inverses"""
        return ([hvd.allreduce_async_(a, op=op) for a in self.A_invs] +
                [hvd.allreduce_async_(g, op=op) for g in self.G_invs])

    def get_preconditioned_gradients(self):
        """Get precondition gradients of each weight in module

        Returns:
          List of preconditioned gradients for each weight where each has 
          shape [output_dim, input_dim].
        """
        # Compute preconditioned gradient using specified inverse method
        if self.use_eigen_decomp:
            grads = self._precondition_gradients_eigen()
        else:
            grads = self._precondition_gradients_inv()

        # Reshape appropriately
        for i, grad in enumerate(grads):
            if self.has_bias:
                grad = [grad[:, :-1], grad[:, -1:]]
                grad[0] = grad[0].view(self._get_weight(i).grad.data.size())
                grad[1] = grad[1].view(self._get_bias(i).grad.data.size())
            else:
                grad = [grad.view(self._get_weight(i).grad.data.size())]
            grads[i] = grad
        return grads

    def save_inputs(self, input):
        """Save inputs locally"""
        self.a_inputs = [a.data for a in input if a is not None]

    def save_grad_outputs(self, grad_output):
        """Save grad w.r.t outputs locally"""
        self.g_outputs = [g.data for g in grad_output if g is not None]

    def update_A_factors(self):
        """Compute factors A and add to running averages"""
        A_new = self._compute_A_factors()
        if self.A_factors is None:
            self._init_A_buffers(A_new)
        for A, A_factor in zip(A_new, self.A_factors):
            self._update_running_avg(A, A_factor)

    def update_G_factors(self):
        """Compute factors G and add to running averages"""
        G_new = self._compute_G_factors()
        if self.G_factors is None:
            self._init_G_buffers(G_new)
        for G, G_factor in zip(G_new, self.G_factors):
            self._update_running_avg(G, G_factor)

    def _compute_A_factors(self):
        """Compute A factors

        Returns:
          list of factors A for each weight in module
        """
        raise NotImplementedError

    def _compute_G_factors(self):
        """Compute G factors

        Returns:
          list of factors G for each weight in module
        """
        raise NotImplementedError

    def _distributed_factor_inv(self, factor, inv, ranks):
        """Computes the inverse of a factor across ranks

        Assigns each rank in `ranks` to enter this function to compute a
        diagonal block of `factor`. If `len(ranks)==1`, then that rank 
        computes the inv/eigendecomp of the entire `factor`.

        Args:
          factor (tensor): tensor to invert
          inv (tensor): tensor to save inplace inverse to
          ranks (list): list of ranks that will enter this function
        """
        i = ranks.index(hvd.rank())
        n = len(ranks)
        if n > min(factor.shape):
            n = min(factor.shape)

        if i < n:
            start, end = get_block_boundary(i, n, factor.shape)
            block = factor[start[0]:end[0], start[1]:end[1]]
            if self.use_eigen_decomp:
                block_inv = self._get_block_eigen(block)
                inv.data[start[0]:end[0], -1].copy_(block_inv[:,-1])
                inv.data[start[0]:end[0], start[1]:end[1]].copy_(
                        block_inv[:,:-1])
            else:
                block_inv = self._get_block_inv(block)
                inv.data[start[0]:end[0], start[1]:end[1]].copy_(block_inv)

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

    def _get_bias(self, i):
        """Get i'th bias tensor in module

        For most layers, there is only one bias tensor but certain layers
        like RNNs can have multiple.
        """
        raise NotImplementedError

    def _get_weight(self, i):
        """Get i'th weight tensor in module

        For most layers, there is only one weight but certain layers
        like RNNs can have multiple.
        """
        raise NotImplementedError

    def _init_A_buffers(self, factors):
        """Create buffers for each factor A and its inverse"""
        assert self.A_factors is None, ('A buffers have already been '
                'initialized. Was _init_A_buffers() called more than once?')
        self.A_factors = [factor.new(factor.shape).fill_(1) 
                          for factor in factors]
        if self.use_eigen_decomp:
            # add one axtra column for eigenvalues
            shapes = [(factor.shape[0], factor.shape[0] + 1) 
                      for factor in factors]
        else:
            shapes = [factor.shape for factor in factors]
        self.A_invs = [factor.new_zeros(shape) 
                       for factor, shape in zip(factors, shapes)]

    def _init_G_buffers(self, factors):
        """Create buffers for each factor G and its inverse"""
        assert self.G_factors is None, ('G buffers have already been '
                'initialized. Was _init_G_buffers() called more than once?')
        self.G_factors = [factor.new(factor.shape).fill_(1) 
                          for factor in factors]
        if self.use_eigen_decomp:
            # add one axtra column for eigenvalues
            shapes = [(factor.shape[0], factor.shape[0] + 1) 
                      for factor in factors]
        else:
            shapes = [factor.shape for factor in factors]
        self.G_invs = [factor.new_zeros(shape) 
                       for factor, shape in zip(factors, shapes)]
    
    def _precondition_gradients_eigen(self):
        """Compute preconditioned gradients for eigendecomp method"""
        grads = self.get_gradients()
        precon_grads = []
        for i, grad in enumerate(grads):
            A_inv, G_inv = self.A_invs[i], self.G_invs[i]
            QA, QG = A_inv[:,:-1], G_inv[:,:-1]
            dA, dG = A_inv[:,-1], G_inv[:,-1]
            v1 = QG.t() @ grad @ QA
            v2 = v1 / (dG.unsqueeze(1) * dA.unsqueeze(0) + self.damping)
            precon_grads.append(QG @ v2 @ QA.t())
        return precon_grads

    def _precondition_gradients_inv(self):
        """Compute preconditioned gradients for inverse method"""
        grads = self.get_gradients() 
        precon_grads = []
        for i, grad in enumerate(grads):
            precon_grads.append(self.G_invs[i] @ grad @ self.A_invs[i])
        return precon_grads

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


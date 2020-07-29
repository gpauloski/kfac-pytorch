import torch

from kfac.layers.base import KFACLayer

class LinearRNNLayer(KFACLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_bias = self.module.bias is not None
        self.use_eigen_decomp = True

        self.num_timesteps = None
        self.a_inputs = []
        self.g_outputs = []
        self.A_0 = None
        self.A_1 = None
        self.G_0 = None
        self.G_1 = None
        self.A_0_sqrt = None
        self.G_0_sqrt = None
        self.A_psi = None
        self.G_psi = None
        self.A_inv = None
        self.G_inv = None

    def compute_A_inv(self, rank):
        """Compute A inverse on specified ranks
        
        Args:
          rank (int): rank of worker entering function
        """
        if self.A_rank is None:
            raise ValueError('Workers have not been assigned to layer yet.')
        if rank == self.A_rank:
            self._preprocess(self.A_0, self.A_1, self.A_0_sqrt, self.A_psi, self.A_inv)

    def compute_G_inv(self, rank):
        """Compute G inverse on specified ranks
        See `compute_A_inv` for more info`
        """
        if self.G_rank is None:
            raise ValueError('Workers have not been assigned to layer yet.')
        if rank == self.G_rank:
            self._preprocess(self.G_0, self.G_1, self.G_0_sqrt, self.G_psi, self.G_inv)

    def get_factors(self):
        return [self.A_0, self.A_1, self.G_0, self.G_1]

    def get_inverses(self, return_ranks=False):
        tensors =  [self.A_0_sqrt, self.A_psi, self.A_inv,
                self.G_0_sqrt, self.G_psi, self.G_inv]
        if return_ranks:
            ranks = 3 * [self.A_rank] + 3 * [self.G_rank]
            return tensors, ranks
        return tensors

    def save_inputs(self, input):
        # Linear layer may be called multiple times in one training step
        # b/c of multiple timestep passes in the KFAC RNN implementation
        # so we accumulate the inputs/grad_outputs for each timestep
        self.a_inputs.append(input[0].data)

    def save_grad_outputs(self, grad_output):
        # See save_inputs() above
        self.g_outputs.append(grad_output[0].data)

    def update_A_factor(self):
        """Compute factor A and add to running averages"""
        A_0_new, A_1_new = self._compute_A_factor()
        if self.A_0 is None:
            self._init_A_buffers(A_0_new)
        self._update_running_avg(A_0_new, self.A_0)
        # if seq len in 1, A_1 will be none
        if A_1_new is not None:
            self._update_running_avg(A_1_new, self.A_1)

    def update_G_factor(self):
        """Compute factor G and add to running averages"""
        G_0_new, G_1_new = self._compute_G_factor()
        if self.G_0 is None:
            self._init_G_buffers(G_0_new)
        self._update_running_avg(G_0_new, self.G_0)
        if G_1_new is not None:
            self._update_running_avg(G_1_new, self.G_1)

    def _compute_A_factor(self):
        self.num_timesteps = len(self.a_inputs)
        A_0 = None
        for t in range(self.num_timesteps):
            a_t = self.a_inputs[t]
            batch_size = a_t.size(0)

            if self.has_bias:
                a_t = torch.cat([a_t, a_t.new(a_t.size(0), 1).fill_(1)], 1)
            A = a_t.t() @ (a_t / batch_size)

            if A_0 is None:
                A_0 = A
            else:
                A_0 += A

        A_1 = None
        for t in range(self.num_timesteps - 1):
            a_t = self.a_inputs[t]
            a_s = self.a_inputs[t + 1]
            batch_size = a_t.size(0)

            if self.has_bias:
                a_t = torch.cat([a_t, a_t.new(a_t.size(0), 1).fill_(1)], 1)
                a_s = torch.cat([a_s, a_s.new(a_s.size(0), 1).fill_(1)], 1)
            A = a_s.t() @ (a_t / batch_size)

            if A_1 is None:
                A_1 = A
            else:
                A_1 += A

        if A_0 is not None:
            A_0 /= self.num_timesteps
        if A_1 is not None:
            A_1 /= self.num_timesteps

        self.a_inputs = []  # Clear input accumulation

        return A_0, A_1

    def _compute_G_factor(self):
        self.num_timesteps = len(self.g_outputs)
        G_0 = None
        for t in range(self.num_timesteps):
            g_t = self.g_outputs[t]
            batch_size = g_t.size(0)

            if self.batch_averaged:
                G = g_t.t() @ (g_t * batch_size)  # TODO removed (just divide)
            else:
                G = g_t.t() @ (g_t / batch_size)

            if G_0 is None:
                G_0 = G
            else:
                G_0 += G

        G_1 = None
        for t in range(self.num_timesteps - 1):
            g_t = self.g_outputs[t]
            g_s = self.g_outputs[t + 1]
            batch_size = g_t.size(0)

            if self.batch_averaged:
                G = g_s.t() @ (g_t * batch_size)
            else:
                G = g_s.t() @ (g_t / batch_size)

            if G_1 is None:
                G_1 = G
            else:
                G_1 += G

        if G_0 is not None:
            G_0 /= self.num_timesteps
        if G_1 is not None:
            G_1 /= self.num_timesteps

        self.g_outputs = []  # Clear grad_output accumulation now that we have
                             # computed G for these grad_outputs
        return G_0, G_1

    def _init_A_buffers(self, factor):
        """Initialize memory for factors and inverses/eigendecompositions"""
        self.A_0 = torch.diag(factor.new(factor.shape[0]).fill_(1))
        self.A_1 = torch.diag(factor.new(factor.shape[0]).fill_(1))
        self.A_0_sqrt = factor.new_empty(self.A_0.shape)
        self.A_psi = factor.new_empty(self.A_0.shape)
        self.A_inv = factor.new_zeros((factor.shape[0], factor.shape[0] + 1)) 

    def _init_G_buffers(self, factor):
        """Initialize memory for factors and inverses/eigendecompositions"""
        self.G_0 = torch.diag(factor.new(factor.shape[0]).fill_(1))
        self.G_1 = torch.diag(factor.new(factor.shape[0]).fill_(1))
        self.G_0_sqrt = factor.new_empty(self.G_0.shape)
        self.G_psi = factor.new_empty(self.G_0.shape)
        self.G_inv = factor.new_zeros((factor.shape[0], factor.shape[0] + 1))

    def _factor_invsqrt(self, factor):
        """Computes the matrix inverse square root of a symmetric factor"""
        d, Q = torch.symeig(factor, eigenvectors=True)
        invsqrt = (Q * torch.rsqrt(d + self.damping)) @ Q.t()
        return (invsqrt + invsqrt.t()) / 2.0  # force symmetry

    def _constrained_eigen_decomp(self, factor):
        """Computes a constrained eigen decomposition

        Contrains eigenvalues to min(1.0, e)
        
        Args:
          factor (torch.Tensor): n x n square tensor to eigen decompose

        Returns:
          Tensor of shape (n, n+1) where (0:n, 0:n) are the eigenvectors
          and (:, -1) is the eigenvalues
        """
        d, Q = torch.symeig(factor, eigenvectors=True)
        d = torch.mul(d, (d > self.eps).float())
        d = torch.min(d, d.new_ones([1]))
        return torch.cat([Q, d.unsqueeze(1)], 1)
        

    def _precondition_gradient_eigen(self):
        """Computes the preconditioned gradient

        Performs the preconditioning steps for option 2 defined in Appendix C.3
        of https://openreview.net/pdf?id=HyMTkQZAb
        See https://github.com/tensorflow/kfac/blob/master/kfac/python/ops/fisher_blocks.py
        """
        grad = self.get_gradient()
        QA, QG = self.A_inv[:,:-1], self.G_inv[:,:-1]
        dA, dG = self.A_inv[:,-1], self.G_inv[:,-1]

        Z_0 = self.G_0_sqrt @ grad @ self.A_0_sqrt
        Z_1 = Z_0 - (self.G_psi.t() @ Z_0 @ self.A_psi)
        a = QG.t() @ Z_1 @ QA
        b = 1.0 - (dG[..., None] @ dA[None, ...])
        b[b == 0.0] = 1.0  # prevent numerical issues by setting 0.0 values to 1.0
        Z_2 = QG @ torch.div(a, b) @ QA.t()
        Z_3 = Z_2 - (self.G_psi @ Z_2 @ self.A_psi.t())
        Z_4 = Z_3 / self.num_timesteps
        Z_5 = self.G_0_sqrt @ Z_4 @ self.A_0_sqrt

        return Z_5

    def _preprocess(self, C_0, C_1, C_0_sqrt, C_psi, C_inv):
        """Performs factor transformation and eigendecomposition

        Performs the preprocessing steps for option 2 defined in Appendix C.3
        of https://openreview.net/pdf?id=HyMTkQZAb
        See https://github.com/tensorflow/kfac/blob/master/kfac/python/ops/fisher_factors.py#L2631

        Note:
          C is used to represent the factor A or G as the preprocessing is
          identical for both A and G.

        Args:
          C_0: factor, sum(c_t * c_t.t()) for t in [1, T]
          C_1: factor, sum(c_t+1 * c_t.t()) for t in [1, T-1]
          C_0_sqrt: tensor to save matrix square-root of C_0
          C_psi: tensor to save transformed factor to
          C_inv: tensor to save eigen decomposition to
        """
        C_0_sqrt.data.copy_(self._factor_invsqrt(C_0))
        C_psi.data.copy_(C_0_sqrt @ C_1 @ C_0_sqrt)
        C_inv.data.copy_(self._constrained_eigen_decomp(C_psi.t() @ C_psi))

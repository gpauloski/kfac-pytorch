import torch

import horovod.torch as hvd

from kfac.layers.base import KFACLayer

class EmbeddingLayer(KFACLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_bias = False
        self.use_eigen_decomp = False

    def _init_buffers_A(self, factor):
        """Create buffers for factor A and its inv
        For embedding layers, A is a diagonal matrix so we just store the
        diagonal to save memory.
        """
        self.A_factor = factor.new(self.module.num_embeddings).fill_(1)
        if self.use_eigen_decomp:
            raise NotImplementedError('Use inv method for embedding layers')
        else:
            shape = factor.shape
        self.A_inv = factor.new_zeros(shape)

    def _compute_A(self):
        """Compute A for Embedding layer
        Input to Embedding layer is (batch_size, input_size) representing
        indicies into the embedding matrix of size [vocab_size, embed_size].
        The factor is represented by:
          (1/batch_size) sum_{i} diag(one_hot(inputs[i]) ** 2)
        where inputs[i] is the input for the ith batch and the output is size
        [vocab_size, vocab_size]
        source: https://github.com/tensorflow/kfac/blob/master/kfac/python/ops/fisher_factors.py#L1107
        """
        a = self.a_input
        batch_size = a.size(0)
        tokens = torch.LongTensor(batch_size, self.module.num_embeddings).cuda()
        tokens.zero_()
        tokens.scatter_(1, a, 1)
        return torch.mean(tokens.float(), dim=0)

    def _compute_G(self):       
        g = self.g_output
        batch_size = g.size(0)
        if len(g.shape) > 2:
            # TODO(gpauloski): should we average middle dim here?
            g = torch.mean(g, list(range(len(g.shape)))[1:-1])
        if self.batch_averaged:
            G = g.t() @ (g * batch_size)
        else:
            G = g.t() @ (g / batch_size)
        return G
    
    def _get_vector_inv(self, v):
        """Compute inverse of each non-zero element of v"""
        assert len(v.shape) == 1
        idx = v.nonzero()
        v[idx[:, 0]] = 1 / v[idx[:, 0]]
        return v

    def _get_block_eigen(self, block):
        """Compute eigendecomposition of tensor. Append eigenvalues"""
        raise NotImplementedError('Use inv method for embedding layers')

    def _precondition_gradient_inv(self):
        """Compute preconditioned gradient for inverse method
        Note: For embedding layers, A is a diagonal matrix stored as a 1-D
        tensors of the diagonal and the gradient is (input_size, output_size).
        The KFAC update expects the gradient to be (output_size, input_size)
        so we use this update:
            precon_grad = (G_inv @ grad.t()) * A_inv
        instead of the traditional:
            precon_grad = G_inv.t() @ grad @ A_inv
        where @ is torch.matmul() and * is torch.mv()/
        """
        grad = self.get_gradient()
        return torch.matmul(self.G_inv, grad.t()) * self.A_inv
    
    def _precondition_gradient_eigen(self):
        """Compute preconditioned gradient for eigendecomp method"""
        raise NotImplementedError('Use inv method for embedding layers')
    
    def compute_A_inv(self, ranks):
        """Compute inv of A for module on specified workers
        Note: Embedding layer currently ignores all but first rank
        Args:
          ranks: list of horovod ranks (i.e. workers) to use.
        """
        if hvd.rank() == ranks[0]:
            self.A_inv.copy_(self._get_vector_inv(self.A_factor))
        else:
            self.A_inv.fill_(0)

    def compute_G_inv(self, ranks):
        """Compute inv of G for module on specified workers
        See `compute_A_inv` for more info`
        """
        if hvd.rank() == ranks[0]:
            self.G_inv.copy_(self._get_block_inv(self.G_factor))
        else:
            self.G_inv.fill_(0)

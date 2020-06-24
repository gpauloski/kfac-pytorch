import torch

from kfac.layers.base import KFACLayer

class EmbeddingLayer(KFACLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_bias = False
        self.use_eigen_decomp = False
        self.num_weights = 1
 
    def compute_A_invs(self, rank):
        """Compute inv of A for module on specified workers

        Note: Embedding layer currently ignores all but first rank

        Args:
          rank (int) rank of worker entering function.
        """
        # TODO assert only one rank computes a b/c its diagonal
        if rank == self.A_ranks[0]:
            for factor, inv in zip(self.A_factors, self.A_invs):
                inv.copy_(self._get_vector_inv(factor))
        else:
            for inv in self.A_invs:
               inv.fill_(0)

    def get_gradients(self):
        return [self.module.weight.grad.data]

    def _compute_A_factors(self):
        """Compute A for Embedding layer

        Input to Embedding layer is (batch_size, input_size) representing
        indicies into the embedding matrix of size [vocab_size, embed_size].
        The factor is represented by:
          (1/batch_size) sum_{i} diag(one_hot(inputs[i]) ** 2)
        where inputs[i] is the input for the ith batch and the output is size
        [vocab_size, vocab_size]
        source: https://github.com/tensorflow/kfac/blob/master/kfac/python/ops/fisher_factors.py#L1107
        """
        assert len(self.a_inputs) == 1
        a = self.a_inputs[0]
        batch_size = a.size(0)
        tokens = torch.LongTensor(batch_size, self.module.num_embeddings).cuda()
        tokens.zero_()
        tokens.scatter_(1, a, 1)
        return [torch.mean(tokens.float(), dim=0)]  # TODO(gpauloski) why batch average here?

    def _compute_G_factors(self):       
        # g: batch_size * out_dim
        assert len(self.g_outputs) == 1
        g = self.g_outputs[0]
        batch_size = g.size(0)
        if len(g.shape) > 2:
            g = g[:, 0]  # TODO(gpauloski) this is a quick fix for mismatch dims
        if self.batch_averaged:
            return [g.t() @ (g * batch_size)]
        else:
            return [g.t() @ (g / batch_size)]
    
    def _get_bias(self, i):
        raise ValueError('Embedding layer does not have bias')
 
    def _get_block_eigen(self, block):
        """Compute eigendecomposition of tensor. Append eigenvalues"""
        raise NotImplementedError('Use inv method for embedding layers')

    def _get_vector_inv(self, v):
        """Compute inverse of each non-zero element of v"""
        assert len(v.shape) == 1
        idx = v.nonzero()
        v[idx[:, 0]] = 1 / v[idx[:, 0]]
        return v

    def _get_weight(self, i):
        if i == 0:
            return self.module.weight
        else:
            raise ValueError('Invalid weight index {}. Embedding layer only '
                             'has 1 weight tensor'.format(i))

    def _init_A_buffers(self, factors):
        """Create buffers for factor A and its inv
        For embedding layers, A is a diagonal matrix so we just store the
        diagonal to save memory.
        """
        assert len(factors) == 1
        factor = factors[0]
        self.A_factors = [factor.new(self.module.num_embeddings).fill_(1)]
        if self.use_eigen_decomp:
            raise NotImplementedError('Use inv method for embedding layers')
        else:
            shape = factor.shape
        self.A_invs = [factor.new_zeros(shape)]
    
    def _precondition_gradients_eigen(self):
        """Compute preconditioned gradient for eigendecomp method"""
        raise NotImplementedError('Use inv method for embedding layers')

    def _precondition_gradients_inv(self):
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
        grads = self.get_gradients()
        precon_grads = []
        for i, grad in enumerate(grads):
            precon_grads.append(
                    torch.matmul(self.G_invs[i], grad.t()) * self.A_invs[i])
        return precon_grads


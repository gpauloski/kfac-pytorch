import torch

from kfac.layers.base import KFACLayer

class EmbeddingLayer(KFACLayer):
    """
    Note: 
      Defaults to batch_first=False

    Note: 
      If the embedding weights are tied with the Linear "decoding" weights, then
      the forward and backward pass contributions from the Linear module will not be
      correctly incorporated.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_bias = False
        self.use_eigen_decomp = False
 
    def compute_A_inv(self, rank):
        if self.A_rank is None:
            raise ValueError('Workers have not been assigned to layer yet.')
        if rank == self.A_rank:
            self.A_inv.copy_(self._get_vector_inv(self.A_factor))
        else:
            self.A_inv.fill_(0)

    def _compute_A_factor(self):
        """Compute A for Embedding layer

        Input to Embedding layer is (seq_len, batch_size) or (batch_size, seq_len)
        representing indicies into the embedding matrix of size (vocab_size, embed_size).
        The factor is represented by (1/batch_size) sum_{i} diag(one_hot(inputs[i]) ** 2)
        where inputs[i] is the input for the ith batch and the output is size [vocab_size, 
        vocab_size].

        Note: If the embedding matrix has its weights tied to a decoding linear layer,
          and the layers are registerd as a shared weight KFAC layer, then self.a_inputs 
          will be a mix of tensors of shape (seq_len, batch_size) and (seq_len, batch_size,
          num_tokens). I.e. the contributions from the linear layer are already one hot
          encoded so we just need to one hot encode the contributions from the embedding
          layer.

        Reference: 
          https://github.com/tensorflow/kfac/blob/master/kfac/python/ops/fisher_factors.py#L1107
        """
        self.a_inputs = self.a_inputs[:1]
        batch_dim = int(not(self.batch_first))
        # one hot encode all non-one-hot encoded inputs
        for i, a in enumerate(self.a_inputs):
            if a.size(-1) != self.module.num_embeddings:
                one_hot = torch.nn.functional.one_hot(a.long(), 
                        num_classes=self.module.num_embeddings)
                self.a_inputs[i] = one_hot.float()
        a = self._reshape_data(self.a_inputs, collapse_dims=True)
        a = a ** 2
        return  torch.mean(a, dim=0)

    def _compute_G_factor(self):
        #for i, g in enumerate(self.g_outputs):
        #    print(i, g.shape, torch.min(g), torch.max(g))
        self.g_outputs = self.g_outputs[:1]
        g = self._reshape_data(self.g_outputs, collapse_dims=True)
        # The output of the embedding layer is (*, H) and since the input is at
        # least shape (input_size, batch_size) (or batch_first), then the
        # output has at least 3 dimensions so after reshaping, the batch_dim
        # will be 0 regardless of self.batch_first.
        #print('g1', torch.min(g), torch.max(g), g.shape)
        batch_size = g.size(0)
        if self.batch_averaged:
            g = g.t() @ (g * batch_size)
        else:
            g = g.t() @ (g / batch_size)
        #print('g2', torch.min(g), torch.max(g), g.shape)
        return g
 
    def _get_block_eigen(self, block):
        """Compute eigendecomposition of tensor. Append eigenvalues"""
        raise NotImplementedError('Use inv method for embedding layers')

    def _get_vector_inv(self, v):
        """Compute inverse of each non-zero element of v"""
        assert len(v.shape) == 1
        idx = v.nonzero()
        v[idx[:, 0]] = 1 / v[idx[:, 0]]
        return v

    def _init_A_buffers(self, factor):
        """Create buffers for factor A and its inv
        For embedding layers, A is a diagonal matrix so we just store the
        diagonal to save memory.
        """
        self.A_factor = factor
        if self.use_eigen_decomp:
            raise NotImplementedError('Use inv method for embedding layers')
        else:
            shape = factor.shape
        self.A_inv = factor.new_zeros(shape)
    
    def _precondition_gradient_eigen(self):
        """Compute preconditioned gradient for eigendecomp method"""
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
        return (self.A_inv[:, None] * grad) @ self.G_inv


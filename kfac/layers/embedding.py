import torch

from . import utils
from .base import KFACLayer


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
        super(EmbeddingLayer, self).__init__(*args, **kwargs)
        # TODO(gpauloski): update embedding class to support recent KFAC changes
        raise ValueError('Embedding layer does not currently work')
        self.has_bias = False
        self.use_eigen_decomp = False
 
    def compute_A_inv(self, rank):
        if self.A_rank is None:
            raise ValueError('Workers have not been assigned to layer yet.')
        if rank == self.A_rank: 
            self.A_inv.copy_(
                utils.get_elementwise_inverse(self.A_factor, damping=self.damping)
            )

    def _get_A_factor(self, a_inputs):
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
        # one hot encode all non-one-hot encoded inputs
        for i, a in enumerate(a_inputs):
            if a.size(-1) != self.module.num_embeddings:
                one_hot = torch.nn.functional.one_hot(a.long(), 
                        num_classes=self.module.num_embeddings)
                a_inputs[i] = one_hot.float()
        a = utils.reshape_data(a_inputs, batch_first=self.batch_first, 
                collapse_dims=True)
        assert a.size(-1) == self.module.num_embeddings
        assert len(a.shape) == 2  # shape should be (batch, vocab_size) where batch dim
                                  # has size batch_size * seq_len
        a = a ** 2
        return  torch.mean(a, dim=0)

    def _get_G_factor(self, g_outputs):
        g = utils.reshape_data(g_outputs, batch_first=self.batch_first, 
                collapse_dims=True)
        assert len(g.shape) == 2
        return utils.get_cov(g)

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
    
    def _get_precondition_gradient_eigen(self):
        """Compute preconditioned gradient for eigendecomp method"""
        raise NotImplementedError('Use inv method for embedding layers')

    def _get_precondition_gradient_inv(self):
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


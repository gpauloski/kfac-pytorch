import torch

from kfac.layers.base import KFACLayer

class LinearLayer(KFACLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_weights = 1
        self.has_bias = self.module.bias is not None

    def get_diag_blocks(self, diag_blocks):
        return diag_blocks

    def _compute_A_factors(self):
        # a: batch_size * in_dim
        assert len(self.a_inputs) == 1
        a = self.a_inputs[0]
        batch_size = a.size(0)
        # If previous layer is embedding/RNN, a will be shape
        # [batch_size, time_steps, emb_size] so we just take the first
        # time_step as a simplification.
        if len(a.shape) > 2:
            a = a[:, 0]
        if self.has_bias:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        return [a.t() @ (a / batch_size)]

    def _compute_G_factors(self):       
        # g: batch_size * out_dim
        assert len(self.g_outputs) == 1
        g = self.g_outputs[0]
        batch_size = g.size(0)
        if len(g.shape) > 2:
            g = g[:, 0]
        if self.batch_averaged:
            return [g.t() @ (g * batch_size)]
        else:
            return [g.t() @ (g / batch_size)]

    def _get_bias(self, i):
        if  self.has_bias and i == 0:
            return self.module.bias
        elif self.has_bias and i != 0:
            raise ValueError('Invalid bias index {}. Linear layer only has 1 '
                             'bias tensor'.format(i))
        else:
            return None

    def _get_weight(self, i):
        if i == 0:
            return self.module.weight
        else:
            raise ValueError('Invalid weight index {}. Linear layer only has '
                             '1 weight tensor'.format(i))
 

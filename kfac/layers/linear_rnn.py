import torch

from kfac.layers.base import KFACLayer

class LinearRNNLayer(KFACLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_bias = self.module.bias is not None
        self.is_symmetric = False
        self.num_weights = 1
        self.a_inputs = []
        self.g_outputs = []

    def get_diag_blocks(self, diag_blocks):
        return diag_blocks

    def save_inputs(self, input):
        # Linear layer may be called multiple times in one training step
        # b/c of multiple timestep passes in the KFAC RNN implementation
        # so we accumulate the inputs/grad_outputs for each timestep
        self.a_inputs.append(input[0].data)

    def save_grad_outputs(self, grad_output):
        # See save_inputs() above
        self.g_outputs.append(grad_output[0].data)

    def _compute_A_factors(self):
        # a: batch_size * in_dim
        a_o = []
        for t in range(len(self.a_inputs))
            a_t = self.a_inputs[t]
            batch_size = a_t.size(0)

            if self.has_bias:
                a_t = torch.cat([a_t, a_t.new(a_t.size(0), 1).fill_(1)], 1)
            a_t = a_t.t() @ (a_t / batch_size)
            a_o.append(a_t)

        a_o = torch.stack(a_o)
        a_o = torch.mean(a_o, dim=0)

        self.a_inputs = []  # Clear input accumulation
        return [a_o]

    def _compute_G_factors(self):       
        # g: batch_size * out_dim
        g_o = []
        for t in range(len(self.g_outputs)):
            g_t = self.g_outputs[t]
            batch_size = g_t.size(0)

            if self.batch_averaged:
                g_t = g_t.t() @ (g_t * batch_size)
            else:
                g_t = g_t.t() @ (g_t / batch_size)
            g_o.append(g_t)

        g_o = torch.stack(g_o)
        g_o = torch.mean(g_o, dim=0)

        self.g_outputs = []  # Clear grad_output accumulation now that we have
                             # computed G for these grad_outputs
        return [g_o]

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
 

import torch

from kfac.layers.base import KFACLayer

class LinearRNNLayer(KFACLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_bias = self.module.bias is not None
        self.a_inputs = []
        self.g_outputs = []

    def save_inputs(self, input):
        # Linear layer may be called multiple times in one training step
        # b/c of multiple timestep passes in the KFAC RNN implementation
        # so we accumulate the inputs/grad_outputs for each timestep
        self.a_inputs.append(input[0].data)

    def save_grad_outputs(self, grad_output):
        # See save_inputs() above
        self.g_outputs.append(grad_output[0].data)

    def _compute_A_factor(self):
        # a: batch_size * in_dim
        a_o = None
        for t in range(len(self.a_inputs)):
            a_t = self.a_inputs[t]
            batch_size = a_t.size(0)

            if self.has_bias:
                a_t = torch.cat([a_t, a_t.new(a_t.size(0), 1).fill_(1)], 1)
            a_t = a_t.t() @ (a_t / batch_size)

            if a_o is None:
                a_o = a_t
            else:
                a_o += a_t

        self.a_inputs = []  # Clear input accumulation

        return a_o

    def _compute_G_factor(self): 
        # g: batch_size * out_dim
        g_o = None
        for t in range(len(self.g_outputs)):
            g_t = self.g_outputs[t]
            batch_size = g_t.size(0)

            if self.batch_averaged:
                g_t = g_t.t() @ (g_t * batch_size)
            else:
                g_t = g_t.t() @ (g_t / batch_size)

            if g_o is None:
                g_o = g_t
            else:
                g_o += g_t

        self.g_outputs = []  # Clear grad_output accumulation now that we have
                             # computed G for these grad_outputs

        return g_o


import torch

from kfac.layers.base import KFACLayer


class LinearLayer(KFACLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_bias = self.module.bias is not None

    def _compute_A_factor(self):
        # a: batch_size * in_dim
        a = self._reshape_data(self.a_inputs, collapse_dims=True)
        batch_size = a.size(0)
        if self.has_bias:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        return a.t() @ (a / batch_size)

    def _compute_G_factor(self): 
        # g: batch_size * out_dim
        g = self._reshape_data(self.g_outputs, collapse_dims=True)
        batch_size = g.size(0)
        if self.batch_averaged:
            return g.t() @ (g * batch_size)
        else:
            return g.t() @ (g / batch_size)


class LinearMultiLayer(KFACLayer):
    """KFAC Layer for Linear modules called mutliple times.

    Broadly based on `FullyConnectedMultiKF` in https://github.com/tensorflow/kfac.
    Used for RNNs/LSTMs where forward() gets called once for each timestep.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_bias = self.module.bias is not None
        self.a_inputs = []
        self.g_outputs = []

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

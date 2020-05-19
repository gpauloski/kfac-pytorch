import torch

from kfac.layers.base import KFACLayer

class RNNLayer(KFACLayer):
    """
    Limitations:
      num_layers=1
      batch_first=True
      bidirectional=False
      bias=True
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(dir(self.module))
        #print('flat_weight_names', self.module._flat_weights_names)
        #print('flat_weights', type(self.module._flat_weights), len(self.module._flat_weights))
        for j, p in enumerate(self.module._flat_weights):
            print(j, p.shape)
        def _save_hidden_output(grad):
            print('hidden grad', grad.shape)
            self._g_hidden = grad
        self.module.weight_hh_l0.register_hook(_save_hidden_output)
        self.has_bias = True

    def save_input(self, a):
        """Save input `a` locally

        For RNN, we have to save input and hidden state.
        """
        print('a', a[0].shape, a[1].shape)
        self.a_input = a[0].data
        self.a_hidden = a[1].data

    def save_grad_output(self, g):
        """Save grad w.r.t output `g` locally"""
        print('g', type(g), len(g))
        for i in range(len(g)):
            if g[i] is not None:
                print('output', i, '/', len(g), g[i].shape)
        self.g_output = g[0].data

    def get_diag_blocks(self, diag_blocks):
        return 1

    def _compute_A(self):
        # a: batch_size * in_dim
        a = self.a_input
        batch_size = a.size(0)
        if len(a.shape) > 2:
            a = torch.mean(a, list(range(len(a.shape)))[1:-1])
        if self.module.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        return a.t() @ (a / batch_size)

    def _compute_G(self):       
        # g: batch_size * out_dim
        g = self.g_output
        batch_size = g.size(0)
        if len(g.shape) > 2:
            g = torch.mean(g, list(range(len(g.shape)))[1:-1])
        if self.batch_averaged:
            return g.t() @ (g * batch_size)
        else:
            return g.t() @ (g / batch_size)

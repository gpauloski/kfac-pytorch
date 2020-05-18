import torch

from kfac.layers.base import KFACLayer

class LinearLayer(KFACLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_bias = True

    def get_diag_blocks(self, diag_blocks):
        return diag_blocks

    def _compute_A(self):
        # a: batch_size * in_dim
        a = self.a_input
        batch_size = a.size(0)
        # TODO(gpauloski) should we even bother averaging seq_len dim
        # if we do not use linear for RNNs
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

import torch

from . import utils
from .base import KFACLayer


class LinearLayer(KFACLayer):
    def __init__(self, *args, **kwargs):
        super(LinearLayer, self).__init__(*args, **kwargs)
        self.has_bias = self.module.bias is not None

    def _get_A_factor(self, a_inputs):
        # a: batch_size * in_dim
        a = utils.reshape_data(a_inputs, batch_first=self.batch_first,
                collapse_dims=True)
        if self.has_bias:
            a = utils.append_bias_ones(a)
        return utils.get_cov(a)

    def _get_G_factor(self, g_outputs): 
        # g: batch_size * out_dim
        g = utils.reshape_data(g_outputs, batch_first=self.batch_first,
                collapse_dims=True)
        return utils.get_cov(g)


class LinearMultiLayer(LinearLayer):
    """KFAC Layer for Linear modules called mutliple times.

    Broadly based on `FullyConnectedMultiKF` in https://github.com/tensorflow/kfac.
    Used for RNNs/LSTMs where forward() gets called once for each timestep.
    """
    def __init__(self, *args, **kwargs):
        super(LinearMultiLayer, self).__init__(*args, **kwargs)
        self.has_bias = self.module.bias is not None

    def _get_A_factor(self, a_inputs):
        a_0 = None
        for a in a_inputs:
            a = super(LinearMultiLayer, self)._get_A_factor([a])

            if a_0 is None:
                a_0 = a
            else:
                a_0 += a
        
        return a_0

    def _get_G_factor(self, g_outputs): 
        g_0 = None
        for g in g_outputs:
            g = super(LinearMultiLayer, self)._get_G_factor([g])
            
            if g_0 is None:
                g_0 = g
            else:
                g_0 += g

        return g_0


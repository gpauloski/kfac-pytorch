import torch.nn as nn

import kfac.modules as km
from kfac.layers.conv import Conv2dLayer
from kfac.layers.embedding import EmbeddingLayer
from kfac.layers.linear import LinearLayer
from kfac.layers.rnn import RNNLayer

__all__ = ['get_kfac_layer', 'KFAC_LAYERS', 'KNOWN_MODULES',
           'module_requires_grad']

# PyTorch modules that have KFACLayer implementations.
KFAC_LAYERS = {'linear', 'conv2d', 'embedding'}

# PyTorch/KFAC parent modules that can be registered using KFACLayers.
# This is different from `KFAC_LAYERS` because some modules (e.g. 
# kfac.modules.LSTMCell) are made up of PyTorch modules in `KFAC_Layer`
# that we want to register even if the user has decided to manually skip
# that type of module. E.g. a user may want to skip Linear layers but
# register RNNCells; however, RNNCells is made up of Linear layers that we
# do not want to skip, hence this separation of `KFAC_LAYERS` and 
# `KNOWN_MODULES`.
KNOWN_MODULES = {'linear', 'conv2d', 'embedding', 'LSTMCell', 'RNNCell',
                 'LSTMImpl', 'RNNImpl'}

def get_kfac_layer(module, use_eigen_decomp=True, damping=0.001,
                   factor_decay=0.95, batch_averaged=True):
    if isinstance(module, nn.Linear):
        layer = LinearLayer
    elif isinstance(module, nn.Conv2d):
        layer = Conv2dLayer
    elif isinstance(module, km.RNNCell) or isinstance(module, km.LSTMCell):
        layer = RNNLayer
    elif isinstance(module, nn.Embedding):
        layer = EmbeddingLayer
    elif isinstance(module, nn.RNNCellBase):
        raise TypeError('KFAC does not support torch.nn.{RNN,LSTM}Cell. Use '
                        'kfac.modules.{RNN,LSTM}Cell instead for KFAC support.')
    else:
        raise NotImplementedError('KFAC does not support layer {}'.format(
                                  module.__class__.__name__))

    return layer(module, use_eigen_decomp, damping, factor_decay,
                 batch_averaged)

def module_requires_grad(module):
    """Returns False if any module param has .requires_grad=False"""
    for param in module.parameters():
        if not param.requires_grad:
            return False
    return True

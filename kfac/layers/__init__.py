import torch.nn as nn

from kfac.layers.conv import Conv2dLayer
from kfac.layers.embedding import EmbeddingLayer
from kfac.layers.linear import LinearLayer
from kfac.layers.rnn import RNNLayer

__all__ = ['get_kfac_layer', 'KNOWN_MODULES', 'module_requires_grad']

KNOWN_MODULES = {'linear', 'conv2d', 'embedding'}

def get_kfac_layer(module, use_eigen_decomp=True, damping=0.001,
                   factor_decay=0.95, batch_averaged=True):
    if isinstance(module, nn.Linear):
        layer = LinearLayer
    elif isinstance(module, nn.Conv2d):
        layer = Conv2dLayer
    elif isinstance(module, nn.RNNCellBase):
        layer = RNNLayer
    elif isinstance(module, nn.Embedding):
        layer = EmbeddingLayer
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

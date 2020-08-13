import torch.nn as nn

import kfac.modules as km
from kfac.layers.conv import Conv2dLayer
from kfac.layers.embedding import EmbeddingLayer
from kfac.layers.linear import LinearLayer
from kfac.layers.linear import LinearMultiLayer

__all__ = ['KNOWN_MODULES', 'get_kfac_layers', 'module_requires_grad']

KNOWN_MODULES = {'linear', 'conv2d', 'embedding', 'lstmcell'}

def get_kfac_layers(module, **kwargs):
    """Instantiates KFACLayer(s) for module

    Args:
      module: module to register
      **kwargs: parameters to pass to KFACLayer

    Returns:
      list of tuples where each tuple is (module, KFACLayer)
    """
    if isinstance(module, nn.Linear):
        return [(module, LinearLayer(module, **kwargs))]
    elif isinstance(module, nn.Conv2d):
        return [(module, Conv2dLayer(module, **kwargs))]
    elif isinstance(module, nn.Embedding):
        return [(module, EmbeddingLayer(module, **kwargs))]
    elif isinstance(module, km.LSTMCellBase):
        return [(m, LinearMultiLayer(m, **kwargs)) for m in module.children()]
    elif isinstance(module, nn.RNNCellBase):
        raise TypeError('KFAC does not support torch.nn.{RNN,LSTM}Cell. Use '
                        'kfac.modules.{RNN,LSTM}Cell instead for KFAC support.')
    else:
        raise NotImplementedError('KFAC does not support layer {}'.format(
                                  module.__class__.__name__))

def module_requires_grad(module):
    """Returns False if any module param has .requires_grad=False"""
    return all([p.requires_grad for p in module.parameters()])

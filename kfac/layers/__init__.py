import torch.nn as nn

from kfac.layers.conv import Conv2dLayer
from kfac.layers.embedding import EmbeddingLayer
from kfac.layers.linear import LinearLayer
from kfac.layers.rnn import RNNLayer

__all__ = ['get_kfac_layer', 'KNOWN_MODULES']

KNOWN_MODULES = {'Linear', 'Conv2d', 'Embedding', 'RNNBase', 'RNN', 'LSTM'}

def get_kfac_layer(module, use_eigen_decomp=True, damping=0.001,
                   factor_decay=0.95, batch_averaged=True):
    if isinstance(module, nn.Linear):
        layer = LinearLayer
    elif isinstance(module, nn.Conv2d):
        layer = Conv2dLayer
    #elif isinstance(module, nn.RNNBase):
    #    layer = RNNLayer
    #elif isinstance(module, nn.Embedding):
    #    layer = EmbeddingLayer
    else:
        raise NotImplementedError('KFAC does not support layer {}'.format(
                                  layer))

    return layer(module, use_eigen_decomp, damping, factor_decay,
                 batch_averaged)

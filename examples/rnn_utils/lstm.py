'''
LSTM Model for PyTorch's word_language_model example

Source: https://github.com/pytorch/examples/blob/master/word_language_model/model.py
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kfac.modules import LSTM

class LSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, 
                 batch_first=False, init_weight=0.05):
        super(LSTMModel, self).__init__()
        self.nhid = nhid
        self.nlayers = nlayers
        self.ntoken = ntoken
        self.batch_first = batch_first
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.layers = nn.ModuleList([
            LSTM(ninp, nhid, dropout=dropout, batch_first=batch_first)
            for i in range(nlayers)
        ])
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        for param in self.parameters():
            nn.init.uniform_(param, -init_weight, init_weight)

    def detach(self, hidden):
        return [(h.detach(), c.detach()) for (h, c) in hidden]

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(input.size(0 if self.batch_first else 1))
        x = self.drop(self.encoder(input))
        for i, layer in enumerate(self.layers):
            x, hidden[i] = layer(x, hidden[i])
            x = self.drop(x)
        decoded = self.decoder(x)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        w = next(self.parameters())
        return [(w.new_zeros(1, bsz, self.nhid), w.new_zeros(1, bsz, self.nhid))
                for i in range(self.nlayers)]

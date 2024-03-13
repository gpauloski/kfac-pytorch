"""Simple Transformer Model.

Based on Attention is All You Need and
https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class TransformerModel(nn.Module):
    """Transformer Model."""

    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
    ) -> None:
        """Init TransformerModel.

        Args:
            ntoken (int): number of tokens in vocabulary.
            d_model (int): number of expected features in encoder/decoder
                inputs.
            nhead (int): number of attention heads.
            d_hid (int): hidden dimension size.
            nlayers (int): number of encoder layers in the model.
            dropout (float): dropout layer probability.
        """
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            d_hid,
            dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            nlayers,
        )
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights."""
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Transformer forward pass.

        Args:
            src (Tensor): tensor with shape [seq_len, batch_size].
            src_mask (Tensor): tensor with shape [seq_len, seq_len].

        Returns:
            output tensor with shape [seq_len, batch_size, ntoken].
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):
    """Positional Encoder."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        """Init PositionalEncoding.

        Args:
            d_model (int): number of expected features in encoder/decoder
                inputs.
            dropout (float): dropout layer probability.
            max_len (int): max vocabulary size (I think).
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model),
        )
        self.pe: torch.Tensor
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Positional encoder forward pass.

        Args:
            x (Tensor): tensor with shape [seq_len, batch_size, embedding_dim].

        Returns:
            tensor with same shape as input injected with some information
                about the relative or absolute position of the tokens in the
                sequence.
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


def gen_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
from .rnn_cell import RNNCell, LSTMCell, LSTMImpl

def apply_permutation(tensor, permutation, dim=1):
    # type: (Tensor, Tensor, int) -> Tensor
    return tensor.index_select(dim, permutation)

class Bidirectional(nn.Module):
    """Bidirectional wrapper for LSTM/RNNImpl.
    https://stackoverflow.com/questions/56835100/adding-layers-and-bidirectionality-to-custom-lstm-cell-in-pytorch
    """
    def __init__(self, forward_cell, backward_cell=None):
        super().__init__()
        self.forward_cell = forward_cell
        self.backward_cell = backward_cell

    def __call__(self, x, hx=None):
        seq_dim = 1 if self.forward_cell.batch_first else 0
        h, c = hx
        x_forward, x_backward = torch.chunk(x, 2, 2)
        output, (h_t, c_t) = self.forward_cell(x_forward, (h[0], c[0]))
        output_back, (h_t_back, c_t_back) = self.backward_cell(x_backward, (h[1], c[1]))
        output = torch.cat((output, output_back), dim=-1)
        hx = (torch.stack([h_t, h_t_back]),
              torch.stack([c_t, c_t_back]))
        return output, hx

    def init_weights(self, init_weight):
        self.forward_cell.init_weights(init_weight)
        self.backward_cell.init_weights(init_weight)

class RNN(nn.Module):
    pass

class LSTM(nn.Module):
    """Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

    Uses custom LSTMCells to allow for KFAC support
    """

    def __init__(self,
                 input_size, 
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 dropout=0.0,
                 bidirectional=False,
                 init_weight=0.1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.init_weight = init_weight
        
        layers = []
        layers_reverse = []
        for i in range(num_layers):
            #input_size = self.num_directions * self.hidden_size if i > 0 else
            #        self.input_size
            layers.append(
                    LSTMImpl(input_size, hidden_size, batch_first=batch_first))
            if self.bidirectional:
                layers_reverse.append(
                        LSTMImpl(input_size, hidden_size, batch_first=batch_first))

        layers.extend([LSTMImpl(hidden_size, hidden_size, 
                num_directions=self.num_directions, batch_first=batch_first)
                for i in range(num_layers - 1)])

        if self.bidirectional:
            layers = [Bidirectional(l1, l2) for l1, l2 in zip(layers, layers_reverse)]

        self.layers = nn.ModuleList(layers)
        self.rnn_drops = nn.ModuleList(
                [nn.Dropout(dropout) for i in range(num_layers - 1)])

    def forward(self, input, hx=None):
        orig_input = input
        if isinstance(input, PackedSequence):
            batch_sizes = input.batch_sizes
            max_batch_size = batch_sizes[0]
            sorted_indices = input.sorted_indices
            unsorted_indices = input.unsorted_indices
            input, lens_unpacked = pad_packed_sequence(input, batch_first=self.batch_first)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        output, hidden = self._lstm_impl(input, hx)

        if isinstance(orig_input, PackedSequence):
            output_packed = pack_padded_sequence(output, lens_unpacked, batch_first=self.batch_first)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return output, self.permute_hidden(hidden, unsorted_indices)

    def permute_hidden(self, hx, permutation):
        # type: (Tuple[Tensor, Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor]
        if permutation is None:
            return hx
        return apply_permutation(hx[0], permutation), apply_permutation(hx[1], permutation)

    def _lstm_impl(self, input, hx):
        if self.num_directions > 1:
            h = hx[0].reshape(self.num_layers, self.num_directions, -1, self.hidden_size)
            c = hx[1].reshape(self.num_layers, self.num_directions, -1, self.hidden_size)
            seq_dim = 1 if self.batch_first else 0
            input = torch.cat([input, torch.flip(input, (seq_dim,))], dim=-1)
        else:
            h, c = hx
        h_n, c_n = [], []   # TODO(gpauloski): should we preallocate tensor?
        output = input
        for i, layer in enumerate(self.layers): 
            #print('layer', i, output.shape, h[i].shape)
            output, (h_i, c_i) = layer(output, (h[i], c[i]))
            h_n.append(h_i)
            c_n.append(c_i)
            if i + 1 < len(self.layers):
                 output = self.rnn_drops[i](output)
        return output, (torch.stack(h_n), torch.stack(c_n))


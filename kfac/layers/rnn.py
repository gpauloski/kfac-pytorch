import torch

from kfac.layers.base import KFACLayer

class RNNLayer(KFACLayer):
    """

    Note: Only works with RNNCellBase modules (e.g. RNNCell, LSTMCell).
          Does not work with torch.nn.RNN or torch.nn.LSTM.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_bias = self.module.bias
        self.num_weights = 2

        self._instantiate_inputs()
        self._instantiate_outputs()

    def get_diag_blocks(self, diag_blocks):
        return 1

    def get_gradients(self):
        grad_0 = self.module.weight_ih.grad.data
        grad_1 = self.module.weight_hh.grad.data
        bias_0 = self.module.bias_ih.grad.data
        bias_1 = self.module.bias_hh.grad.data
        
        if self.has_bias:
            grad_0 = torch.cat([grad_0, bias_0.view(-1, 1)], 1)
            grad_1 = torch.cat([grad_1, bias_1.view(-1, 1)], 1)
        return [grad_0, grad_1]

    def save_inputs(self, input):
        """Save inputs locally.

        Override kfac.layers.base.KFACLayer b/c `save_inputs()` will be called
        once for each timestep since RNNCells take as input only one timestep
        (unlike torch.nn.RNN that takes all timesteps as the input) and we want
        to accumulate them
        """
        x = input[0]
        hidden = input[1][0] if isinstance(input[1], tuple) else input[1]
        self.a_inputs[0].append(x.data)
        self.a_inputs[1].append(hidden.data)

    def save_grad_outputs(self, grad_output):
        if grad_output[0] is not None:
            self.g_outputs[0].append(grad_output[0].data)
        if grad_output[1] is not None:
            self.g_outputs[1].append(grad_output[1].data)

    def _compute_A_factors(self):
        """Compute A for x and hidden state

        Note: We only calculate A for the first time step (i.e. first sequence)

        Returns:
          tuple(A_x, A_hidden) where A_x is shape (input_size, input_size) 
          and A_hidden is shape (num_dir * hid_size, num_dir * hid_size)/
        """
        a_0 = self.a_inputs[0][-1]
        a_1 = self.a_inputs[1][-1]  # temp: just use last timestep

        self._instantiate_inputs()  # reset input timestep accumulation

        # at this point a_0 = (batch, input_size) and
        #               a_1 = (batch, hid_size)
        batch_size = a_0.size(0)

        a_0 = torch.cat([a_0, a_0.new(a_0.size(0), 1).fill_(1)], 1)
        a_1 = torch.cat([a_1, a_1.new(a_1.size(0), 1).fill_(1)], 1)

        a_0 = a_0.t() @ (a_0 / batch_size)
        a_1 = a_1.t() @ (a_1 / batch_size)
        return [a_0, a_1]

    def _compute_G_factors(self):
        """Compute G for x and hidden state

        Note: We only calculate G for the first time step (i.e. first sequence)

        Returns: 
          tuple(G_x, G_hidden) where G_x is shape (num_directions * hid_size)^2
          and G_hidden is shape (nlayers * hid_size)^2.
          TODO(gpauloski): is G_hidden shape right here?
        """
        g_0 = self.g_outputs[0][-1]
        g_1 = self.g_outputs[1][-1]  # temp: just use last timestep
        batch_size = g_0.size(0)

        self._instantiate_outputs()  # reset gradient timestep accumulation

        if self.batch_averaged:
            g_0 = g_0.t() @ (g_0 * batch_size)
            g_1 = g_1.t() @ (g_1 * batch_size)
        else:
            g_0 = g_0.t() @ (g_0 / batch_size)
            g_1 = g_1.t() @ (g_1 / batch_size)
        return [g_0, g_1]

    def _get_bias(self, i):
        if self.has_bias and i == 0:
            return self.module.bias_ih
        if self.has_bias and i == 1:
            return self.module.bias_hh0
        elif self.has_bias:
            raise ValueError('Invalid bias index {}. RNNBaseCell layer only has 1 '
                             'bias tensor'.format(i))
        else:
            return None

    def _get_weight(self, i):
        if i == 0:
            return self.module.weight_ih
        if i == 1:
            return self.module.weight_hh
        else:
            raise ValueError('Invalid weight index {}. RNNBaseCell layer only has '
                             '1 weight tensor'.format(i))

    def _instantiate_inputs(self):
        self.a_inputs = [[], []]

    def _instantiate_outputs(self):
        self.g_outputs = [[], []]

import torch

from kfac.layers.base import KFACLayer

class RNNLayer(KFACLayer):
    """
    Limitations:
      num_layers=1
      batch_first=True
      bidirectional=False
      bias=True
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(dir(self.module))
        for j, p in enumerate(self.module._flat_weights):
            print(j, p.shape)
        self.use_eigen_decomp = False
        self.has_bias = True

    def save_input(self, a):
        """Save input `a` locally

        For RNN, we have to save input and hidden state.
        """
        print('a', a[0].shape, a[1].shape)
        self.a_0 = a[0].data
        self.a_1 = a[1].data

    def save_grad_output(self, g):
        """Save grad w.r.t output `g` locally"""
        print('g', type(g), len(g))
        for i in range(len(g)):
            if g[i] is not None:
                print('output', i, '/', len(g), g[i].shape)
        self.g_0 = g[0].data
        # TODO(gpauloski): Find out how to get hidden state output
        self.g_1 = None

    def _init_buffers_A(self, A_0, A_1):
        """Create buffers for factors A_0, A_1 and its inv"""
        self.A_0_factor = A_0.new(A_0.shape).fill_(1)
        self.A_1_factor = A_1.new(A_1.shape).fill_(1)
        if self.use_eigen_decomp:
            raise NotImplementedError('Eigen decomp not supported yet')
        else:
            A_0_shape = A_0.shape
        self.A_0_inv = A_0.new_zeros(A_0_shape)
        self.A_1_inv = A_1.new_zeros(A_1_shape)

    def _init_buffers_G(self, G_0, G_1):
        """Create buffers for factors A_0, G_0 and its inv"""
        self.G_0_factor = G_0.new(G_0.shape).fill_(1)
        self.G_1_factor = G_1.new(G_1.shape).fill_(1)
        if self.use_eigen_decomp:
            raise NotImplementedError('Eigen decomp not supported yet')
        else:
            G_0_shape = G_0.shape
        self.G_0_inv = G_0.new_zeros(G_0_shape)
        self.G_1_inv = G_1.new_zeros(G_1_shape)

    def get_gradients(self):
        """Get formated gradient of module
        Returns:
          Formatted gradient with shape [ouput_dim, input_dim] for module
        """
        grad_0 = self.module.weight_ih_l0
        grad_1 = self.module.weight_hh_l0
        bias_0 = self.module.bias_ih_l0
        bias_1 = self.module.bias_hh_l0
        
        grad_0 = torch.cat([grad_0, bias_0.view(-1, 1)], 1)
        grad_1 = torch.cat([grad_1, bias_1.view(-1, 1)], 1)
        return (grad_0, grad_1)

    def get_diag_blocks(self, diag_blocks):
        return 1

    def _compute_A(self):
        """Compute A for x and hidden state

        Note: We only calculate A for the first time step (i.e. first sequence)

        Returns:
          tuple(A_x, A_hidden) where A_x is shape (input_size, input_size) 
          and A_hidden is shape (num_dir * hid_size, num_dir * hid_size)/
        """
        a_0 = self.a_0[:, 0]
        a_1 = self.a_1[0]  # nlayers is first dim and we only support
                                     # nlayers = 1 right now

        # at this point a_0 = (batch, input_size) and
        #               a_1 = (batch, hid_size)
        batch_size = a_0.size(0)

        a_0 = torch.cat([a_0, a_0.new(a_0.size(0), 1).fill_(1)], 1)
        a_1 = torch.cat([a_1, a_1.new(a_1.size(0), 1).fill_(1)], 1)

        a_0 = a_0.t() @ (a_0 / batch_size)
        a_1 = a_1.t() @ (a_1 / batch_size)
        return (a_0, a_1)

    def _compute_G(self):
        """Compute G for x and hidden state

        Note: We only calculate G for the first time step (i.e. first sequence)

        Returns: 
          tuple(G_x, G_hidden) where G_x is shape (num_directions * hid_size)^2
          and G_hidden is shape (nlayers * hid_size)^2.
          TODO(gpauloski): is G_hidden shape right here?
        """
        g_0 = self.g_0[:, 0]
        g_1 = self.g_1  # TODO(gpauloski) this is None right now
        batch_size = g_0.size(0)

        if g_1 is None:
            g_1 = g_0.new((batch_size, g_0.shape[1]))

        if self.batch_averaged:
            g_0 = g_0.t() @ (g_0 * batch_size)
            g_1 = g_1.t() @ (g_1 * batch_size)
        else:
            g_0 = g_0.t() @ (g_0 / batch_size)
            g_1 = g_1.t() @ (g_1 / batch_size)
        return (g_0, g_1)

    def update_A(self):
        """Compute factor A_0, A_1 and add to running average"""
        A_0, G_1 = self._compute_A()
        if self.A_0_factor is None or self.A_1_factor:
            self._init_buffers_A(A_0, A_1)
        update_running_avg(A_0, self.A_0_factor, self.factor_decay)
        update_running_avg(A_1, self.A_1_factor, self.factor_decay)

    def update_G(self):
        """Compute factor G_0, G_1 and add to running average"""
        G_0, G_1 = self._compute_G()
        if self.G_0_factor is None or self.G_1_factor:
            self._init_buffers_G(G_0, G_1)
        update_running_avg(G_0, self.G_0_factor, self.factor_decay)
        update_running_avg(G_1, self.G_1_factor, self.factor_decay)

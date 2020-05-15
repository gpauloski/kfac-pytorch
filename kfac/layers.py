import math
import torch
import horovod.torch as hvd

from kfac.utils import update_running_avg
from kfac.utils import try_contiguous
from kfac.utils import get_block_boundary

KNOWN_MODULES = {'Linear', 'Conv2d', 'Embedding'}

def get_kfac_layer(module, use_eigen_decomp=True, damping=0.001,
                   factor_decay=0.95, batch_averaged=True):
    if isinstance(module, torch.nn.Linear):
        layer = LinearLayer
    elif isinstance(module, torch.nn.Conv2d):
        layer = Conv2dLayer
    elif isinstance(module, torch.nn.Embedding):
        layer = EmbeddingLayer
    else:
        raise NotImplementedError('KFAC does not support layer {}'.format(
                                  layer))
    return layer(module, use_eigen_decomp, damping, factor_decay,
                 batch_averaged)


class KFACLayer(object):
    def __init__(self,
                 module,
                 use_eigen_decomp=True,
                 damping = 0.001,
                 factor_decay=0.95,
                 batch_averaged=True):
        self.module = module
        self.use_eigen_decomp=use_eigen_decomp
        self.damping = damping
        self.factor_decay = factor_decay
        self.batch_averaged = batch_averaged
        self.eps = 1e-10

        self.a_input = None
        self.g_output = None
        self.A_factor = None
        self.G_factor = None

    def save_input(self, a):
        """Save input `a` locally"""
        self.a_input = a[0].data

    def save_grad_output(self, g):
        """Save grad w.r.t output `g` locally"""
        self.g_output = g[0].data

    def _init_buffers_A(self, factor):
        """Create buffers for factor A and its inv"""
        self.A_factor = torch.diag(factor.new(factor.shape[0]).fill_(1))
        if self.use_eigen_decomp:
            # add one axtra column for eigenvalues
            shape = (factor.shape[0], factor.shape[0] + 1)
        else:
            shape = factor.shape
        self.A_inv = factor.new_zeros(shape)

    def _init_buffers_G(self, factor):
        """Create buffers for factor G and its inv"""
        self.G_factor = torch.diag(factor.new(factor.shape[0]).fill_(1))
        if self.use_eigen_decomp:
            # add one axtra column for eigenvalues
            shape = (factor.shape[0], factor.shape[0] + 1)
        else:
            shape = factor.shape
        self.G_inv = factor.new_zeros(shape)

    def clear_inverse(self):
        """Clear inverse buffers

        Useful for when switching between `diag_blocks=1` and `diag-blocks>1`
        because eigendecompositions saved in place and the off-diagonals must
        be cleared.
        """
        if self.A_inv is not None:
            self.A_inv.fill_(0)
        if self.G_inv is not None:
            self.G_inv.fill_(0)

    def get_gradient(self):
        """Get formated gradient of module

        Args:
          module: module/layer to get gradient of

        Returns:
          Formatted gradient with shape [ouput_dim, input_dim] for module
        """
        grad = self.module.weight.grad.data
        if self.has_bias:
            grad = torch.cat([grad, self.module.bias.grad.data.view(-1, 1)], 1)
        return grad

    def _precondition_gradient_inv(self):
        """Compute preconditioned gradient for inverse method"""
        grad = self.get_gradient() 
        return self.G_inv @ grad @ self.A_inv
    
    def _precondition_gradient_eigen(self):
        """Compute preconditioned gradient for eigendecomp method"""
        QA, QG = self.A_inv[:,:-1], self.G_inv[:,:-1]
        dA, dG = self.A_inv[:,-1], self.G_inv[:,-1]
        grad = self.get_gradient()
        v1 = QG.t() @ grad @ QA
        v2 = v1 / (dG.unsqueeze(1) * dA.unsqueeze(0) + self.damping)
        return QG @ v2 @ QA.t()

    def get_preconditioned_gradient(self):
        """Precondition gradient of module

        Args:
          module: module to compute preconditioned gradient for
          grad: formatted gradient from `_get_grad()`

        Returns:
          preconditioned gradient with shape [output_dim, input_dim]
        """
        if self.use_eigen_decomp:
            v = self._precondition_gradient_eigen()
        else:
            v = self._precondition_gradient_inv()

        if self.has_bias:
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(self.module.weight.grad.data.size())
            v[1] = v[1].view(self.module.bias.grad.data.size())
        else:
            v = [v.view(self.module.weight.grad.data.size())]
        return v

    def _compute_A(self):
        """Compute factor A"""
        raise NotImplementedError

    def _compute_G(self):
        """Compute factor G"""
        raise NotImplementedError

    def update_A(self):
        """Compute factor A and add to running average"""
        A = self._compute_A()
        if self.A_factor is None:
            self._init_buffers_A(A)
        update_running_avg(A, self.A_factor, self.factor_decay)

    def update_G(self):
        """Compute factor G and add to running average"""
        G = self._compute_G()
        if self.G_factor is None:
            self._init_buffers_G(G)
        update_running_avg(G, self.G_factor, self.factor_decay)

    def _get_block_inv(self, block):
        """Compute inverse of tensor"""
        diag = block.new(block.shape[0]).fill_(self.damping)
        cholesky = torch.cholesky(block + torch.diag(diag))
        return torch.cholesky_inverse(cholesky)

    def _get_block_eigen(self, block):
        """Compute eigendecomposition of tensor. Append eigenvalues"""
        d, Q = torch.symeig(block, eigenvectors=True)
        d = torch.mul(d, (d > self.eps).float())
        d = d.unsqueeze(1)
        return torch.cat([Q, d], 1)

    def get_diag_blocks(self, diag_blocks):
        """Helper method for determining number of diag_blocks to use

        Overrides `diag_blocks` if the `module` does not support
        `diag_blocks>1`.
        """
        return 1

    def _distributed_factor_inv(self, factor, inv, ranks):
        """Computes the inv/eigendecomp of a factor across ranks

        Assigns each rank in `ranks` to enter this function to compute a
        diagonal block of `factor`. If `len(ranks)==1`, then that rank 
        computes the inv/eigendecomp of the entire `factor`.

        Args:
            factor (tensor): tensor to eigendecompose
            inv (tensor): tensor to save inplace inverse to
            ranks (list): list of ranks that will enter this function
        """
        i = ranks.index(hvd.rank())
        n = len(ranks)
        if n > min(factor.shape):
            n = min(factor.shape)

        if i < n:
            start, end = get_block_boundary(i, n, factor.shape)
            block = factor[start[0]:end[0], start[1]:end[1]]
            if self.use_eigen_decomp:
                block_inv = self._get_block_eigen(block)
                inv.data[start[0]:end[0], -1].copy_(block_inv[:,-1])
                inv.data[start[0]:end[0], start[1]:end[1]].copy_(
                        block_inv[:,:-1])
            else:
                block_inv = self._get_block_inv(block)
                inv.data[start[0]:end[0], start[1]:end[1]].copy_(block_inv)

    def compute_A_inv(self, ranks):
        """Compute eigendecomp/inv of A for module on specified workers

        Note: all ranks will enter this function but only the ranks specified
        in `ranks` will continue to actually compute the eigendecomp/inv.
        All other ranks will simply zero out their buffer for the
        eigendecomp/inv for the current module. This is done so we can sum
        the eigendecomp/inv across all ranks to communicate the results
        of locally computed eigendecomp/inv.

        Args:
          ranks: list of horovod ranks (i.e. workers) to use.
        """
        if hvd.rank() in ranks:
            self._distributed_factor_inv(self.A_factor, self.A_inv, ranks)
        else:
            self.A_inv.fill_(0)

    def compute_G_inv(self, ranks):
        """Compute eigendecomp/inv of A for module on specified workers

        See `compute_A_inv` for more info`
        """
        if hvd.rank() in ranks:
            self._distributed_factor_inv(self.G_factor, self.G_inv, ranks)
        else:
            self.G_inv.fill_(0)

    def get_factor_handles(self, op=hvd.Average):
        """Get list of handles to call to average factors"""
        return [hvd.allreduce_async_(self.A_factor, op=op),
                hvd.allreduce_async_(self.G_factor, op=op)]

    def get_inverse_handles(self, op=hvd.Sum):
        """Get list of handles to call to sum inverse"""
        return [hvd.allreduce_async_(self.A_inv, op=op),
                hvd.allreduce_async_(self.G_inv, op=op)]


class LinearLayer(KFACLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_bias = True

    def get_diag_blocks(self, diag_blocks):
        return diag_blocks

    def _compute_A(self):
        # a: batch_size * in_dim
        a = self.a_input
        batch_size = a.size(0)
        if len(a.shape) > 2:
            a = torch.mean(a, list(range(len(a.shape)))[1:-1])
        if self.module.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        return a.t() @ (a / batch_size)

    def _compute_G(self):       
        # g: batch_size * out_dim
        g = self.g_output
        batch_size = g.size(0)
        if len(g.shape) > 2:
            g = torch.mean(g, list(range(len(g.shape)))[1:-1])
        if self.batch_averaged:
            return g.t() @ (g * batch_size)
        else:
            return g.t() @ (g / batch_size)


class Conv2dLayer(KFACLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_bias = self.module.bias is not None

    def get_gradient(self):
        grad = self.module.weight.grad.data.view(
                self.module.weight.grad.data.size(0), -1)  
        if self.module.bias is not None:
            grad = torch.cat([grad, module.bias.grad.data.view(-1, 1)], 1)
        return grad

    def get_diag_blocks(self, diag_blocks):
        return diag_blocks

    # TODO: refactor extract_params to not reuire x arg
    def _extract_patches(self, x):
        """Extract patches from convolutional layer

        Args:
          x: The input feature maps.  (batch_size, in_c, h, w)
    
        Returns:
          Tensor of shape (batch_size, out_h, out_w, in_c*kh*kw)
        """
        padding = self.module.padding
        kernel_size = self.module.kernel_size
        stride = self.module.stride
        if padding[0] + padding[1] > 0:
            x = torch.nn.functional.pad(x, 
                    (padding[1], padding[1], padding[0], padding[0])).data
        x = x.unfold(2, kernel_size[0], stride[0])
        x = x.unfold(3, kernel_size[1], stride[1])
        x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
        x = x.view(
            x.size(0), x.size(1), x.size(2),
            x.size(3) * x.size(4) * x.size(5))
        return x

    def _compute_A(self):
        a = self.a_input
        batch_size = a.size(0)
        a = self._extract_patches(a)
        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        if self.module.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        a = a/spatial_size
        return a.t() @ (a / batch_size)

    def _compute_G(self):
        # g: batch_size * n_filters * out_h * out_w
        # n_filters is actually the output dimension (analogous to Linear layer)
        g = self.g_output
        spatial_size = g.size(2) * g.size(3)
        batch_size = g.shape[0]
        g = g.transpose(1, 2).transpose(2, 3)
        g = try_contiguous(g)
        g = g.view(-1, g.size(-1))

        if self.batch_averaged:
            g = g * batch_size
        g = g * spatial_size

        return g.t() @ (g / g.size(0))


class EmbeddingLayer(KFACLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_bias = False
        self.use_eigen_decomp = False

    def _compute_A(self):
        """Compute A for Embedding layer

        Input to Embedding layer is (batch_size, input_size) representing
        indicies into the embedding matrix of size [vocab_size, embed_size].
        The factor is represented by:
          (1/batch_size) sum_{i} diag(one_hot(inputs[i]) ** 2)
        where inputs[i] is the input for the ith batch and the output is size
        [vocab_size, vocab_size]

        source: https://github.com/tensorflow/kfac/blob/master/kfac/python/ops/fisher_factors.py#L1107
        """
        a = self.a_input
        batch_size = a.size(0)
        tokens = torch.LongTensor(batch_size, self.module.num_embeddings)
        tokens.zero_()
        tokens.scatter_(1, a.cpu(), 1) # scatter seems to require cpu backend
        tokens = torch.mean(tokens.float(), dim=0)
        return torch.diag(tokens)

    def _compute_G(self):       
        # g: batch_size * out_dim
        g = self.g_output
        batch_size = g.size(0)
        if len(g.shape) > 2:
            g = torch.mean(g, list(range(len(g.shape)))[1:-1])
        if self.batch_averaged:
            return g.t() @ (g * batch_size)
        else:
            return g.t() @ (g / batch_size)
    
    def _get_diag_block_inv(self, block):
        """Compute inverse of diagonal tensor"""
        return block.pow(-1)

    def _get_block_eigen(self, block):
        """Compute eigendecomposition of tensor. Append eigenvalues"""
        raise NotImplementedError('Use inv method for embedding layers')

    def _precondition_gradient_inv(self):
        """Compute preconditioned gradient for inverse method

        We compute on the CPU because the size of A is very large (order 
        ntokens^2) and then move back to GPU
        """
        grad = self.get_gradient()
        v1 = torch.matmul(self.G_inv, grad.t())
        v1 = v1.cpu()
        v2 = torch.matmul(v1, self.A_inv).cuda().t()
        return v2
    
    def _precondition_gradient_eigen(self):
        """Compute preconditioned gradient for eigendecomp method"""
        raise NotImplementedError('Use inv method for embedding layers')
    
    def compute_A_inv(self, ranks):
        """Compute inv of A for module on specified workers

        Note: Embedding layer currently ignores all but first rank

        Args:
          ranks: list of horovod ranks (i.e. workers) to use.
        """
        if hvd.rank() == ranks[0]:
            self.A_inv.copy_(self._get_diag_block_inv(self.A_factor))
        else:
            self.A_inv.fill_(0)

    def compute_G_inv(self, ranks):
        """Compute inv of G for module on specified workers

        See `compute_A_inv` for more info`
        """
        if hvd.rank() == ranks[0]:
            self.G_inv.copy_(self._get_block_inv(self.G_factor))
        else:
            self.G_inv.fill_(0)


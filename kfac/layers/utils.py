import torch


def append_bias_ones(tensor):
    """Appends vector of ones to last dimension of tensor.

    For examples, if the input is of shape [4, 6], then the outputs has shape
    [4, 7] where the slice [:, -1] is a tensor of all ones.
    """
    shape = list(tensor.shape[:-1]) + [1]
    return torch.cat([tensor, tensor.new_ones(shape)], dim=-1)

def get_cov(a, b=None, scale=None):
    """Computes the empirical second moment of a 2D tensor

    Reference:
      - https://github.com/tensorflow/kfac/blob/master/kfac/python/ops/fisher_factors.py#L220
      - https://arxiv.org/pdf/1602.01407.pdf#subsection.2.2

    Args:
      a (tensor): 2D tensor to compute second moment of using cov_a = a^T @ a.
      b (tensor, optional): optional tensor of equal shape to a such that
          cov_a = a^T @ b.
      scale (float, optional): optional tensor to divide cov_a by. Default is
          a.size(0).

    Returns:
      A square tensor representing the second moment of a.
    """
    if len(a.shape) != 2:
        raise ValueError('Input tensor must have 2 dimensions.')
    if b is not None and a.shape != b.shape:
        raise ValueError('Input tensors must have same shape. Got tensors of '
                         'shape {} and {}.'.format(a.shape, b.shape))

    if scale is None:
        scale = a.size(0)

    if b is None:
        cov_a = a.t() @ (a / scale)
        return (cov_a + cov_a.t()) / 2.0
    else:
        return a.t() @ (b / scale)

def get_eigendecomp(tensor, clip=0.0, concat=True, symmetric=True):
    """Compute eigendecomposition of a block.

    Args:
      tensor: tensor of shape (x, x) to eigendecompose.
      clip (float, optional): value to clip eigenvalues using 
          `torch.max(eigenvalues, clip)` such that the minimum eigenvalue is
          `clip`. If `None`, no clipping is applied. (default: 0.0)
      concat (bool, optional): concatenate eigenvalues to the last dim
          of the eigenvectors. (default: False)
      symmetric (bool, optional): is `tensor` symmetric. (default: False)

    Returns:
      Tensor of shape (x, x+1) where (0:x, 0:x) are the eigenvectors and
      (:, -1) is the eigenvalues if `concat=True` else `tuple(eigenvectors,
      eigenvalues)`.
    """
    if symmetric:
        d, Q = torch.symeig(tensor, eigenvectors=True)
    else:
        d, Q = torch.eig(tensor, eigenvectors=True)
        d = d[:, 0]

    if clip is not None:
        d = torch.max(d, d.new_tensor([clip]))

    if concat:
       return torch.cat([Q, d.unsqueeze(-1)], -1)
    else:
       return Q, d

def get_inverse(tensor, damping=None, symmetric=True):
    """Compute inverse of tensor.

    Args:
      tensor: block of shape (x, x) to invert
      damping (float, optional): optionally add `identity * damping` to 
          `tensor` before inverting.
      symmetric (bool, optional): if True, `tensor` is symmetric and Cholesky
          decomposition will be used for computing the inverse. (default: True)

    Returns:
      The inverse of tensor
    """
    if damping is not None:
        d = tensor.new(tensor.shape[0]).fill_(damping)
        tensor = tensor + torch.diag(d)

    if symmetric:
        return torch.cholesky_inverse(torch.cholesky(tensor))
    else:
        return torch.inverse(tensor)

def get_elementwise_inverse(vector, damping=None):
    """Computes the reciprocal of each non-zero element of v"""
    if damping is not None:
        vector = vector + damping
    mask = vector != 0.0
    reciprocal = vector.clone()
    reciprocal[mask] = torch.reciprocal(reciprocal[mask])
    return reciprocal

def reshape_data(data_list, batch_first=True, collapse_dims=False):
    """Concat input/output data and clear buffers

    Args:
      data_list (list): list of tensors of equal, arbitrary shape where the
          batch_dim is either 0 or 1 depending on self.batch_first.
      batch_first (bool, optional): is batch dim first. (default: True)
      collapse_dim (bool, optional): if True, collapse all but the last dim
          together forming a 2D output tensor.

    Returns:
      Single tensor with all tensors from data_list concatenated across
      batch_dim. Guarenteed to be 2D if collapse_dims=True.
    """
    d = torch.cat(data_list, dim=int(not batch_first))
    if collapse_dims and len(d.shape) > 2:
        d = d.view(-1, d.shape[-1])
    return d

def get_triu(tensor):
    """Returns flattened upper triangle of 2D tensor"""
    if len(tensor.shape) != 2:
        raise ValueError('triu(tensor) requires tensor to be 2 dimensional')
    if tensor.shape[0] > tensor.shape[1]:
        raise ValueError('tensor cannot have more rows than columns')
    idxs = torch.triu_indices(tensor.shape[0], tensor.shape[1],
                              device=tensor.device)
    return tensor[idxs[0], idxs[1]]

def fill_triu(shape, triu_tensor):
    """Reconstruct symmetric 2D tensor from flattened upper triangle

    Usage:
      >>> x = tensor.new_empty([10, 10])
      >>> triu_x = get_triu(x)
      >>> x_new = fill_triu([10, 10], triu_tensor)
      >>> assert torch.equal(x, x_new)  # true

    Args:
      shape (tuple): tuple(rows, cols) of size of output tensor
      triu_tensor (tensor): flattened upper triangle of the tensor returned by
          get_triu()

    Returns:
      Symmetric tensor with `shape` where the upper/lower triangles are filled
          with the data in `triu_tensor`
    """
    if len(shape) != 2:
        raise ValueError('shape must be 2 dimensional') 
    rows, cols = shape
    dst_tensor = triu_tensor.new_empty(shape)
    idxs = torch.triu_indices(rows, cols, device=triu_tensor.device)
    dst_tensor[idxs[0], idxs[1]] = triu_tensor
    idxs = torch.triu_indices(rows, rows, 1, device=dst_tensor.device)
    dst_tensor.transpose(0, 1)[idxs[0], idxs[1]] = dst_tensor[idxs[0], idxs[1]]
    return dst_tensor

def update_running_avg(new, current, alpha=1.0):
    """Computes in-place running average

    current = alpha*current + (1-alpha)*new

    Args:
      new (tensor): tensor to add to current average
      current (tensor): tensor containing current average. Result will be
          saved in place to this tensor.
      alpha (float, optional): (default: 1.0)
    """
    if alpha != 1:
        current *= alpha / (1 - alpha)
        current += new
        current *= (1 - alpha)

import math
import torch
import warnings

import kfac

from . import utils
from .. import comm

class KFACLayer(object):
    def __init__(self,
                 module,
                 accumulate_data=True,
                 batch_first=True,
                 inv_dtype=None,
                 grad_scaler=None,
                 factor_dtype=None,
                 prediv_eigenvalues=True,
                 symmetry_aware_comm=False,
                 use_eigen_decomp=True):
        self.module = module
        self.accumulate_data = accumulate_data
        self.batch_first = batch_first
        self.grad_scaler = grad_scaler
        self.inv_dtype = inv_dtype
        self.factor_dtype = factor_dtype
        self.prediv_eigenvalues = prediv_eigenvalues
        self.symmetry_aware_comm=symmetry_aware_comm
        self.use_eigen_decomp=use_eigen_decomp
        self.eps = 1e-10

        # Should be overridden by implementing class
        self.has_bias = False
        self.factors_are_symmetric = True

        self.a_inputs  = []  # inputs accumulated from module hooks
        self.g_outputs = []  # outputs accumulated from module hooks
        self.state = {
            'A': None,
            'G': None,
        }
        self.preconditioned_gradient = None

    def __repr__(self):
        return 'KFAC {}({})'.format(self.__class__.__name__, repr(self.module))

    def state_dict(self, include_inverses=False):
        """Returns the state of the KFACLayer as a dictionary.

        Used by kfac.KFAC for state saving/loading. Note by default only the
        factors are saved because the inverses can be recomputed from the
        factors, however, the `include_inverses` flag can override this. If
        `keep_inv_copy=False` and `compute_A_inv_rank != compute_G_inv_rank != get_rank()` then
        the inverses may be `None` because this worker is not responsible for
        this layer.
        """
        if include_inverses:
            return self.state
        else:
            return {'A': self.state['A'], 'G': self.state['G']}

    def load_state_dict(self, state_dict):
        """Loads the KFACLayer state."""
        if 'A' not in state_dict or 'G' not in state_dict:
            # Backwards compatibility for old state dict keys
            if 'A_factor' in state_dict and 'G_factor' in state_dict:
                state_dict['A'] = state_dict.pop('A_factor')
                state_dict['G'] = state_dict.pop('G_factor')
            else:
                raise KeyError('KFACLayer state_dict must contain keys '
                               '"A" and "G"')
        device = next(self.module.parameters()).device
        self.state = state_dict
        for key in self.state:
            self.state[key] = self.state[key].to(device)

    def assign_inverse_workers(self, compute_A_inv_rank, compute_G_inv_rank,
        broadcast_A_inv_group, broadcast_G_inv_group):
        """Assign ranks to compute inverses

        Args:
          compute_A_inv_rank (int): rank to compute A inverse on
          compute_G_inv_rank (int): rank to compute G inverse on
          broadcast_A_inv_group (BroadcastGroup): broadcast group for A inv
          broadcast_G_inv_group (BroadcastGroup): broadcast group for G inv
        """
        if compute_A_inv_rank != compute_G_inv_rank and self.prediv_eigenvalues:
            raise ValueError('When precomputing 1 / (dG * dA.T + damping), '
                             'A and G inverse worker ranks must be equal. '
                             'I.e. distribute_layer_factors=False.')
        self.compute_A_inv_rank = compute_A_inv_rank
        self.compute_G_inv_rank = compute_G_inv_rank
        self.broadcast_A_inv_group = broadcast_A_inv_group
        self.broadcast_G_inv_group = broadcast_G_inv_group

    def assign_gradient_workers(self, compute_grad_ranks,
            broadcast_grad_groups):
        """Assign ranks to compute preconditioned gradients
        
        Args:
          compute_grad_ranks (list(int)): ranks to compute grads on
          broadcast_grad_groups (list(tuple(int, BroadcastGroup))): list where
              the indices are the ranks in the world and the values are tuples
              (src, group) indicating the src rank and broadcast group to use
              when broadcasting the gradients.
        """
        if len(broadcast_grad_groups) != comm.backend.size():
            raise ValueError('len(broadcast_grad_groups) != world size')

        self.compute_grad_ranks = compute_grad_ranks
        self.broadcast_grad_groups = broadcast_grad_groups
        self.keep_inv_copy = comm.backend.rank() in self.compute_grad_ranks

    def allreduce_factors(self):
        """Allreduce A and G factors

        Returns:
          list of async work handles
        """
        if self.factors_are_symmetric and self.symmetry_aware_comm:
            # Only broadcast upper triangle
            self.state['A_flat'] = utils.get_triu(self.state['A'])
            self.state['G_flat'] = utils.get_triu(self.state['G'])
            return [comm.backend.allreduce(self.state['A_flat']),
                    comm.backend.allreduce(self.state['G_flat'])]
        return [comm.backend.allreduce(self.state['A']),
                comm.backend.allreduce(self.state['G'])]

    def broadcast_inverses(self):
        """Broadcast A and G inverses

        Note: all ranks enter this function but some ranks may not be in the
          broadcast group for the inverses. comm.backend.broadcast() will be a
          no-op if a group is provided in rank is not in the group.

        Returns:
          list of async work handles
        """
        if not self.keep_inv_copy:
            return []

        if self.use_eigen_decomp:
            ops = [comm.backend.broadcast(self.state['QA'],
                           src=self.compute_A_inv_rank,
                           group=self.broadcast_A_inv_group),
                   comm.backend.broadcast(self.state['QG'], 
                           src=self.compute_G_inv_rank,
                           group=self.broadcast_G_inv_group)]
            if self.prediv_eigenvalues:
                ops.append(comm.backend.broadcast(self.state['dGdA'], 
                        src=self.compute_A_inv_rank, 
                        group=self.broadcast_A_inv_group))
            else:
                ops.append(comm.backend.broadcast(self.state['dA'], 
                        src=self.compute_A_inv_rank, 
                        group=self.broadcast_A_inv_group))
                ops.append(comm.backend.broadcast(self.state['dG'], 
                        src=self.compute_G_inv_rank, 
                        group=self.broadcast_G_inv_group))
            return ops

        if self.factors_are_symmetric and self.symmetry_aware_comm:
            # Only broadcast upper triangle
            self.state['A_inv'] = utils.get_triu(self.state['A_inv'])
            self.state['G_inv'] = utils.get_triu(self.state['G_inv'])
        return [comm.backend.broadcast(self.state['A_inv'], 
                        src=self.compute_A_inv_rank,
                        group=self.broadcast_A_inv_group),
                comm.backend.broadcast(self.state['G_inv'], 
                        src=self.compute_G_inv_rank,
                        group=self.broadcast_G_inv_group)]

    def broadcast_gradient(self):
        """Broadcast preconditioned gradient

        Returns:
          list of async work handles
        """
        if self.compute_grad_ranks is None:
            raise ValueError('Gradient compute ranks have not been assigned '
                             'yet. Use assign_workers().')
        # If the preconditioned gradient is None, initialize it so
        # we can correctly perform the broadcast op between ranks
        if self.preconditioned_gradient is None:
            w = self._get_weight_grad()
            self.preconditioned_gradient = [w.new_zeros(w.shape)]
            if self.has_bias:
                b = self._get_bias_grad()
                self.preconditioned_gradient.append(b.new_zeros(b.shape))

        self.preconditioned_gradient = [t.contiguous() for t in
                self.preconditioned_gradient]
        
        src, group = self.broadcast_grad_groups[comm.backend.rank()]
        return [comm.backend.broadcast(tensor, src=src, group=group)
                for tensor in self.preconditioned_gradient]

    def compute_A_inv(self, damping=0.001, ignore_rank=False):
        """Compute A inverse on assigned rank

        Note: 
          - all ranks will enter this function but only the ranks assigned
            to this layer will continue to actually compute the inverses.
            All other ranks will simply zero out their inverse buffers for.
            This is done so we can sum the inverses across all ranks to 
            communicate the results of locally computed inverses.
          - tensors for storing the inverse will be initialized based on the
            shape of the factor if the inv is None. This means that
            self.update_A_factor() must be called at least once before this
            function.

        Args:
          damping (float, optional): damping value to condition inverse 
             (default: 0.001)
          ignore_rank (bool, optional): ignore assigned rank and compute
             inverse (default: False)
        """
        if self.compute_A_inv_rank is None:
            raise ValueError('Workers have not been assigned to layer yet.')
        if self.keep_inv_copy is None:
            raise ValueError('Grad workers have not been assigned to layer yet.')
        if self.state['A'] is None:
            raise RuntimeError('update_A_factor() must be called at least '
                               'once before calling compute_A_inv().')
        
        if self.factors_are_symmetric and self.symmetry_aware_comm:
            # Reconstruct factor if it was flattened for communication
            if 'A_flat' in self.state:
                self.state['A'] = utils.fill_triu(
                        self.state['A'].shape, self.state['A_flat'])
                del self.state['A_flat']

        # Init inv buffer for ranks that will receive the inverse
        if self.keep_inv_copy:
            if self.use_eigen_decomp and 'QA' not in self.state:
                self.state['QA'] = torch.empty_like(
                        self.state['A'], dtype=self.inv_dtype)
                if self.prediv_eigenvalues and 'dGdA' not in self.state:
                    self.state['dGdA'] = self.state['A'].new_empty(
                            (self.state['G'].shape[0], self.state['A'].shape[0]),
                            dtype=self.inv_dtype)
                elif not self.prediv_eigenvalues and 'dA' not in self.state:
                    self.state['dA'] = self.state['A'].new_empty(
                            self.state['A'].shape[0], dtype=self.inv_dtype)
            elif not self.use_eigen_decomp and 'A_inv' not in self.state:
                self.state['A_inv'] = torch.empty_like(
                        self.state['A'], dtype=self.inv_dtype)

        if ignore_rank or comm.backend.rank() == self.compute_A_inv_rank:
            results = self._compute_factor_inverse(self.state['A'], damping)

            if isinstance(results, tuple):
                self.state['QA'] = results[0]
                self.state['dA'] = results[1]
            else:
                self.state['A_inv'] = results

    def compute_G_inv(self, damping=0.001, ignore_rank=False):
        """Compute G inverse on specified ranks

        See `compute_A_inv` for more info`
        """
        if self.compute_G_inv_rank is None:
            raise ValueError('Workers have not been assigned to layer yet.')
        if self.keep_inv_copy is None:
            raise ValueError('Grad workers have not been assigned to layer yet.')
        if self.state['G'] is None:
            raise RuntimeError('update_G_factor() must be called at least '
                               'once before calling compute_G_inv().')
        
        if self.factors_are_symmetric and self.symmetry_aware_comm:
            # Reconstruct factor if it was flattened for communication
            if 'G_flat' in self.state:
                self.state['G'] = utils.fill_triu(
                        self.state['G'].shape, self.state['G_flat'])
                del self.state['G_flat']

        # Init inv buffer for ranks that will receive the inverse
        if self.keep_inv_copy:
            if self.use_eigen_decomp and 'QG' not in self.state:
                self.state['QG'] = torch.empty_like(
                        self.state['G'], dtype=self.inv_dtype)
                if self.prediv_eigenvalues and 'dGdA' not in self.state:
                    self.state['dGdA'] = self.state['A'].new_empty(
                            (self.state['G'].shape[0], self.state['A'].shape[0]),
                            dtype=self.inv_dtype)
                elif not self.prediv_eigenvalues and 'dG' not in self.state:
                    self.state['dG'] = self.state['G'].new_empty(
                            self.state['G'].shape[0], dtype=self.inv_dtype)
            elif not self.use_eigen_decomp and 'G_inv' not in self.state:
                self.state['G_inv'] = torch.empty_like(
                        self.state['G'], dtype=self.inv_dtype)

        if ignore_rank or comm.backend.rank() == self.compute_G_inv_rank:
            results = self._compute_factor_inverse(self.state['G'], damping)

            if isinstance(results, tuple):
                self.state['QG'] = results[0]
                self.state['dG'] = results[1]
                if self.prediv_eigenvalues:
                    if 'dA' not in self.state:
                        raise ValueError('compute_A_inv must be called before '
                                         'compute_G_inv if prediv_eigenvalues '
                                         'is True.')
                    self.state['dGdA'] = 1 / (self.state['dG'].unsqueeze(1) *
                            self.state['dA'].unsqueeze(0) + damping)
            else:
                self.state['G_inv'] = results

    def get_gradient(self):
        """Get formated gradients (weight and bias) of module

        Returns:
          gradient of shape [ouput_dim, input_dim]. If bias != None, concats bias
        """
        g = self._get_weight_grad()
        if self.has_bias:
            g = torch.cat([g, self._get_bias_grad().data.view(-1, 1)], 1)
        return g

    def compute_preconditioned_gradient(self, damping=0.001):
        """Compute precondition gradient of each weight in module
        
        Preconditioned gradients can be applied to the actual gradients with 
        `update_gradient()`. Note the steps are separate in the event that
        intermediate steps will be applied to the preconditioned gradient.

        Args:
          damping (float, optional): damping to use if preconditioning using
              the eigendecomposition method. (default: 0.001)
        """
        if self.compute_grad_ranks is None:
            raise ValueError('Gradient preconditioning workers have not been '
                             'assigned yet. Have you called assign_workers() '
                             'yet?')
        if comm.backend.rank() not in self.compute_grad_ranks:
            return

        # Compute preconditioned gradient using specified inverse method
        if self.use_eigen_decomp:
            grad = self._get_precondition_gradient_eigen(damping)
        else:
            if self.factors_are_symmetric and self.symmetry_aware_comm:
                # Reconstruct inv if it was flattened for communication
                if len(self.state['A_inv'].shape) == 1:
                    rows, cols = self.state['A'].shape
                    self.state['A_inv'] = utils.fill_triu(
                            [rows, cols], self.state['A_inv'])
                if len(self.state['G_inv'].shape) == 1:
                    rows, cols = self.state['G'].shape
                    self.state['G_inv'] = utils.fill_triu(
                            [rows, cols], self.state['G_inv'])
            grad = self._get_precondition_gradient_inv()

        # Reshape appropriately
        if self.has_bias:
            grad = [grad[:, :-1], grad[:, -1:]]
            grad[0] = grad[0].view(self._get_weight_grad().data.size())
            grad[1] = grad[1].view(self._get_bias_grad().data.size())
        else:
            grad = [grad.view(self._get_weight_grad().data.size())]
        self.preconditioned_gradient = grad

    def save_inputs(self, input):
        """Save inputs locally"""
        if self.accumulate_data:
            self.a_inputs.append(input[0].data)
        else:
            self.a_inputs = [input[0].data]

    def save_grad_outputs(self, grad_output):
        """Save grad w.r.t outputs locally"""
        g = grad_output[0].data
        if self.grad_scaler is not None:
            g = (g, self.grad_scaler.get_scale())
        if self.accumulate_data:
            self.g_outputs.append(g)
        else:
            self.g_outputs = [g]

    def update_A_factor(self, alpha=0.95):
        """Compute factor A and add to running averages"""
        if len(self.a_inputs) == 0:
            return
        self.a_inputs = [x.to(self.factor_dtype) for x in self.a_inputs]
        A_new = self._get_A_factor(self.a_inputs)
        del self.a_inputs[:]  # clear accumulated inputs
        if self.state['A'] is None:
            self.state['A'] = torch.diag(A_new.new(A_new.shape[0]).fill_(1))
        utils.update_running_avg(A_new, self.state['A'], alpha=alpha)

    def update_G_factor(self, alpha=0.95):
        """Compute factor G and add to running averages"""
        # If half precision training: unscale accumulated gradients and discard
        # any with inf/NaN because of round-off errors. We need to unscale
        # to correctly compute G.
        if self.grad_scaler is not None:
            self.g_outputs = [g.to(self.factor_dtype) / scale 
                              for g, scale in self.g_outputs]
            length = len(self.g_outputs)
            self.g_outputs = [g for g in self.g_outputs 
                    if not (torch.isinf(g).any() or torch.isnan(g).any())]
            if len(self.g_outputs) != length:
                warnings.warn('Some gradients were discarded when computing '
                              'G because they were unable to be unscaled. '
                              'Note this can degrade KFAC performance if too '
                              'many gradients are discarded.')
        else:
            self.g_outputs = [x.to(self.factor_dtype) for x in self.g_outputs]

        if len(self.g_outputs) == 0:
            return
        G_new = self._get_G_factor(self.g_outputs)
        del self.g_outputs[:]  # clear accumulated outputs
        if self.state['G'] is None:
            self.state['G'] = torch.diag(G_new.new(G_new.shape[0]).fill_(1))
        utils.update_running_avg(G_new, self.state['G'], alpha=alpha)

    def update_gradient(self, scale=None):
        """Updates gradients of module with computed precondition gradients"""
        if self.preconditioned_gradient is None:
            raise RuntimeError('self.compute_preconditioned_gradient() should'
                    ' be called before update_gradient()')
        if scale is not None:
            v = [scale * x for x in self.preconditioned_gradient]
        else:
            v = self.preconditioned_gradient
        self._set_weight_grad(v[0])
        if self.has_bias:
            self._set_bias_grad(v[1])

    def _compute_factor_inverse(self, factor, damping=0.001):
        """Computes inverse/eigendecomp of factor and saves result to inverse"""
        if self.use_eigen_decomp:
            Q, d = utils.get_eigendecomp(factor.to(torch.float32), concat=False, 
                                         symmetric=self.factors_are_symmetric)
            return Q.to(self.inv_dtype), d.to(self.inv_dtype)
        else:
            inv = utils.get_inverse(factor.to(torch.float32), damping=damping, 
                        symmetric=self.factors_are_symmetric)
            return inv.to(self.inv_dtype)

    def _get_A_factor(self, a_inputs):
        """Compute A factor. Returns A."""
        raise NotImplementedError

    def _get_G_factor(self, g_inputs):
        """Compute G factor. Returns G."""
        raise NotImplementedError

    def _get_bias_grad(self):
        """Get bias.grad tensor of module"""
        return self.module.bias.grad

    def _get_weight_grad(self):
        """Get weight.grad tensor of module"""
        return self.module.weight.grad

    def _get_precondition_gradient_eigen(self, damping=0.001):
        """Compute preconditioned gradient for eigendecomp method"""
        QA = self.state['QA']
        QG = self.state['QG']
        grad = self.get_gradient().to(self.inv_dtype)
        v1 = QG.t() @ grad @ QA
        if self.prediv_eigenvalues:
            v2 = v1 * self.state['dGdA']
        else:
            v2 = v1 / (self.state['dG'].unsqueeze(1) * 
                       self.state['dA'].unsqueeze(0) + damping)
        return (QG @ v2 @ QA.t()).to(torch.float32)

    def _get_precondition_gradient_inv(self):
        """Compute preconditioned gradient for inverse method"""
        grad = self.get_gradient().to(self.inv_dtype)
        return (self.state['G_inv'] @ grad @ self.state['A_inv']).to(torch.float32)

    def _set_bias_grad(self, grad):
        """Set bias.grad tensor of module"""
        self.module.bias.grad = grad.contiguous()

    def _set_weight_grad(self, grad):
        """Set weight.grad tensor of module"""
        self.module.weight.grad = grad.contiguous()


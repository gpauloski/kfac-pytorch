from enum import Enum

import torch
import torch.distributed as dist

from kfac.distributed import Future


class AllreduceMethod(Enum):
    ALLREDUCE = 1
    ALLREDUCE_BUCKETED = 2


class BroadcastMethod(Enum):
    BROADCAST = 1
    ALLREDUCE = 2
    ALLREDUCE_BUCKETED = 3


class KFACBaseLayer:
    def __init__(
        self,
        module,
        module_helper,
        tdc,
        allreduce_method=AllreduceMethod.ALLREDUCE,
        broadcast_method=BroadcastMethod.BROADCAST,
        factor_dtype=None,
        grad_scaler=None,
        inv_dtype=None,
        symmetry_aware=False,
    ):
        self.module = module
        self.module_helper = module_helper
        self.tdc = tdc
        self.allreduce_method = allreduce_method
        self.broadcast_method = broadcast_method
        self.factor_dtype = factor_dtype
        self.grad_scaler = grad_scaler
        self.inv_dtype = inv_dtype
        self.symmetry_aware = symmetry_aware

        self.eps = 1e-10
        self.has_bias = self.module_helper.has_bias()
        self.symmetric_factors = self.module_helper.has_symmetric_factors()

        # Communication internal implementation helpers
        if self.allreduce_method == AllreduceMethod.ALLREDUCE:
            self._allreduce_fn = self.tdc.allreduce
        elif self.allreduce_method == AllreduceMethod.ALLREDUCE_BUCKETED:
            self._allreduce_fn = self.tdc.allreduce
        if self.broadcast_method == BroadcastMethod.BROADCAST:
            self._broadcast_fn = self.tdc.broadcast
        elif self.broadcast_method == BroadcastMethod.ALLREDUCE:
            self._broadcast_fn = self.tdc.allreduce
        elif self.broadcast_method == BroadcastMethod.ALLREDUCE_BUCKETED:
            self._broadcast_fn = self.tdc.allreduce_bucketed

        # KFAC State Variables
        self.a = None  # A factor being accumulated for current batch
        self.g = None  # G factor being accumulated for current batch
        self.a_count = None  # Number of inputs accumulated in self.a
        self.g_count = None  # Number of grads accumulated in self.g
        self.A = None  # Running average of A factor
        self.G = None  # Running average of G factor
        self.grad = None  # Preconditioned gradient

    def __del__(self):
        """Wait on communications before destruction

        It is possible the training script exits while a future to a
        communication op is still held so we sync before object deletion.
        Note that Python does not actually guarentee that __del__ is called
        when a script exits.
        """
        self.sync_a_factor()
        self.sync_g_factor()
        self.sync_grad()
        self.sync_a_inv()
        self.sync_g_inv()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({repr(self.module)}) {{\n"
            f"    A_inv worker: {self.a_inv_worker},\n"
            f"    G_inv worker: {self.g_inv_worker},\n"
            f"    is A_inv worker: {self.is_a_inv_worker},\n"
            f"    is G_inv worker: {self.is_g_inv_worker},\n"
            f"    is grad worker: {self.is_grad_worker},\n"
            f"    grad src worker: {self.grad_src_worker},\n"
            f"    grad worker group: {self.grad_worker_ranks},\n"
            f"    grad receiver group: {self.grad_receiver_ranks}\n"
            f"}}"
        )

    def state_dict(self):
        """Returns the state of the KFACLayer as a dictionary.

        Note:
            Only the factors are saved because because the factors are a
            running average so need to be restored properly and the remaining
            variables (e.g., inverses) can be recomputed.
        """
        self.sync_a_factor()
        self.sync_g_factor()
        return {"A": self.A, "G": self.G}

    def load_state_dict(self, state_dict):
        """Loads the KFACLayer state.

        Note:
            Factors will be placed on same device as module weights.
        """
        if "A" not in state_dict or "G" not in state_dict:
            raise KeyError(
                "KFACLayer state_dict must contain keys 'A' and 'G'",
            )
        device = next(self.module.parameters()).device
        self.A = state_dict["A"].to(device)
        self.G = state_dict["G"].to(device)

    def assign_workers(
        self,
        a_inv_worker,
        g_inv_worker,
        grad_src_worker,
        grad_worker_ranks,
        grad_worker_group,
        grad_receiver_ranks,
        grad_receiver_group,
    ):
        """Assign ranks to compute inverses

        Args:
          a_inv_worker (int): rank to compute A inverse.
          g_inv_worker (int): rank to compute G inverse.
          grad_src_worker (int): gradient worker that is responsible for
              sharing the preconditioned gradient with this rank
          grad_worker_ranks (List[int]): ranks that will compute preconditioned
              gradient.
          grad_worker_group (ProcessGroup): Torch ProcessGroup composed of
              grad_worker_ranks
          grad_receiver_ranks (List[int]): list of ranks for gradient
              communication groups.
          grad_receiver_group (ProcessGroup): Torch ProcessGroup for
              communicating the preconditioned gradients.
        """
        if a_inv_worker != g_inv_worker and self.prediv_eigenvalues:
            raise ValueError(
                "When precomputing 1 / (dG * dA.T + damping), A and G inverse "
                "workers must be the same. I.e. colocate_factors=True.",
            )
        if grad_src_worker not in grad_worker_ranks:
            raise ValueError(
                f"grad_src_worker is worker {grad_src_worker} which is not a "
                f"member of grad_worker_ranks={grad_worker_ranks}.",
            )
        if (
            a_inv_worker not in grad_worker_ranks
            or g_inv_worker not in grad_worker_ranks
        ):
            raise ValueError(
                f"a_inv_worker={a_inv_worker} and g_inv_worker={g_inv_worker} "
                f"must be members of grad_worker_ranks={grad_worker_ranks}.",
            )
        self.a_inv_worker = a_inv_worker
        self.g_inv_worker = g_inv_worker
        self.grad_src_worker = grad_src_worker
        self.grad_worker_ranks = grad_worker_ranks
        self.grad_worker_group = grad_worker_group
        self.grad_receiver_ranks = grad_receiver_ranks
        self.grad_receiver_group = grad_receiver_group
        self.is_a_inv_worker = dist.get_rank() == self.a_inv_worker
        self.is_g_inv_worker = dist.get_rank() == self.g_inv_worker
        self.is_grad_worker = dist.get_rank() in self.grad_worker_ranks

    def memory_usage(self):
        """Returns memory usage of this layer for this worker"""
        return {
            "a_factors": 0
            if self.A is None
            else self.A.nelement() * self.A.element_size(),
            "g_factors": 0
            if self.G is None
            else self.G.nelement() * self.G.element_size(),
        }

    def broadcast_a_inv(self):
        """Initiate A inv broadcast and store future to result

        Note:
            all ranks must enter this function even if the rank is not
            a part of the inverse broadcast group.
        """
        raise NotImplementedError

    def broadcast_g_inv(self):
        """Initiate G inv broadcast and store future to result

        Note:
            all ranks must enter this function even if the rank is not
            a part of the inverse broadcast group.
        """
        raise NotImplementedError

    def broadcast_grad(self):
        """Broadcast preconditioned gradient and store future to result

        Note:
            all ranks must enter this function
        """
        if len(self.grad_receiver_ranks) == 1:
            # COMM-OPT case -> no gradient communication
            return

        if self.grad is None:
            self.grad = torch.empty_like(self.module_helper.get_grad())

        kwargs = {}
        if self.broadcast_method == BroadcastMethod.BROADCAST:
            kwargs["src"] = self.grad_src_worker
        elif not self.is_grad_worker:
            self.grad.zero_()

        self.grad = self._broadcast_fn(
            self.grad,
            group=self.grad_receiver_group,
            **kwargs,
        )

    def compute_a_inv(self, damping=0.001):
        """Compute A inverse on assigned rank

        Note:
          - All ranks must enter this function even if the rank is not
            a part of the inverse broadcast group.
          - self.update_A_factor() must be called at least once before this
            function.

        Args:
          damping (float, optional): damping value to condition inverse
             (default: 0.001)
        """
        raise NotImplementedError

    def compute_g_inv(self, damping=0.001):
        """See `compute_A_inv`"""
        raise NotImplementedError

    def preconditioned_grad(self, damping=0.001):
        """Compute precondition gradient of each weight in module

        Preconditioned gradients can be applied to the actual gradients with
        `update_gradient()`. Note the steps are separate in the event that
        intermediate steps will be applied to the preconditioned gradient.

        Args:
          damping (float, optional): damping to use if preconditioning using
              the eigendecomposition method. (default: 0.001)
        """
        raise NotImplementedError

    def reduce_a_factor(self):
        """Initiate reduction of A and store future to result"""
        self.A = self._allreduce_fn(
            self.A,
            average=True,
            symmetric=self.symmetric_factors and self.symmetry_aware,
        )

    def reduce_g_factor(self):
        """Initiate reduction of G and store future to result"""
        self.G = self._allreduce_fn(
            self.G,
            average=True,
            symmetric=self.symmetric_factors and self.symmetry_aware,
        )

    def save_layer_input(self, input):
        """Save input for layer"""
        a = self.module_helper.get_a_factor(input[0])
        if self.a is None:
            self.a = a
            self.a_count = 1
        else:
            self.a = self.a + a
            self.a_count += 1

    def save_layer_grad_output(self, grad_output):
        """Save grad w.r.t outputs for layer"""
        g = self.module_helper.get_g_factor(grad_output[0])
        if self.grad_scaler is not None:
            g = g / self.grad_scaler.get_scale()
        if self.g is None:
            self.g = g
            self.g_count = 1
        else:
            self.g = self.g + g
            self.g_count += 1

    def sync_a_factor(self):
        if isinstance(self.A, Future):
            self.A = self.A.wait()

    def sync_a_inv(self):
        raise NotImplementedError

    def sync_g_factor(self):
        if isinstance(self.G, Future):
            self.G = self.G.wait()

    def sync_g_inv(self):
        raise NotImplementedError

    def sync_grad(self):
        if isinstance(self.grad, Future):
            self.grad = self.grad.wait()

    def update_a_factor(self, alpha=0.95):
        """Compute factor A and add to running averages"""
        if self.a is None:
            return
        if self.a_count > 1:
            self.a = (1 / self.a_count) * self.a
        A_new = self.a
        self.a = None
        self.sync_a_factor()
        if self.A is None:
            self.A = torch.diag(A_new.new(A_new.shape[0]).fill_(1))
        self.A = (alpha * self.A) + ((1 - alpha) * A_new)

    def update_g_factor(self, alpha=0.95):
        """Compute factor G and add to running averages"""
        if self.g is None:
            return
        if self.g_count > 1:
            self.g = (1 / self.g_count) * self.g
        G_new = self.g
        self.g = None
        self.sync_g_factor()
        if self.G is None:
            self.G = torch.diag(G_new.new(G_new.shape[0]).fill_(1))
        self.G = (alpha * self.G) + ((1 - alpha) * G_new)

    def update_grad(self, scale=None):
        """Updates gradients of module with computed precondition gradients"""
        self.sync_grad()
        if self.has_bias:
            weight = self.grad[:, :-1].view(self._get_weight_grad().size())
            bias = self.grad[:, -1:].view(self._get_bias_grad().size())
        else:
            weight = self.grad.view(self._get_weight_grad().size())
        if scale is not None:
            weight = scale * weight
        self._set_weight_grad(weight)
        if self.has_bias:
            if scale is not None:
                bias = scale * bias
            self._set_bias_grad(bias)
        self.grad = None

    def _get_bias_grad(self):
        """Get bias.grad tensor of module"""
        return self.module.bias.grad

    def _get_weight_grad(self):
        """Get weight.grad tensor of module"""
        return self.module.weight.grad

    def _set_bias_grad(self, grad):
        """Set bias.grad tensor of module"""
        self.module.bias.grad = grad.contiguous()

    def _set_weight_grad(self, grad):
        """Set weight.grad tensor of module"""
        self.module.weight.grad = grad.contiguous()

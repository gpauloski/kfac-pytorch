from __future__ import annotations

from enum import Enum
from typing import Any
from typing import Callable
from typing import cast

import torch
import torch.distributed as dist

from kfac.distributed import Future
from kfac.distributed import FutureType
from kfac.distributed import TorchDistributedCommunicator
from kfac.layers.modules import ModuleHelper


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
        module: torch.nn.Module,
        *,
        module_helper: ModuleHelper,
        tdc: TorchDistributedCommunicator,
        allreduce_method: AllreduceMethod = AllreduceMethod.ALLREDUCE,
        broadcast_method: BroadcastMethod = BroadcastMethod.BROADCAST,
        factor_dtype: torch.dtype | None = None,
        grad_scaler: (
            torch.cuda.amp.GradScaler | Callable[[], float] | None
        ) = None,
        inv_dtype: torch.dtype | None = None,
        symmetry_aware: bool = False,
    ):
        self.module = module
        self.module_helper = module_helper
        self.tdc = tdc
        self.allreduce_method = allreduce_method
        self.broadcast_method = broadcast_method
        self.factor_dtype = factor_dtype
        if isinstance(grad_scaler, torch.cuda.amp.GradScaler):
            grad_scaler = grad_scaler.get_scale
        self.grad_scaler: Callable[[], float] | None = grad_scaler
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
            self._broadcast_fn = self.tdc.allreduce  # type: ignore
        elif self.broadcast_method == BroadcastMethod.ALLREDUCE_BUCKETED:
            self._broadcast_fn = self.tdc.allreduce_bucketed  # type: ignore

        # KFAC State Variables
        # A factor being accumulated for current batch
        self._a_batch: torch.Tensor | None = None
        # G factor being accumulated for current batch
        self._g_batch: torch.Tensor | None = None
        # Number of inputs accumulated in self.a_batch
        self._a_count: int = 0
        # Number of grads accumulated in self.g_batch
        self._g_count: int = 0
        # Running average of A factor
        self._a_factor: torch.Tensor | FutureType | None = None
        # Running average of G factor
        self._g_factor: torch.Tensor | FutureType | None = None
        # Preconditioned gradient
        self._grad: torch.Tensor | FutureType | None = None

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}({repr(self.module)}) {{\n'
            f'    A_inv worker: {self.a_inv_worker},\n'
            f'    G_inv worker: {self.g_inv_worker},\n'
            f'    is A_inv worker: {self.is_a_inv_worker},\n'
            f'    is G_inv worker: {self.is_g_inv_worker},\n'
            f'    is grad worker: {self.is_grad_worker},\n'
            f'    grad src worker: {self.grad_src_worker},\n'
            f'    grad worker group: {self.grad_worker_ranks},\n'
            f'    grad receiver group: {self.grad_receiver_ranks}\n'
            f'}}'
        )

    @property
    def a_factor(self) -> torch.Tensor | None:
        if isinstance(self._a_factor, Future):
            self._a_factor = cast(torch.Tensor, self._a_factor.wait())
        return self._a_factor

    @a_factor.setter
    def a_factor(self, value: torch.Tensor | FutureType | None) -> None:
        self._a_factor = value

    @property
    def g_factor(self) -> torch.Tensor | None:
        if isinstance(self._g_factor, Future):
            self._g_factor = cast(torch.Tensor, self._g_factor.wait())
        return self._g_factor

    @g_factor.setter
    def g_factor(self, value: torch.Tensor | FutureType | None) -> None:
        self._g_factor = value

    @property
    def grad(self) -> torch.Tensor | None:
        if isinstance(self._grad, Future):
            self._grad = cast(torch.Tensor, self._grad.wait())
        return self._grad

    @grad.setter
    def grad(self, value: torch.Tensor | FutureType | None) -> None:
        self._grad = value

    def state_dict(self) -> dict[str, torch.Tensor | None]:
        """Returns the state of the KFACLayer as a dictionary.

        Note:
            Only the factors are saved because because the factors are a
            running average so need to be restored properly and the remaining
            variables (e.g., inverses) can be recomputed.
        """
        return {'A': self.a_factor, 'G': self.g_factor}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Loads the KFACLayer state.

        Note:
            Factors will be placed on same device as module weights.
        """
        if 'A' not in state_dict or 'G' not in state_dict:
            raise KeyError(
                "KFACLayer state_dict must contain keys 'A' and 'G'",
            )
        device = next(self.module.parameters()).device
        self.a_factor = state_dict['A'].to(device)
        self.g_factor = state_dict['G'].to(device)

    def assign_workers(
        self,
        a_inv_worker: int,
        g_inv_worker: int,
        grad_src_worker: int,
        grad_worker_ranks: frozenset[int],
        grad_worker_group: dist.ProcessGroup,
        grad_receiver_ranks: frozenset[int],
        grad_receiver_group: dist.ProcessGroup,
    ) -> None:
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
        if grad_src_worker not in grad_worker_ranks:
            raise ValueError(
                f'grad_src_worker is worker {grad_src_worker} which is not a '
                f'member of grad_worker_ranks={grad_worker_ranks}.',
            )
        if (
            a_inv_worker not in grad_worker_ranks
            or g_inv_worker not in grad_worker_ranks
        ):
            raise ValueError(
                f'a_inv_worker={a_inv_worker} and g_inv_worker={g_inv_worker} '
                f'must be members of grad_worker_ranks={grad_worker_ranks}.',
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

    def memory_usage(self) -> dict[str, int]:
        """Returns memory usage of this layer for this worker"""
        return {
            'a_factors': self.a_factor.nelement()
            * self.a_factor.element_size()
            if self.a_factor is not None
            else 0,
            'g_factors': self.g_factor.nelement()
            * self.g_factor.element_size()
            if self.g_factor is not None
            else 0,
            'a_batch': self._a_batch.nelement() * self._a_batch.element_size()
            if self._a_batch is not None
            else 0,
            'g_batch': self._g_batch.nelement() * self._g_batch.element_size()
            if self._g_batch is not None
            else 0,
        }

    def broadcast_a_inv(self) -> None:
        """Initiate A inv broadcast and store future to result

        Note:
            all ranks must enter this function even if the rank is not
            a part of the inverse broadcast group.
        """
        raise NotImplementedError

    def broadcast_g_inv(self) -> None:
        """Initiate G inv broadcast and store future to result

        Note:
            all ranks must enter this function even if the rank is not
            a part of the inverse broadcast group.
        """
        raise NotImplementedError

    def broadcast_grad(self) -> None:
        """Broadcast preconditioned gradient and store future to result

        Note:
            all ranks must enter this function
        """
        if len(self.grad_receiver_ranks) == 1:
            # COMM-OPT case -> no gradient communication
            return

        if self.grad is None:
            self.grad = torch.empty_like(self.module_helper.get_grad())

        kwargs: dict[str, Any] = {}
        if self.broadcast_method == BroadcastMethod.BROADCAST:
            kwargs['src'] = self.grad_src_worker
        elif not self.is_grad_worker:
            self.grad.zero_()

        self.grad = self._broadcast_fn(  # type: ignore
            self.grad,
            group=self.grad_receiver_group,
            **kwargs,
        )

    def compute_a_inv(self, damping: float = 0.001) -> None:
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

    def compute_g_inv(self, damping: float = 0.001) -> None:
        """See `compute_A_inv`"""
        raise NotImplementedError

    def preconditioned_grad(self, damping: float = 0.001) -> None:
        """Compute precondition gradient of each weight in module

        Preconditioned gradients can be applied to the actual gradients with
        `update_gradient()`. Note the steps are separate in the event that
        intermediate steps will be applied to the preconditioned gradient.

        Args:
          damping (float, optional): damping to use if preconditioning using
              the eigendecomposition method. (default: 0.001)
        """
        raise NotImplementedError

    def reduce_a_factor(self) -> None:
        """Initiate reduction of A and store future to result"""
        if self.a_factor is None:
            raise RuntimeError('a_factor is None, cannot reduce')
        self.a_factor = self._allreduce_fn(  # type: ignore
            self.a_factor,
            average=True,
            symmetric=self.symmetric_factors and self.symmetry_aware,
        )

    def reduce_g_factor(self) -> None:
        """Initiate reduction of G and store future to result"""
        if self.g_factor is None:
            raise RuntimeError('g_factor is None, cannot reduce')
        self.g_factor = self._allreduce_fn(  # type: ignore
            self.g_factor,
            average=True,
            symmetric=self.symmetric_factors and self.symmetry_aware,
        )

    def reset_batch(self) -> None:
        """Clears current buffers for A and G."""
        self._a_batch = None
        self._a_count = 0
        self._g_batch = None
        self._g_count = 0

    def save_layer_input(self, input: list[torch.Tensor]) -> None:
        """Save input for layer"""
        a = input[0]
        a = self.module_helper.get_a_factor(a)
        if self._a_batch is None:
            self._a_batch = a
            self._a_count = 1
        else:
            self._a_batch = self._a_batch + a
            self._a_count += 1

    def save_layer_grad_output(
        self,
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        """Save grad w.r.t outputs for layer"""
        g = grad_output[0]
        if self.grad_scaler is not None:
            g = g / self.grad_scaler()
        g = self.module_helper.get_g_factor(g)
        if self._g_batch is None:
            self._g_batch = g
            self._g_count = 1
        else:
            self._g_batch = self._g_batch + g
            self._g_count += 1

    def update_a_factor(self, alpha: float = 0.95) -> None:
        """Compute factor A and add to running averages"""
        if self._a_batch is None:
            return
        if self._a_count > 1:
            self._a_batch = (1 / self._a_count) * self._a_batch
        a_new = self._a_batch
        self._a_batch = None
        if self.a_factor is None:
            self.a_factor = torch.diag(a_new.new(a_new.shape[0]).fill_(1))
        self.a_factor = (alpha * self.a_factor) + ((1 - alpha) * a_new)

    def update_g_factor(self, alpha: float = 0.95) -> None:
        """Compute factor G and add to running averages"""
        if self._g_batch is None:
            return
        if self._g_count > 1:
            self._g_batch = (1 / self._g_count) * self._g_batch
        g_new = self._g_batch
        self._g_batch = None
        if self.g_factor is None:
            self.g_factor = torch.diag(g_new.new(g_new.shape[0]).fill_(1))
        self.g_factor = (alpha * self.g_factor) + ((1 - alpha) * g_new)

    def update_grad(self, scale: float | None = None) -> None:
        """Updates gradients of module with computed precondition gradients"""
        grad = self.grad
        if grad is None:
            raise RuntimeError(
                'preconditionined gradient is None. This may be because '
                'update_grad() was called before preconditioned_grad()',
            )
        if self.has_bias:
            weight = grad[:, :-1].view(self._get_weight_grad().size())
            bias = grad[:, -1:].view(self._get_bias_grad().size())
        else:
            weight = grad.view(self._get_weight_grad().size())
        if scale is not None:
            weight = scale * weight
        self._set_weight_grad(weight)
        if self.has_bias:
            if scale is not None:
                bias = scale * bias
            self._set_bias_grad(bias)
        self.grad = None

    def _get_bias_grad(self) -> torch.Tensor:
        """Get bias.grad tensor of module"""
        return cast(torch.Tensor, self.module.bias.grad)

    def _get_weight_grad(self) -> torch.Tensor:
        """Get weight.grad tensor of module"""
        return cast(torch.Tensor, self.module.weight.grad)

    def _set_bias_grad(self, grad: torch.Tensor) -> None:
        """Set bias.grad tensor of module"""
        self.module.bias.grad = grad.contiguous()

    def _set_weight_grad(self, grad: torch.Tensor) -> None:
        """Set weight.grad tensor of module"""
        self.module.weight.grad = grad.contiguous()

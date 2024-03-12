"""Base KFAC layer implementation."""

from __future__ import annotations

from typing import Callable
from typing import cast

import torch
import torch.distributed as dist

from kfac.distributed import Future
from kfac.distributed import FutureType
from kfac.distributed import get_rank
from kfac.distributed import TorchDistributedCommunicator
from kfac.enums import AllreduceMethod
from kfac.layers.modules import ModuleHelper


class KFACBaseLayer:
    """KFAC base layer implementation.

    There is a 1:1 mapping of KFAC layers to PyTorch modules in the model
    being preconditioned with KFAC. The KFACBaseLayer provides methods
    for the computations and communications required in the KFAC process.
    """

    def __init__(
        self,
        module: ModuleHelper,
        *,
        tdc: TorchDistributedCommunicator,
        allreduce_method: AllreduceMethod = AllreduceMethod.ALLREDUCE,
        factor_dtype: torch.dtype | None = None,
        grad_scaler: (
            torch.cuda.amp.GradScaler | Callable[[], float] | None
        ) = None,
        inv_dtype: torch.dtype = torch.float32,
        symmetry_aware: bool = False,
    ) -> None:
        """Init KFACBaseLayer.

        Args:
            module (ModuleHelper): module helper that exposes interfaces for
                getting the factors and gradients of a PyTorch module.
            tdc (TorchDistributedCommunicator): communicator object. Typically
                the communicator object should be shared by all KFACBaseLayers.
            allreduce_method (AllreduceMethod): allreduce method (default:
                AllreduceMethod.ALLREDUCE).
            factor_dtype (torch.dtype): data format to store factors in. If
                None, factors are stored in the format used in training
                (default: None).
            grad_scaler (optional): optional GradScaler or callable that
                returns the scale factor used in AMP training (default: None).
            inv_dtype (torch.dtype): data format to store inverses in.
                Inverses (or eigen decompositions) may be unstable in half-
                precision (default: torch.float32).
            symmetry_aware (bool): use symmetry aware communication method.
                This is typically more helpful when the factors are very
                large (default: False).
        """
        self.module = module
        self.tdc = tdc
        self.allreduce_method = allreduce_method
        self.factor_dtype = factor_dtype
        if isinstance(grad_scaler, torch.cuda.amp.GradScaler):
            grad_scaler = grad_scaler.get_scale
        self.grad_scaler: Callable[[], float] | None = grad_scaler
        self.inv_dtype = inv_dtype
        self.symmetry_aware = symmetry_aware

        self.eps = 1e-10
        self.symmetric_factors = self.module.has_symmetric_factors()

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
        """Representation of KFACBaseLayer."""
        return f'{self.__class__.__name__}({repr(self.module)})'

    @property
    def a_factor(self) -> torch.Tensor | None:
        """Get A factor."""
        if isinstance(self._a_factor, Future):
            self._a_factor = cast(torch.Tensor, self._a_factor.wait())
        return self._a_factor

    @a_factor.setter
    def a_factor(self, value: torch.Tensor | FutureType | None) -> None:
        """Set A factor."""
        self._a_factor = value

    @property
    def g_factor(self) -> torch.Tensor | None:
        """Get G factor."""
        if isinstance(self._g_factor, Future):
            self._g_factor = cast(torch.Tensor, self._g_factor.wait())
        return self._g_factor

    @g_factor.setter
    def g_factor(self, value: torch.Tensor | FutureType | None) -> None:
        """Set G factor."""
        self._g_factor = value

    @property
    def grad(self) -> torch.Tensor | None:
        """Get grad."""
        if isinstance(self._grad, Future):
            self._grad = cast(torch.Tensor, self._grad.wait())
        return self._grad

    @grad.setter
    def grad(self, value: torch.Tensor | FutureType | None) -> None:
        """Set grad."""
        self._grad = value

    def state_dict(self) -> dict[str, torch.Tensor | None]:
        """Returns the state of the KFACLayer as a dictionary.

        Note:
            Only the factors are saved because because the factors are a
            running average so need to be restored properly and the remaining
            variables (e.g., inverses) can be recomputed.

        Returns:
            dict containing two keys, 'A' and 'G', and the corresponding
            tensors for A and G.
        """
        return {'A': self.a_factor, 'G': self.g_factor}

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor | None],
    ) -> None:
        """Loads the KFACLayer state.

        Note:
            Factors will be placed on same device as module weights.

        Args:
            state_dict (dict): dict containing two keys, 'A' and 'G', and the
            corresponding tensors for A and G.
        """
        if 'A' not in state_dict or 'G' not in state_dict:
            raise KeyError(
                "KFACLayer state_dict must contain keys 'A' and 'G'",
            )
        device = self.module.device
        if state_dict['A'] is not None:
            self.a_factor = state_dict['A'].to(device)
        if state_dict['G'] is not None:
            self.g_factor = state_dict['G'].to(device)

    def memory_usage(self) -> dict[str, int]:
        """Returns memory usage of variables in this layer for this worker."""
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

    def broadcast_a_inv(
        self,
        src: int,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        """Initiate A inv broadcast and store future to result.

        Note:
            all ranks must enter this function even if the rank is not
            a part of the inverse broadcast group.

        Args:
            src (int): src rank that computed A inverse.
            group (ProcessGroup): process group to which src should broadcast
                A inv. All ranks in group should enter this function.
                Defaults to None, the default process group.
        """
        raise NotImplementedError

    def broadcast_g_inv(
        self,
        src: int,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        """Initiate G inv broadcast and store future to result.

        Note:
            all ranks must enter this function even if the rank is not
            a part of the inverse broadcast group.

        Args:
            src (int): src rank that computed G inverse.
            group (ProcessGroup): process group to which src should broadcast
                G inv. All ranks in group should enter this function.
                Defaults to None, the default process group.
        """
        raise NotImplementedError

    def broadcast_grad(
        self,
        src: int,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        """Broadcast preconditioned gradient and store future to result.

        Note:
            all ranks must enter this function.

        Args:
            src (int): src rank that preconditioned the gradient.
            group (ProcessGroup): process group to which src should broadcast
                the gradient. All ranks in group should enter this function.
                Defaults to None, the default process group.
        """
        if self.grad is None:
            if get_rank() == src:
                raise RuntimeError(
                    f'Attempt to broadcast gradient from src={src} but this '
                    'rank has not computed the preconditioned gradient yet.',
                )
            self.grad = torch.empty_like(self.module.get_grad())

        self.grad = self.tdc.broadcast(  # type: ignore
            self.grad,
            src=src,
            group=group,
        )

    def compute_a_inv(self, damping: float = 0.001) -> None:
        """Compute A inverse on assigned rank.

        update_a_factor() must be called at least once before this function.

        Args:
            damping (float, optional): damping value to condition inverse
                (default: 0.001).
        """
        raise NotImplementedError

    def compute_g_inv(self, damping: float = 0.001) -> None:
        """See `compute_g_inv`."""
        raise NotImplementedError

    def preconditioned_grad(self, damping: float = 0.001) -> None:
        """Compute precondition gradient of each weight in module.

        Preconditioned gradients can be applied to the actual gradients with
        `update_gradient()`. Note the steps are separate in the event that
        intermediate steps will be applied to the preconditioned gradient.

        Args:
            damping (float, optional): damping to use if preconditioning using
                the eigendecomposition method (default: 0.001).
        """
        raise NotImplementedError

    def reduce_a_factor(self, group: dist.ProcessGroup | None = None) -> None:
        """Initiate reduction of A and store future to result.

        Note:
            all ranks should enter this function.

        Args:
            group (ProcessGroup): process group to use for the reduce
                operation. All ranks in the group should enter this function.
                Defaults to None, the default process group.
        """
        if self.a_factor is None:
            raise RuntimeError('a_factor is None, cannot reduce')
        if self.allreduce_method == AllreduceMethod.ALLREDUCE:
            allreduce = self.tdc.allreduce
        elif self.allreduce_method == AllreduceMethod.ALLREDUCE_BUCKETED:
            allreduce = self.tdc.allreduce_bucketed
        else:
            raise AssertionError(
                f'Unknown allreduce_method={self.allreduce_method}',
            )
        self.a_factor = allreduce(  # type: ignore
            self.a_factor,
            average=True,
            symmetric=self.symmetric_factors and self.symmetry_aware,
            group=group,
        )

    def reduce_g_factor(self, group: dist.ProcessGroup | None = None) -> None:
        """Initiate reduction of G and store future to result.

        Note:
            all ranks should enter this function.

        Args:
            group (ProcessGroup): process group to use for the reduce
                operation. All ranks in the group should enter this function.
                Defaults to None, the default process group.
        """
        if self.g_factor is None:
            raise RuntimeError('g_factor is None, cannot reduce')
        if self.allreduce_method == AllreduceMethod.ALLREDUCE:
            allreduce = self.tdc.allreduce
        elif self.allreduce_method == AllreduceMethod.ALLREDUCE_BUCKETED:
            allreduce = self.tdc.allreduce_bucketed
        else:
            raise AssertionError(
                f'Unknown allreduce_method={self.allreduce_method}',
            )
        self.g_factor = allreduce(  # type: ignore
            self.g_factor,
            average=True,
            symmetric=self.symmetric_factors and self.symmetry_aware,
            group=group,
        )

    def reset_batch(self) -> None:
        """Clears current buffers for A and G."""
        self._a_batch = None
        self._a_count = 0
        self._g_batch = None
        self._g_count = 0

    def save_layer_input(self, input_: list[torch.Tensor]) -> None:
        """Save input for layer."""
        # Note: the clone here is a fix for "RuntimeError: one of the variables
        # needed for gradient computation has been modified by an inplace
        # operation" in the ResNet50 + ImageNet example.
        a = input_[0].to(self.factor_dtype).clone()
        a = self.module.get_a_factor(a)
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
        """Save grad w.r.t outputs for layer."""
        g = grad_output[0].to(self.factor_dtype)
        if self.grad_scaler is not None:
            g = g / self.grad_scaler()
        g = self.module.get_g_factor(g)
        if self._g_batch is None:
            self._g_batch = g
            self._g_count = 1
        else:
            self._g_batch = self._g_batch + g
            self._g_count += 1

    def update_a_factor(self, alpha: float = 0.95) -> None:
        """Compute factor A and add to running averages.

        Args:
            alpha (float): running average parameter (default: 0.95).
        """
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
        """Compute factor G and add to running averages.

        Args:
            alpha (float): running average parameter (default: 0.95).
        """
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
        """Updates gradients of module with computed precondition gradients.

        Args:
            scale (float, optional): optional factor to scale gradient by
                (default: None).
        """
        grad = self.grad
        if grad is None:
            raise RuntimeError(
                'preconditionined gradient is None. This may be because '
                'update_grad() was called before preconditioned_grad()',
            )
        if scale is not None:
            grad = scale * grad
        self.module.set_grad(grad)
        self.grad = None

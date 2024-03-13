"""Inverse preconditioning implementation."""

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
from kfac.layers.base import KFACBaseLayer
from kfac.layers.modules import ModuleHelper


class KFACInverseLayer(KFACBaseLayer):
    """KFAC layer that preconditions gradients with inverse factors."""

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
        """Init KFACInverseLayer.

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
        super().__init__(
            module=module,
            tdc=tdc,
            allreduce_method=allreduce_method,
            factor_dtype=factor_dtype,
            grad_scaler=grad_scaler,
            inv_dtype=inv_dtype,
            symmetry_aware=symmetry_aware,
        )

        # Inverse state variables
        # Inverse of self.a_factor
        self._a_inv: torch.Tensor | FutureType | None = None
        # Inverse of self.g_factor
        self._g_inv: torch.Tensor | FutureType | None = None

    @property
    def a_inv(self) -> torch.Tensor | None:
        """Get A inverse."""
        if isinstance(self._a_inv, Future):
            self._a_inv = cast(torch.Tensor, self._a_inv.wait())
        return self._a_inv

    @a_inv.setter
    def a_inv(self, value: torch.Tensor | FutureType | None) -> None:
        """Set A inverse."""
        self._a_inv = value

    @property
    def g_inv(self) -> torch.Tensor | None:
        """Get G inverse."""
        if isinstance(self._g_inv, Future):
            self._g_inv = cast(torch.Tensor, self._g_inv.wait())
        return self._g_inv

    @g_inv.setter
    def g_inv(self, value: torch.Tensor | FutureType | None) -> None:
        """Set G inverse."""
        self._g_inv = value

    def memory_usage(self) -> dict[str, int]:
        """Get memory usage for all variables in the layer."""
        sizes = super().memory_usage()
        sizes['a_inverses'] = (
            self.a_inv.nelement() * self.a_inv.element_size()
            if self.a_inv is not None
            else 0
        )
        sizes['g_inverses'] = (
            self.g_inv.nelement() * self.g_inv.element_size()
            if self.g_inv is not None
            else 0
        )
        return sizes

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
        if self.a_inv is None:
            if get_rank() == src:
                raise RuntimeError(
                    f'Attempt to broadcast A inv from src={src} but this rank '
                    'has not computed A inv yet.',
                )
            assert isinstance(self.a_factor, torch.Tensor)
            self.a_inv = torch.empty(
                self.a_factor.shape,
                device=self.a_factor.device,
                dtype=self.inv_dtype,
            )

        self.a_inv = self.tdc.broadcast(  # type: ignore
            self.a_inv,
            src=src,
            group=group,
            symmetric=self.symmetric_factors and self.symmetry_aware,
        )

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
        if self.g_inv is None:
            if get_rank() == src:
                raise RuntimeError(
                    f'Attempt to broadcast G inv from src={src} but this rank '
                    'has not computed G inv yet.',
                )
            assert isinstance(self.g_factor, torch.Tensor)
            self.g_inv = torch.empty(
                self.g_factor.shape,
                device=self.g_factor.device,
                dtype=self.inv_dtype,
            )

        self.g_inv = self.tdc.broadcast(  # type: ignore
            self.g_inv,
            src=src,
            group=group,
            symmetric=self.symmetric_factors and self.symmetry_aware,
        )

    def compute_a_inv(self, damping: float = 0.001) -> None:
        """Compute A inverse on assigned rank.

        update_a_factor() must be called at least once before this function.

        Args:
            damping (float, optional): damping value to condition inverse
                (default: 0.001).
        """
        if self.a_factor is None:
            raise RuntimeError('Cannot invert A before A has been computed')

        d = torch.diag(
            self.a_factor.new(self.a_factor.shape[0]).fill_(damping),
        )
        a = self.a_factor + d
        self.a_inv = torch.linalg.inv(a.to(torch.float32)).to(self.inv_dtype)

    def compute_g_inv(self, damping: float = 0.001) -> None:
        """See `compute_g_inv`."""
        if self.g_factor is None:
            raise RuntimeError('Cannot invert G before G has been computed')

        d = torch.diag(
            self.g_factor.new(self.g_factor.shape[0]).fill_(damping),
        )
        g = self.g_factor + d
        self.g_inv = torch.linalg.inv(g.to(torch.float32)).to(self.inv_dtype)

    def preconditioned_grad(self, damping: float = 0.001) -> None:
        """Compute precondition gradient of each weight in module.

        Preconditioned gradients can be applied to the actual gradients with
        `update_gradient()`. Note the steps are separate in the event that
        intermediate steps will be applied to the preconditioned gradient.

        Args:
            damping (float, optional): damping to use if preconditioning using
                the eigendecomposition method (default: 0.001).
        """
        if self.a_inv is None or self.g_inv is None:
            raise RuntimeError(
                'Cannot precondition gradient before A and G have been '
                'inverted',
            )
        grad = self.module.get_grad()
        grad_type = grad.dtype
        grad = grad.to(self.a_inv.dtype)
        self.grad = (self.g_inv @ grad @ self.a_inv).to(grad_type)

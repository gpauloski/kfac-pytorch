from __future__ import annotations

from typing import Any
from typing import Callable
from typing import cast

import torch

from kfac.distributed import Future
from kfac.distributed import FutureType
from kfac.distributed import TorchDistributedCommunicator
from kfac.layers.base import AllreduceMethod
from kfac.layers.base import BroadcastMethod
from kfac.layers.base import KFACBaseLayer
from kfac.layers.modules import ModuleHelper


class KFACInverseLayer(KFACBaseLayer):
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
    ) -> None:
        super().__init__(
            module=module,
            module_helper=module_helper,
            tdc=tdc,
            allreduce_method=allreduce_method,
            broadcast_method=broadcast_method,
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
        if isinstance(self._a_inv, Future):
            self._a_inv = cast(torch.Tensor, self._a_inv.wait())
        return self._a_inv

    @a_inv.setter
    def a_inv(self, value: torch.Tensor | FutureType | None) -> None:
        self._a_inv = value

    @property
    def g_inv(self) -> torch.Tensor | None:
        if isinstance(self._g_inv, Future):
            self._g_inv = cast(torch.Tensor, self._g_inv.wait())
        return self._g_inv

    @g_inv.setter
    def g_inv(self, value: torch.Tensor | FutureType | None) -> None:
        self._g_inv = value

    def memory_usage(self) -> dict[str, int]:
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

    def broadcast_a_inv(self) -> None:
        if not self.is_grad_worker:
            return
        if self.a_inv is None:
            raise ValueError(
                'Cannot broadcast A inverse before A has been inverted',
            )

        kwargs: dict[str, Any] = {}
        if self.broadcast_method == BroadcastMethod.BROADCAST:
            kwargs['src'] = self.g_inv_worker
        elif not self.is_a_inv_worker:
            self.a_inv.zero_()

        self.a_inv = self._broadcast_fn(  # type: ignore
            self.a_inv,
            group=self.grad_worker_group,
            symmetric=self.symmetric_factors and self.symmetry_aware,
            **kwargs,
        )

    def broadcast_g_inv(self) -> None:
        if not self.is_grad_worker:
            return
        if self.g_inv is None:
            raise ValueError(
                'Cannot broadcast A inverse before A has been inverted',
            )

        kwargs: dict[str, Any] = {}
        if self.broadcast_method == BroadcastMethod.BROADCAST:
            kwargs['src'] = self.g_inv_worker
        elif not self.is_g_inv_worker:
            self.g_inv.zero_()

        self.g_inv = self._broadcast_fn(  # type: ignore
            self.g_inv,
            group=self.grad_worker_group,
            symmetric=self.symmetric_factors and self.symmetry_aware,
            **kwargs,
        )

    def compute_a_inv(self, damping: float = 0.001) -> None:
        if not self.is_grad_worker:
            return
        if self.a_factor is None:
            raise RuntimeError('Cannot invert A before A has been computed')

        if self.a_inv is None:
            self.a_inv = torch.empty_like(self.a_factor)
        if self.is_a_inv_worker:
            d = torch.diag(
                self.a_factor.new(self.a_factor.shape[0]).fill_(damping),
            )
            a = self.a_factor + d
            torch.linalg.inv(a.to(torch.float32), out=self.a_inv)
            self.a_inv = self.a_inv.to(self.inv_dtype)

    def compute_g_inv(self, damping: float = 0.001) -> None:
        if not self.is_grad_worker:
            return
        if self.g_factor is None:
            raise RuntimeError('Cannot invert G before G has been computed')

        if self.g_inv is None:
            self.g_inv = torch.empty_like(self.g_factor)
        if self.is_g_inv_worker:
            d = torch.diag(
                self.g_factor.new(self.g_factor.shape[0]).fill_(damping),
            )
            g = self.g_factor + d
            torch.linalg.inv(g.to(torch.float32), out=self.g_inv)
            self.g_inv = self.g_inv.to(self.inv_dtype)

    def preconditioned_grad(self, damping: float = 0.001) -> None:
        if not self.is_grad_worker:
            return
        if self.a_inv is None or self.g_inv is None:
            raise RuntimeError(
                'Cannot precondition gradient before A and G have been '
                'inverted',
            )
        grad = self.module_helper.get_grad()
        grad_type = grad.dtype
        grad = grad.to(self.a_inv.dtype)
        self.grad = (self.g_inv @ grad @ self.a_inv).to(grad_type)

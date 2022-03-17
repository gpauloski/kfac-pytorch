from __future__ import annotations

from typing import Any
from typing import Callable
from typing import cast

import torch
import torch.distributed as dist

from kfac.distributed import Future
from kfac.distributed import FutureType
from kfac.distributed import TorchDistributedCommunicator
from kfac.layers.base import AllreduceMethod
from kfac.layers.base import BroadcastMethod
from kfac.layers.base import KFACBaseLayer
from kfac.layers.modules import ModuleHelper


class KFACEigenLayer(KFACBaseLayer):
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
        prediv_eigenvalues: bool = False,
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
        self.prediv_eigenvalues = prediv_eigenvalues

        # Eigen state variables
        # Eigenvectors of self.a_factor
        self._qa: torch.Tensor | FutureType | None = None
        # Eigenvectors of self.g_factor
        self._qg: torch.Tensor | FutureType | None = None
        # Eigenvalues of self.a_factor
        self._da: torch.Tensor | FutureType | None = None
        # Eigenvalues of self.g_factor
        self._dg: torch.Tensor | FutureType | None = None
        # Outer product + damping of eigenvalues
        # Only used if self.prediv_eigenvalues
        self._dgda: torch.Tensor | FutureType | None = None

    @property
    def qa(self) -> torch.Tensor | None:
        if isinstance(self._qa, Future):
            self._qa = cast(torch.Tensor, self._qa.wait())
        return self._qa

    @qa.setter
    def qa(self, value: torch.Tensor | FutureType | None) -> None:
        self._qa = value

    @property
    def qg(self) -> torch.Tensor | None:
        if isinstance(self._qg, Future):
            self._qg = cast(torch.Tensor, self._qg.wait())
        return self._qg

    @qg.setter
    def qg(self, value: torch.Tensor | FutureType | None) -> None:
        self._qg = value

    @property
    def da(self) -> torch.Tensor | None:
        if isinstance(self._da, Future):
            self._da = cast(torch.Tensor, self._da.wait())
        return self._da

    @da.setter
    def da(self, value: torch.Tensor | FutureType | None) -> None:
        self._da = value

    @property
    def dg(self) -> torch.Tensor | None:
        if isinstance(self._dg, Future):
            self._dg = cast(torch.Tensor, self._dg.wait())
        return self._dg

    @dg.setter
    def dg(self, value: torch.Tensor | FutureType | None) -> None:
        self._dg = value

    @property
    def dgda(self) -> torch.Tensor | None:
        if isinstance(self._dgda, Future):
            self._dgda = cast(torch.Tensor, self._dgda.wait())
        return self._dgda

    @dgda.setter
    def dgda(self, value: torch.Tensor | FutureType | None) -> None:
        self._dgda = value

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
        if a_inv_worker != g_inv_worker and self.prediv_eigenvalues:
            raise ValueError(
                'When precomputing 1 / (dG * dA.T + damping), A and G inverse '
                'workers must be the same. I.e. colocate_factors=True.',
            )
        super().assign_workers(
            a_inv_worker=a_inv_worker,
            g_inv_worker=g_inv_worker,
            grad_src_worker=grad_src_worker,
            grad_worker_ranks=grad_worker_ranks,
            grad_worker_group=grad_receiver_group,
            grad_receiver_ranks=grad_receiver_ranks,
            grad_receiver_group=grad_receiver_group,
        )

    def memory_usage(self) -> dict[str, int]:
        sizes = super().memory_usage()
        a_size = (
            self.qa.nelement() * self.qa.element_size()
            if self.qa is not None
            else 0
        )
        a_size += (
            self.da.nelement() * self.da.element_size()
            if self.da is not None
            else 0
        )
        g_size = (
            self.qg.nelement() * self.qg.element_size()
            if self.qg is not None
            else 0
        )
        g_size += (
            self.dg.nelement() * self.dg.element_size()
            if self.dg is not None
            else 0
        )
        g_size += (
            self.dgda.nelement() * self.dgda.element_size()
            if self.dgda is not None
            else 0
        )
        sizes['a_inverses'] = a_size
        sizes['g_inverses'] = g_size
        return sizes

    def broadcast_a_inv(self) -> None:
        if not self.is_grad_worker:
            return
        if len(self.grad_worker_ranks) == 1:
            # MEM-OPT case -> no communication necessary
            return

        assert self.qa is not None
        kwargs: dict[str, Any] = {}
        if self.broadcast_method == BroadcastMethod.BROADCAST:
            kwargs['src'] = self.a_inv_worker
        elif not self.is_a_inv_worker:
            self.qa.zero_()
            if not self.prediv_eigenvalues:
                assert self.da is not None
                self.da.zero_()

        self.qa = self._broadcast_fn(  # type: ignore
            self.qa,
            group=self.grad_worker_group,
            **kwargs,
        )
        if not self.prediv_eigenvalues:
            self.da = self._broadcast_fn(  # type: ignore
                cast(torch.Tensor, self.da),
                group=self.grad_worker_group,
                **kwargs,
            )

    def broadcast_g_inv(self) -> None:
        if not self.is_grad_worker:
            return
        if len(self.grad_worker_ranks) == 1:
            # MEM-OPT case -> no communication necessary
            return
        if (
            self.qg is None
            or (not self.prediv_eigenvalues and self.dg is None)
            or (self.prediv_eigenvalues and self.dgda is None)
        ):
            raise RuntimeError('G eigendecomp buffers not initialized')

        assert self.qg is not None
        kwargs: dict[str, Any] = {}
        if self.broadcast_method == BroadcastMethod.BROADCAST:
            kwargs['src'] = self.g_inv_worker
        elif not self.is_a_inv_worker:
            self.qg.zero_()
            if not self.prediv_eigenvalues:
                assert self.dg is not None
                self.dg.zero_()
            else:
                assert self.dgda is not None
                self.dgda.zero_()

        self.qg = self._broadcast_fn(  # type: ignore
            self.qg,
            group=self.grad_worker_group,
            **kwargs,
        )
        if not self.prediv_eigenvalues:
            self.dg = self._broadcast_fn(  # type: ignore
                cast(torch.Tensor, self.dg),
                group=self.grad_worker_group,
                **kwargs,
            )
        else:
            self.dgda = self._broadcast_fn(  # type: ignore
                cast(torch.Tensor, self.dgda),
                group=self.grad_worker_group,
                **kwargs,
            )

    def compute_a_inv(self, damping: float = 0.001) -> None:
        if not self.is_grad_worker:
            return
        if self.a_factor is None:
            raise RuntimeError(
                'Cannot eigendecompose A before A has been computed',
            )

        if self.qa is None:
            self.qa = torch.empty_like(self.a_factor, dtype=torch.float32)
            if (
                not self.prediv_eigenvalues or self.is_a_inv_worker
            ) and self.da is None:
                self.da = self.a_factor.new_empty(
                    self.a_factor.shape[0],
                    dtype=torch.float32,
                )
        if self.is_a_inv_worker:
            if self.symmetric_factors:
                torch.linalg.eigh(
                    self.a_factor.to(torch.float32),
                    out=(self.da, self.qa),
                )
            else:
                torch.linalg.eig(
                    self.a_factor.to(torch.float32),
                    out=(self.da, self.qa),
                )
            self.qa = self.qa.to(self.inv_dtype)
            self.da = cast(torch.Tensor, self.da).to(self.inv_dtype)
            self.da = torch.clamp(self.da, min=0.0)

    def compute_g_inv(self, damping: float = 0.001) -> None:
        if not self.is_grad_worker:
            return
        if self.g_factor is None:
            raise RuntimeError(
                'Cannot eigendecompose G before G has been computed',
            )

        if self.qg is None:
            self.qg = torch.empty_like(self.g_factor, dtype=torch.float32)
            if (
                not self.prediv_eigenvalues or self.is_g_inv_worker
            ) and self.dg is None:
                self.dg = self.g_factor.new_empty(
                    self.g_factor.shape[0],
                    dtype=torch.float32,
                )
            elif self.dgda is None:
                if self.a_factor is None:
                    raise RuntimeError(
                        'Cannot compute dGdA before A has been computed',
                    )
                self.dgda = self.g_factor.new_empty(
                    (self.g_factor.shape[0], self.a_factor.shape[0]),
                    dtype=torch.float32,
                )
        if self.is_g_inv_worker:
            if self.symmetric_factors:
                torch.linalg.eigh(
                    self.g_factor.to(torch.float32),
                    out=(self.dg, self.qg),
                )
            else:
                torch.linalg.eig(
                    self.g_factor.to(torch.float32),
                    out=(self.dg, self.qg),
                )
            self.qg = self.qg.to(self.inv_dtype)
            self.dg = cast(torch.Tensor, self.dg).to(self.inv_dtype)
            self.dg = torch.clamp(self.dg, min=0.0)
            if self.prediv_eigenvalues:
                self.dgda = 1 / (
                    torch.outer(self.dg, cast(torch.Tensor, self.da)) + damping
                )

    def preconditioned_grad(self, damping: float = 0.001) -> None:
        if not self.is_grad_worker:
            return
        if self.qa is None or (
            not self.prediv_eigenvalues and self.da is None
        ):
            raise RuntimeError('QA has not been computed yet')
        if (
            self.qg is None
            or (not self.prediv_eigenvalues and self.dg is None)
            or (self.prediv_eigenvalues and self.dgda is None)
        ):
            raise RuntimeError('QG has not been computed yet')
        grad = self.module_helper.get_grad()
        grad_type = grad.dtype
        grad = grad.to(self.qa.dtype)
        v1 = self.qg.t() @ grad @ self.qa
        if self.prediv_eigenvalues:
            v2 = v1 * self.dgda
        else:
            v2 = v1 / (
                torch.outer(
                    cast(torch.Tensor, self.dg),
                    cast(torch.Tensor, self.da),
                )
                + damping
            )
        self.grad = (self.qg @ v2 @ self.qa.t()).to(grad_type)

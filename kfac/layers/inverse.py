import torch

from kfac.distributed import Future
from kfac.layers.base import BroadcastMethod
from kfac.layers.base import KFACBaseLayer


class KFACInverseLayer(KFACBaseLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Inverse state variables
        self._A_inv = None  # Inverse of self.A
        self._G_inv = None  # Inverse of self.G

    @property
    def A_inv(self):
        if isinstance(self._A_inv, Future):
            self._A_inv = self._A_inv.wait()
        return self._A_inv

    @A_inv.setter
    def A_inv(self, value):
        self._A_inv = value

    @property
    def G_inv(self):
        if isinstance(self._G_inv, Future):
            self._G_inv = self._G_inv.wait()
        return self._G_inv

    @G_inv.setter
    def G_inv(self, value):
        self._G_inv = value

    def memory_usage(self):
        sizes = super().memory_usage()
        sizes["a_inverses"] = (
            0
            if self.A_inv is None
            else self.A_inv.nelement() * self.A_inv.element_size()
        )
        sizes["g_inverses"] = (
            0
            if self.G_inv is None
            else self.G_inv.nelement() * self.G_inv.element_size()
        )
        return sizes

    def broadcast_a_inv(self):
        if not self.is_grad_worker:
            return

        kwargs = {}
        if self.broadcast_method == BroadcastMethod.BROADCAST:
            kwargs["src"] = self.g_inv_worker
        elif not self.is_a_inv_worker:
            self.A_inv.zero_()

        self.A_inv = self._broadcast_fn(
            self.A_inv,
            group=self.grad_worker_group,
            symmetric=self.symmetric_factors and self.symmetry_aware,
            **kwargs,
        )

    def broadcast_g_inv(self):
        if not self.is_grad_worker:
            return

        kwargs = {}
        if self.broadcast_method == BroadcastMethod.BROADCAST:
            kwargs["src"] = self.g_inv_worker
        elif not self.is_g_inv_worker:
            self.G_inv.zero_()

        self.G_inv = self._broadcast_fn(
            self.G_inv,
            group=self.grad_worker_group,
            symmetric=self.symmetric_factors and self.symmetry_aware,
            **kwargs,
        )

    def compute_a_inv(self, damping=0.001):
        if not self.is_grad_worker:
            return

        if self.A_inv is None:
            self.A_inv = torch.empty_like(self.A)
        if self.is_a_inv_worker:
            d = torch.diag(self.A.new(self.A.shape[0]).fill_(damping))
            A = self.A + d
            torch.linalg.inv(A.to(torch.float32), out=self.A_inv)
            self.A_inv = self.A_inv.to(self.inv_dtype)

    def compute_g_inv(self, damping=0.001):
        if not self.is_grad_worker:
            return

        if self.G_inv is None:
            self.G_inv = torch.empty_like(self.G)
        if self.is_g_inv_worker:
            d = torch.diag(self.G.new(self.G.shape[0]).fill_(damping))
            G = self.G + d
            torch.linalg.inv(G.to(torch.float32), out=self.G_inv)
            self.G_inv = self.G_inv.to(self.inv_dtype)

    def preconditioned_grad(self, damping=0.001):
        if not self.is_grad_worker:
            return
        grad = self.module_helper.get_grad()
        grad_type = grad.dtype
        grad = grad.to(self.A_inv.dtype)
        self.grad = (self.G_inv @ grad @ self.A_inv).to(grad_type)

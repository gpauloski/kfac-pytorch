import torch

from kfac.distributed import Future
from kfac.layers.base import KFACBaseLayer


class KFACInverseLayer(KFACBaseLayer):
    def __init__(self, *args, **kwargs):
        super(KFACInverseLayer, self).__init__(*args, **kwargs)

        # Inverse state variables
        self.A_inv = None  # Inverse of self.A
        self.G_inv = None  # Inverse of self.G

    def memory_usage(self):
        sizes = super(KFACInverseLayer, self).memory_usage()
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
        self.A_inv = self.comm.broadcast(
            self.A_inv,
            src=self.a_inv_worker,
            group=self.grad_worker_group,
            upper_tri=self.symmetric_factors and self.symmetry_aware,
        )

    def broadcast_g_inv(self):
        if not self.is_grad_worker:
            return
        self.G_inv = self.comm.broadcast(
            self.G_inv,
            src=self.g_inv_worker,
            group=self.grad_worker_group,
            upper_tri=self.symmetric_factors and self.symmetry_aware,
        )

    def compute_a_inv(self, damping=0.001):
        if not self.is_grad_worker:
            return
        self.sync_a_factor()

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
        self.sync_g_factor()

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
        self.sync_a_inv()
        self.sync_g_inv()
        grad = self.module_helper.get_grad()
        grad_type = grad.dtype
        grad = grad.to(self.A_inv.dtype)
        self.grad = (self.G_inv @ grad @ self.A_inv).to(grad_type)

    def sync_a_inv(self):
        if isinstance(self.A_inv, Future):
            self.A_inv = self.A_inv.wait()

    def sync_g_inv(self):
        if isinstance(self.G_inv, Future):
            self.G_inv = self.G_inv.wait()

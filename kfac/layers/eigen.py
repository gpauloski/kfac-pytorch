import torch

from kfac.distributed import Future
from kfac.layers.base import KFACBaseLayer


class KFACEigenLayer(KFACBaseLayer):
    def __init__(self, *args, prediv_eigenvalues=False, **kwargs):
        super(KFACEigenLayer, self).__init__(*args, **kwargs)
        self.prediv_eigenvalues = prediv_eigenvalues

        # Eigen state variables
        self.QA = None  # Eigenvectors of self.A
        self.QG = None  # Eigenvectors of self.G
        self.dA = None  # Eigenvalues of self.A
        self.dG = None  # Eigenvalues of self.G
        # Only used if self.prediv_eigenvalues
        self.dGdA = None  # Outer product + damping of eigenvalues

    def memory_usage(self):
        sizes = super(KFACEigenLayer, self).memory_usage()
        a_size = (
            0
            if self.QA is None
            else self.QA.nelement() * self.QA.element_size()
        )
        a_size += (
            0
            if self.dA is None
            else self.dA.nelement() * self.dA.element_size()
        )
        g_size = (
            0
            if self.QG is None
            else self.QG.nelement() * self.QG.element_size()
        )
        g_size += (
            0
            if self.dG is None
            else self.dG.nelement() * self.dG.element_size()
        )
        g_size += (
            0
            if self.dGdA is None
            else self.dGdA.nelement() * self.dGdA.element_size()
        )
        sizes["a_inverses"] = a_size
        sizes["g_inverses"] = g_size
        return sizes

    def broadcast_a_inv(self):
        if not self.is_grad_worker:
            return
        if len(self.grad_worker_ranks) == 1:
            # MEM-OPT case -> no communication necessary
            return
        self.QA = self.comm.broadcast(
            self.QA, src=self.a_inv_worker, group=self.grad_worker_group
        )
        if not self.prediv_eigenvalues:
            self.dA = self.comm.broadcast(
                self.dA, src=self.a_inv_worker, group=self.grad_worker_group
            )

    def broadcast_g_inv(self):
        if not self.is_grad_worker:
            return
        if len(self.grad_worker_ranks) == 1:
            # MEM-OPT case -> no communication necessary
            return
        self.QG = self.comm.broadcast(
            self.QG, src=self.g_inv_worker, group=self.grad_worker_group
        )
        if not self.prediv_eigenvalues:
            self.dG = self.comm.broadcast(
                self.dG, src=self.g_inv_worker, group=self.grad_worker_group
            )
        else:
            self.dGdA = self.comm.broadcast(
                self.dGdA, src=self.g_inv_worker, group=self.grad_worker_group
            )

    def compute_a_inv(self, damping=0.001):
        if not self.is_grad_worker:
            return
        self.sync_a_factor()

        if self.QA is None:
            self.QA = torch.empty_like(self.A, dtype=self.inv_dtype)
            if (
                not self.prediv_eigenvalues or self.is_a_inv_worker
            ) and self.dA is None:
                self.dA = self.A.new_empty(
                    self.A.shape[0], dtype=self.inv_dtype
                )
        if self.is_a_inv_worker:
            if self.symmetric_factors:
                torch.linalg.eigh(
                    self.A.to(torch.float32), out=(self.dA, self.QA)
                )
            else:
                torch.linalg.eig(
                    self.A.to(torch.float32), out=(self.dA, self.QA)
                )
            self.dA = torch.clamp(self.dA, min=0.0)
            self.QA = self.QA.to(self.inv_dtype)
            self.dA = self.dA.to(self.inv_dtype)

    def compute_g_inv(self, damping=0.001):
        if not self.is_grad_worker:
            return
        self.sync_g_factor()

        if self.QG is None:
            self.QG = torch.empty_like(self.G, dtype=self.inv_dtype)
            if (
                not self.prediv_eigenvalues or self.is_g_inv_worker
            ) and self.dG is None:
                self.dG = self.G.new_empty(
                    self.G.shape[0], dtype=self.inv_dtype
                )
            elif self.dGdA is None:
                self.dGdA = self.G.new_empty(
                    (self.G.shape[0], self.A.shape[0]), dtype=self.inv_dtype
                )
        if self.is_g_inv_worker:
            if self.symmetric_factors:
                torch.linalg.eigh(
                    self.G.to(torch.float32), out=(self.dG, self.QG)
                )
            else:
                torch.linalg.eig(
                    self.G.to(torch.float32), out=(self.dG, self.QG)
                )
            self.dG = torch.clamp(self.dG, min=0.0)
            self.QG = self.QG.to(self.inv_dtype)
            self.dG = self.dG.to(self.inv_dtype)
            if self.prediv_eigenvalues:
                self.dGdA = 1 / (torch.outer(self.dG, self.dA) + damping)

    def preconditioned_grad(self, damping=0.001):
        if not self.is_grad_worker:
            return
        self.sync_a_inv()
        self.sync_g_inv()
        grad = self.module_helper.get_grad()
        grad_type = grad.dtype
        grad = grad.to(self.QA.dtype)
        v1 = self.QG.t() @ grad @ self.QA
        if self.prediv_eigenvalues:
            v2 = v1 * self.dGdA
        else:
            v2 = v1 / (torch.outer(self.dG, self.dA) + damping)
        self.grad = (self.QG @ v2 @ self.QA.t()).to(grad_type)

    def sync_a_inv(self):
        if isinstance(self.QA, Future):
            self.QA = self.QA.wait()
        if isinstance(self.dA, Future):
            self.dA = self.dA.wait()

    def sync_g_inv(self):
        if isinstance(self.QG, Future):
            self.QG = self.QG.wait()
        if isinstance(self.dG, Future):
            self.dG = self.dG.wait()
        if self.prediv_eigenvalues and isinstance(self.dGdA, Future):
            self.dGdA = self.dGdA.wait()

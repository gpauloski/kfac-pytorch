"""Custom GPT NeoX Eigen Layer."""

from __future__ import annotations

import sys
from typing import Any

if sys.version_info >= (3, 9):  # pragma: >=3.9 cover
    from typing import Literal
else:  # pragma: <3.9 cover
    from typing_extensions import Literal

import torch

from kfac.distributed import get_rank
from kfac.distributed import get_world_size
from kfac.gpt_neox.mpu import gather_from_model_parallel_region
from kfac.gpt_neox.mpu import split_tensor_along_dim
from kfac.layers.eigen import KFACEigenLayer
from kfac.layers.modules import ModuleHelper


class GPTNeoXKFACEigenLayer(KFACEigenLayer):
    """Custom Eigen Layer that is model parallel aware."""

    def __init__(
        self,
        module: ModuleHelper,
        *,
        parallelism: Literal['input', 'output'],
        model_parallel_group: torch.distributed.ProcessGroup | None,
        # Use -1 to indicate unset because None is a valid ProcessGroup
        data_parallel_group: torch.distributed.ProcessGroup | None | int = -1,
        pipe_parallel_peer_group: torch.distributed.ProcessGroup
        | None
        | int = -1,
        primary_rank: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Init GPTNeoXKFACEigenLayer.

        Args:
            module (ModuleHelper): module helper that exposes interfaces for
                getting the factors and gradients of a PyTorch module.
            parallelism (str): either "input" or "output" depending on if
                input or output is sharded across model parallel ranks.
            model_parallel_group (ProcessGroup): model parallel distribtued
                group this rank belongs to.
            data_parallel_group (ProcessGroup): data parallel distributed
                group this rank belongs to.
            pipe_parallel_peer_group (ProcessGroup): process group with all
                ranks belonging to same pipe parallel rank.
            primary_rank (int): if model parallel size > 1, the primary rank
                is the rank that data will be gather to, computed on, and
                then scattered back. Can optionally be set later.
            **kwargs: additional keyword arguments to pass to KFACEigenLayer.
        """
        self.parallelism = parallelism
        self.primary_rank = primary_rank
        self.data_parallel_group = data_parallel_group
        self.model_parallel_group = model_parallel_group
        self.pipe_parallel_peer_group = pipe_parallel_peer_group

        super().__init__(module=module, **kwargs)

    def reduce_a_factor(
        self,
        group: torch.distributed.ProcessGroup | None = None,
    ) -> None:  # pragma: no cover
        """Initiate reduction of A and store future to result.

        Note:
            all ranks should enter this function.

        Args:
            group (ProcessGroup): ignored because the correct group depends
                on if the parallelism is on the output or input of the layer.
        """
        if self.primary_rank is None:
            raise RuntimeError('primary rank has not been set yet.')
        valid = (torch.distributed.ProcessGroup, type(None))
        if not isinstance(self.data_parallel_group, valid) or not isinstance(
            self.pipe_parallel_peer_group,
            valid,
        ):
            raise RuntimeError(
                'data_parallel_group or pipe_parallel_peer_group has not '
                'been set yet.',
            )

        if self.parallelism == 'input':
            if get_rank() != self.primary_rank:
                return
            super().reduce_a_factor(self.data_parallel_group)
        elif self.parallelism == 'output':
            super().reduce_a_factor(self.pipe_parallel_peer_group)
        else:
            raise AssertionError('Unreachable.')

    def reduce_g_factor(
        self,
        group: torch.distributed.ProcessGroup | None = None,
    ) -> None:  # pragma: no cover
        """Initiate reduction of G and store future to result.

        Note:
            all ranks should enter this function.

        Args:
            group (ProcessGroup): ignored because the correct group depends
                on if the parallelism is on the output or input of the layer.
        """
        if self.primary_rank is None:
            raise RuntimeError('primary rank has not been set yet.')
        valid = (torch.distributed.ProcessGroup, type(None))
        if not isinstance(self.data_parallel_group, valid) or not isinstance(
            self.pipe_parallel_peer_group,
            valid,
        ):
            raise RuntimeError(
                'data_parallel_group or pipe_parallel_peer_group has not '
                'been set yet.',
            )

        if self.parallelism == 'input':
            super().reduce_g_factor(self.pipe_parallel_peer_group)
        elif self.parallelism == 'output':
            if get_rank() != self.primary_rank:
                return
            super().reduce_g_factor(self.data_parallel_group)
        else:
            raise AssertionError('Unreachable.')

    def save_layer_input(
        self,
        input_: list[torch.Tensor],
    ) -> None:  # pragma: no cover
        """Override to gather input to primary rank."""
        if self.primary_rank is None:
            raise RuntimeError('primary rank has not been set yet.')
        if self.parallelism == 'input':
            a = gather_from_model_parallel_region(
                input_[0],
                dst=self.primary_rank,
                model_parallel_group=self.model_parallel_group,
            )
            if a is not None:
                super().save_layer_input([a])
        else:
            super().save_layer_input(input_)

    def save_layer_grad_output(
        self,
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:  # pragma: no cover
        """Override to gather output to primary rank."""
        if self.primary_rank is None:
            raise RuntimeError('primary rank has not been set yet.')
        if self.parallelism == 'output':
            g = gather_from_model_parallel_region(
                grad_output[0],
                dst=self.primary_rank,
                model_parallel_group=self.model_parallel_group,
            )
            if g is not None:
                super().save_layer_grad_output((g,))
        else:
            super().save_layer_grad_output(grad_output)

    def preconditioned_grad(
        self,
        damping: float = 0.001,
    ) -> None:  # pragma: no cover
        """Compute precondition gradient of each weight in module.

        Note:
            Unlike KFACEigenLayer, every rank in the model parallel group
            should enter this function.

        Preconditioned gradients can be applied to the actual gradients with
        `update_gradient()`. Note the steps are separate in the event that
        intermediate steps will be applied to the preconditioned gradient.

        Args:
            damping (float, optional): damping to use if preconditioning using
                the eigendecomposition method (default: 0.001).
        """
        if self.primary_rank is None:
            raise RuntimeError('primary rank has not been set yet.')

        if get_rank() == self.primary_rank and (
            self.qa is None
            or self.qg is None
            or (not self.prediv_eigenvalues and self.da is None)
            or (not self.prediv_eigenvalues and self.dg is None)
            or (self.prediv_eigenvalues and self.dgda is None)
        ):
            raise RuntimeError(
                'Eigendecompositions for both A and G have not been computed',
            )

        grad_partition = self.module.get_weight_grad()
        grad = gather_from_model_parallel_region(
            grad_partition,
            dst=self.primary_rank,
            model_parallel_group=self.model_parallel_group,
            dim=-1 if self.parallelism == 'input' else 0,
        )

        if self.module.has_bias():
            bias_grad_partition = self.module.get_bias_grad()
            # Bias is only actually partitioned if parallelism is done on
            # output
            if self.parallelism == 'output':
                bias_grad = gather_from_model_parallel_region(
                    bias_grad_partition,
                    dst=self.primary_rank,
                    model_parallel_group=self.model_parallel_group,
                    dim=0,
                )
            else:
                bias_grad = bias_grad_partition
        else:
            bias_grad = None

        if grad is not None:
            # Only perform preconditioning on worker that got the full gradient
            grad_shape = grad.size()
            if self.module.has_bias():
                assert bias_grad is not None
                bias_grad_shape = bias_grad.size()
                grad = torch.cat([grad, bias_grad.view(-1, 1)], 1)

            # mypy won't know these are not none because they are properties
            assert self.da is not None
            assert self.dg is not None
            assert self.qa is not None
            assert self.qg is not None

            grad_type = grad.dtype
            grad = grad.to(self.qa.dtype)

            v1 = self.qg.t() @ grad @ self.qa
            if self.prediv_eigenvalues:
                v2 = v1 * self.dgda
            else:
                v2 = v1 / (torch.outer(self.dg, self.da) + damping)

            grad = (self.qg @ v2 @ self.qa.t()).to(grad_type)

            if self.module.has_bias():
                weight_grad = grad[:, :-1].view(grad_shape)
                bias_grad = grad[:, -1:].view(bias_grad_shape).contiguous()
            else:
                weight_grad = grad.view(grad_shape)

            weight_grads = list(
                split_tensor_along_dim(
                    weight_grad,
                    get_world_size(self.model_parallel_group),
                    dim=-1 if self.parallelism == 'input' else 0,
                    contiguous_split_chunks=True,
                ),
            )
            if self.module.has_bias() and self.parallelism == 'output':
                assert bias_grad is not None
                bias_grads = list(
                    split_tensor_along_dim(
                        bias_grad,
                        get_world_size(self.model_parallel_group),
                        dim=0,
                        contiguous_split_chunks=True,
                    ),
                )
        else:
            weight_grads = [
                torch.zeros_like(grad_partition)
                for _ in range(get_world_size(self.model_parallel_group))
            ]
            if self.parallelism == 'output':
                bias_grads = [
                    torch.zeros_like(bias_grad_partition)
                    for _ in range(get_world_size(self.model_parallel_group))
                ]

        # PyTorch NCCL does not support scatter but we can emulate it
        # with reduce_scatter where the reduction operation is sum and the
        # non_src ranks contribute zero filled tensors
        if get_world_size(self.model_parallel_group) > 1:
            torch.distributed.reduce_scatter(
                grad_partition,
                weight_grads,
                group=self.model_parallel_group,
            )
        else:
            grad_partition = weight_grads[0]

        if self.module.has_bias():
            if get_world_size(self.model_parallel_group) > 1:
                if self.parallelism == 'output':
                    torch.distributed.reduce_scatter(
                        bias_grad_partition,
                        bias_grads,
                        group=self.model_parallel_group,
                    )
                    bias_grad = bias_grad_partition
                else:
                    torch.distributed.broadcast(
                        bias_grad,
                        src=self.primary_rank,
                        group=self.model_parallel_group,
                    )
            assert bias_grad is not None
            self.grad = torch.cat([grad_partition, bias_grad.view(-1, 1)], 1)
        else:
            self.grad = grad_partition

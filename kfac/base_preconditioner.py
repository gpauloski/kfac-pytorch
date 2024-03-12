"""Base KFAC preconditioner."""

from __future__ import annotations

import logging
import math
import warnings
from collections import defaultdict
from typing import Any
from typing import Callable

import torch

from kfac.assignment import WorkAssignment
from kfac.distributed import get_rank
from kfac.distributed import TorchDistributedCommunicator
from kfac.layers.base import KFACBaseLayer

logger = logging.getLogger(__name__)


class BaseKFACPreconditioner:
    """Base KFAC Distributed Gradient Preconditioner.

    Preconditions the gradients of a model with a layer-wise FIM approximation.
    Layer computations are distributed across workers using torch.distributed.
    """

    def __init__(
        self,
        layers: dict[torch.nn.Module, tuple[str, KFACBaseLayer]],
        *,
        # Required kwargs
        assignment: WorkAssignment,
        tdc: TorchDistributedCommunicator,
        # KFAC hyperparameters
        factor_update_steps: Callable[[int], int] | int = 1,
        inv_update_steps: Callable[[int], int] | int = 1,
        damping: Callable[[int], float] | float = 0.001,
        factor_decay: Callable[[int], float] | float = 0.95,
        kl_clip: Callable[[int], float] | float = 0.001,
        lr: Callable[[int], float] | float = 0.1,
        # Other
        accumulation_steps: int = 1,
        update_factors_in_hook: bool = True,
        defaults: dict[str, Any] | None = None,
        loglevel: int = logging.DEBUG,
    ) -> None:
        """Init KFACBasePreconditioner.

        Args:
            layers (dict): dict mapping PyTorch modules to tuples of the
                module name and the corresponding KFAC layer.
            assignment (WorkAssignment): assignment initialized with the
                corresponding layers passed in.
            tdc (TorchDistributedCommunicator): communicator instance.
            factor_update_steps (Callable, int): steps between computing and
                updating the running average of the Kronecker factors or
                callable that takes the K-FAC step and returns the value.
            inv_update_steps (Callble, int): steps between recomputing and
                communicating the second-order information or callable that
                takes the K-FAC step and returns the value.
            damping (Callable, float): Tikhonov damping parameter or a callable
                that takes the K-FAC step and returns the damping parameter
                as a float (default: 0.001).
            factor_decay (Callable, float): running average coefficient for
                Kronecker factors or callable that takes the K-FAC step and
                returns the factor_decay (default: 0.95).
            kl_clip (Callable, float): clipping parameter for gradient scaling
                or a callable that takes the K-FAC step and returns a float.
                If None, no scaling/clipping will be applied (default: 0.001).
            lr (Callable, float): learning rate or callable that takes the
                K-FAC step and returns learning rate (default: 0.1).
            accumulation_steps (int): number of forward/backward passes
                between optimization steps (default: 1).
            update_factors_in_hook (bool): If True, running average of factors
                is updated in the module hook and the async communication is
                started. Otherwise, this will be performed at the start of
                step() (default: True).
            defaults (dict): dictionary of default values to include in the
                representation of the preconditioner instance. The default
                values will not be used for anything but are passed by
                subclasses of KFACBasePreconditioner for bookkeeping
                (default: None).
            loglevel (int): logging level (default: logging.DEBUG).
        """
        if not callable(factor_update_steps) and not 0 < factor_update_steps:
            raise ValueError('factor_update_steps must be > 0')
        if not callable(inv_update_steps) and not 0 < inv_update_steps:
            raise ValueError('inv_update_steps must be > 0')
        if not callable(damping) and not 0.0 < damping:
            raise ValueError('damping must be > 0')
        if not callable(factor_decay) and not 0.0 < factor_decay <= 1:
            raise ValueError('factor_decay must be in (0, 1]')
        if not callable(kl_clip) and not 0.0 < kl_clip:
            raise ValueError('kl_clip must be > 0')
        if not callable(lr) and not 0.0 <= lr:
            raise ValueError('lr be > 0')
        if not 0 < accumulation_steps:
            raise ValueError('accumulation_steps must be > 0')
        if (
            not callable(inv_update_steps)
            and not callable(factor_update_steps)
            and not 0 == inv_update_steps % factor_update_steps
        ):
            warnings.warn(
                'It is suggested that inv_update_steps be an integer multiple '
                'of factor_update_steps',
                stacklevel=2,
            )

        self._accumulation_steps = accumulation_steps
        self._assignment = assignment
        self._damping = damping
        self._defaults = defaults
        self._factor_decay = factor_decay
        self._factor_update_steps = factor_update_steps
        self._inv_update_steps = inv_update_steps
        self._kl_clip = kl_clip
        self._layers = layers
        self._loglevel = loglevel
        self._lr = lr
        self._tdc = tdc
        self._update_factors_in_hook = update_factors_in_hook

        self._steps = 0
        # We count the mini_steps (forward/backward passes between optimization
        # steps on a per-layer basis so the key for the dict is the module
        # registered with KFAC
        self._mini_steps: dict[str, int] = defaultdict(int)

        # Register hooks on all modules
        for module in self._layers:
            module.register_forward_pre_hook(self._save_input)
            module.register_full_backward_hook(self._save_grad_output)

    def __repr__(self) -> str:
        """Return representation of the preconditioner instance."""
        params = [
            ('accumulation_steps', self._accumulation_steps),
            ('assignment', self._assignment.__class__.__name__),
            ('damping', self._damping),
            ('factor_decay', self._factor_decay),
            ('factor_update_steps', self._factor_update_steps),
            ('inv_update_steps', self._inv_update_steps),
            ('kl_clip', self._kl_clip),
            ('layers', len(self._layers)),
            ('loglevel', self._loglevel),
            ('lr', self._lr),
            ('steps', self.steps),
            ('update_factors_in_hook', self._update_factors_in_hook),
        ]
        if self._defaults is not None:
            params.extend(list(self._defaults.items()))
        params = sorted(params, key=lambda x: x[0])
        params_joined = [f'  {name}={value},' for name, value in params]
        params_str = '\n'.join(params_joined)
        return f'{self.__class__.__name__}(\n{params_str}\n)'

    @property
    def damping(self) -> float:
        """Get damping value."""
        return (
            self._damping(self.steps)
            if callable(self._damping)
            else self._damping
        )

    @property
    def factor_decay(self) -> float:
        """Get factor decay value."""
        return (
            self._factor_decay(self.steps)
            if callable(self._factor_decay)
            else self._factor_decay
        )

    @property
    def kl_clip(self) -> float:
        """Get kl clip value."""
        return (
            self._kl_clip(self.steps)
            if callable(self._kl_clip)
            else self._kl_clip
        )

    @property
    def lr(self) -> float:
        """Get lr value."""
        return self._lr(self.steps) if callable(self._lr) else self._lr

    @property
    def factor_update_steps(self) -> int:
        """Get factor update steps."""
        return (
            self._factor_update_steps(self.steps)
            if callable(self._factor_update_steps)
            else self._factor_update_steps
        )

    @property
    def inv_update_steps(self) -> int:
        """Get inverse update steps."""
        return (
            self._inv_update_steps(self.steps)
            if callable(self._inv_update_steps)
            else self._inv_update_steps
        )

    @property
    def steps(self) -> int:
        """Get current steps."""
        return self._steps

    def state_dict(self, include_factors: bool = True) -> dict[str, Any]:
        """Returns KFAC state dict.

        Args:
            include_factors (optional, bool): include tensors with factors
                for all registered KFACLayers as a part of the state_dict.
                Note: can make the state_dict fairly large, but not saving the
                factors can cause issues if the first iteration when resuming
                from a checkpoint is not a KFAC update step (default: True).

        Return:
            dict containing state that can be passed to torch.save() or
            load_state_dict().
        """
        state_dict: dict[str, Any] = {'steps': self.steps}
        if not callable(self._factor_update_steps):
            state_dict['factor_update_steps'] = self._factor_update_steps
        if not callable(self._inv_update_steps):
            state_dict['inv_update_steps'] = self._inv_update_steps
        if not callable(self._damping):
            state_dict['damping'] = self._damping
        if not callable(self._factor_decay):
            state_dict['factor_decay'] = self._factor_decay
        if not callable(self._kl_clip):
            state_dict['kl_clip'] = self._kl_clip
        if not callable(self._lr):
            state_dict['lr'] = self._lr
        if include_factors:
            state_dict['layers'] = {
                name: layer.state_dict()
                for name, layer in self._layers.values()
            }
        return state_dict

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        compute_inverses: bool = True,
    ) -> None:
        """Loads the KFAC state.

        Args:
            state_dict (dict): KFAC state. Should be an object returned from a
                call to `state_dict`.
            compute_inverses (bool, optional): if True, compute the inverses
                from the loaded factors. Note that not computing the inverses
                when loading from a checkpoint will only work if the first
                iteration is a KFAC update step (default: True).
        """
        self._steps = state_dict['steps']
        if 'factor_update_steps' in state_dict:
            self._factor_update_steps = state_dict['factor_update_steps']
        if 'inv_update_steps' in state_dict:
            self._inv_update_steps = state_dict['inv_update_steps']
        if 'damping' in state_dict:
            self._damping = state_dict['damping']
        if 'factor_decay' in state_dict:
            self._factor_decay = state_dict['factor_decay']
        if 'kl_clip' in state_dict:
            self._kl_clip = state_dict['kl_clip']
        if 'lr' in state_dict:
            self._lr = state_dict['lr']
        if 'layers' in state_dict:
            if len(state_dict['layers']) != len(self._layers):
                raise ValueError(
                    'loaded state dict contains a different number of layers',
                )
            # This is O(n^2) with respect to number of registered KFAC
            # layers but models typically aren't larger than O(100) layers
            for found_name, layer_state in state_dict['layers'].items():
                for name, layer in self._layers.values():
                    if found_name == name:
                        layer.load_state_dict(layer_state)
        elif compute_inverses:
            warnings.warn(
                'Layer factors are not included in the state_dict so '
                'inverses cannot be computed. Skipping inverse '
                'computation.',
                stacklevel=2,
            )
            compute_inverses = False  # Cannot be computed if no layers
        if compute_inverses:
            for name, layer in self._layers.values():
                layer.compute_a_inv(damping=self.damping)
                layer.compute_g_inv(damping=self.damping)
                if self._assignment.broadcast_inverses():
                    layer.broadcast_a_inv(
                        src=self._assignment.inv_worker(name, 'A'),
                        group=self._assignment.grad_worker_group(name),
                    )
                    layer.broadcast_g_inv(
                        src=self._assignment.inv_worker(name, 'G'),
                        group=self._assignment.grad_worker_group(name),
                    )

    @torch.no_grad()
    def step(self) -> None:
        """Perform one K-FAC step.

        Note:
            This function should always be called before `optimizer.step()` as
            it modifies the gradients and does not modify the weights.

        Note:
            Gradients must be averaged across ranks before calling `step()`.
            This condition is guaranteed to be true if using the
            `DistributedDataParallel` model wrapper as gradients are
            communicated during `loss.backward()`.
        """
        if (
            not self._update_factors_in_hook
            and self.steps % self.factor_update_steps == 0
        ):
            for name, layer in reversed(list(self._layers.values())):
                self._mini_steps[name] = 0
                layer.update_a_factor(alpha=self.factor_decay)
                layer.reduce_a_factor(self._assignment.factor_group(name, 'A'))
                layer.update_g_factor(alpha=self.factor_decay)
                layer.reduce_g_factor(self._assignment.factor_group(name, 'G'))

        # Flush last allreduce bucket from forward/backward pass.
        # Will be a no-op if bucketing was not used
        self._tdc.flush_allreduce_buckets()

        # Compute Inverses
        if self.steps % self.inv_update_steps == 0:
            for name, layer in reversed(list(self._layers.values())):
                if get_rank() == self._assignment.inv_worker(name, 'A'):
                    layer.compute_a_inv(damping=self.damping)
                if (
                    self._assignment.broadcast_inverses()
                    and self._assignment.is_grad_worker(name)
                ):
                    layer.broadcast_a_inv(
                        src=self._assignment.inv_worker(name, 'A'),
                        group=self._assignment.grad_worker_group(name),
                    )
                if get_rank() == self._assignment.inv_worker(name, 'G'):
                    layer.compute_g_inv(damping=self.damping)
                if (
                    self._assignment.broadcast_inverses()
                    and self._assignment.is_grad_worker(name)
                ):
                    layer.broadcast_g_inv(
                        src=self._assignment.inv_worker(name, 'G'),
                        group=self._assignment.grad_worker_group(name),
                    )
            self._tdc.flush_allreduce_buckets()

        # Compute Preconditioned Gradients
        for name, layer in reversed(list(self._layers.values())):
            if self._assignment.is_grad_worker(name):
                layer.preconditioned_grad(damping=self.damping)
            if self._assignment.broadcast_gradients():
                layer.broadcast_grad(
                    src=self._assignment.src_grad_worker(name),
                    group=self._assignment.grad_receiver_group(name),
                )
        self._tdc.flush_allreduce_buckets()

        scale = None if self.kl_clip is None else self._compute_grad_scale()

        # Update gradients in-place
        for _, layer in reversed(list(self._layers.values())):
            layer.update_grad(scale=scale)

        self._steps += 1
        self._mini_steps = defaultdict(int)

    def reset_batch(self) -> None:
        """Reset all KFAC data from last batch."""
        for _, layer in self._layers.values():
            layer.reset_batch()

    def memory_usage(self) -> dict[str, int]:
        """Returns current approximate memory usage for KFAC on this worker.

        Returns:
            dict containing bytes used across all layers on this worker to
            store the factors and second-order information as well as the
            total.
        """
        sizes: dict[str, int] = defaultdict(int)

        # Need to flush buffered communication operations in case user
        # calls this method at a strange time (e.g., in the middle of
        # KFAC.step())
        self._tdc.flush_allreduce_buckets()
        for _, layer in self._layers.values():
            layer_sizes = layer.memory_usage()
            for key, size in layer_sizes.items():
                sizes[key] += size

        sizes['total'] = sum(sizes.values())
        return sizes

    def _compute_grad_scale(self) -> float:
        """Computes scale factor for preconditioned gradients.

        Returns:
            sum_{layers} (sum_{gradients} precon_grad * grad * lr^2)
        """
        vg_sum = 0.0
        for _, layer in reversed(list(self._layers.values())):
            if layer.grad is None:
                raise AssertionError(
                    'layer gradient has not been preconditioned',
                )
            w = layer.module.get_weight_grad()
            if layer.module.has_bias():
                b = layer.module.get_bias_grad()
                v1 = layer.grad[:, :-1].view(w.size())
                v2 = layer.grad[:, -1:].view(b.size())
            else:
                v1 = layer.grad.view(w.size())
            vg_sum += (v1 * w * self.lr**2).sum().item()
            if layer.module.has_bias():
                vg_sum += (v2 * b * self.lr**2).sum().item()
        if vg_sum == 0.0:
            return 1.0
        return min(1.0, math.sqrt(self.kl_clip / abs(vg_sum)))

    @torch.no_grad()
    def _save_input(
        self,
        module: torch.nn.Module,
        input_: list[torch.Tensor],
    ) -> None:
        """Hook for saving the input during the forward pass of a module."""
        if not module.training:
            return
        if self.steps % self.factor_update_steps == 0:
            name, layer = self._layers[module]
            layer.save_layer_input(input_)
            # Update mini_step here because forward pass should always
            # happen before backward pass
            self._mini_steps[name] += 1
            if (
                self._update_factors_in_hook
                and self._mini_steps[name] % self._accumulation_steps == 0
            ):
                layer.update_a_factor(alpha=self.factor_decay)
                layer.reduce_a_factor(self._assignment.factor_group(name, 'A'))

    @torch.no_grad()
    def _save_grad_output(
        self,
        module: torch.nn.Module,
        grad_input: tuple[torch.Tensor, ...] | torch.Tensor,
        grad_output: tuple[torch.Tensor, ...] | torch.Tensor,
    ) -> None:
        """Hook for saving the gradient w.r.t. output in the backward pass."""
        if not module.training:
            return
        if self.steps % self.factor_update_steps == 0:
            name, layer = self._layers[module]
            if isinstance(grad_output, torch.Tensor):
                grad_output = (grad_output,)
            layer.save_layer_grad_output(grad_output)
            if (
                self._update_factors_in_hook
                and self._mini_steps[name] % self._accumulation_steps == 0
            ):
                layer.update_g_factor(alpha=self.factor_decay)
                layer.reduce_g_factor(self._assignment.factor_group(name, 'G'))

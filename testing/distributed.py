"""Decorator for running tests in simulated distributed environments."""

from __future__ import annotations

import multiprocessing
import os
import time
from typing import Any
from typing import Callable
from typing import cast
from typing import TypeVar

import pytest
import torch.distributed as dist

from testing.utils import open_port

# Worker timeout *after* the first worker has completed.
UNIT_WORKER_TIMEOUT = 30

FuncT = TypeVar('FuncT', bound=Callable[..., Any])


def distributed_test(
    world_size: int | list[int] = 2,
) -> Callable[[FuncT], FuncT]:
    """Decorator for running tests in distributed environment.

    A decorator for executing a function (e.g., a unit test) in adistributed
    manner. This decorator manages the spawning and joining of processes,
    initialization of torch.distributed, and catching of errors.

    This function is copied from: https://github.com/EleutherAI/DeeperSpeed/blob/24026e5bb37c528a222b8635c46256b1e1825d2e/tests/unit/common.py#L16

    Example:
        >>> @distributed_test(worker_size=[2,3])
        >>> def my_test():
        >>>     rank = dist.get_rank()
        >>>     world_size = dist.get_world_size()
        >>>     assert(rank < world_size)

    Args:
        world_size (int, list[int]): number of ranks to spawn. Can be a list to
            spawn to run tests multiple times.
    """  # noqa: E501

    port = open_port()

    def dist_wrap(run_func: FuncT) -> FuncT:
        """Second-level decorator that actually wraps the func."""

        def dist_init(
            local_rank: int,
            num_procs: int,
            *func_args: list[Any],
            **func_kwargs: dict[str, Any],
        ) -> None:
            """Initialize torch.distributed and execute the user function."""
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = str(port)
            os.environ['LOCAL_RANK'] = str(local_rank)
            # NOTE: unit tests don't support multi-node so
            # local_rank == global rank
            os.environ['RANK'] = str(local_rank)
            os.environ['WORLD_SIZE'] = str(num_procs)

            dist.init_process_group('gloo')

            run_func(*func_args, **func_kwargs)

            # Keep faster ranks from exiting and breaking process group
            dist.barrier()

        def dist_launcher(
            num_procs: int,
            *func_args: list[Any],
            **func_kwargs: dict[str, Any],
        ) -> None:
            """Launch processes and gracefully handle failures."""
            # Set multiprocessing to use fork because on MacOS/Windows, the
            # default in Python 3.8 and later is "spawn" which cannot
            # pickle lambda functions.
            # NOTE: fork does not work with CUDA tensors but that is okay
            # because the test suite does not use CUDA
            ctx = multiprocessing.get_context('fork')

            # Spawn all workers on subprocesses.
            processes = []
            for local_rank in range(num_procs):
                p = ctx.Process(
                    target=dist_init,
                    args=(local_rank, num_procs, *func_args),
                    kwargs=func_kwargs,
                )
                p.start()
                processes.append(p)

            # Wait for all other processes to complete
            for p in processes:
                p.join(UNIT_WORKER_TIMEOUT)

            failed = [
                (rank, p)
                for rank, p in enumerate(processes)
                if p.exitcode != 0
            ]
            for rank, p in failed:
                # If it still hasn't terminated, kill it because it hung.
                if p.exitcode is None:
                    p.terminate()
                    pytest.fail(f'Worker {rank} hung.', pytrace=False)
                elif p.exitcode < 0:
                    pytest.fail(
                        f'Worker {rank} killed by signal {-p.exitcode}',
                        pytrace=False,
                    )
                elif p.exitcode > 0:
                    pytest.fail(
                        f'Worker {rank} exited with code {p.exitcode}',
                        pytrace=False,
                    )

        def run_func_decorator(
            *func_args: list[Any],
            **func_kwargs: dict[str, Any],
        ) -> Any:
            """Entry point for @distributed_test()."""
            if isinstance(world_size, int):
                dist_launcher(world_size, *func_args, **func_kwargs)
            elif isinstance(world_size, list):
                for procs in world_size:
                    dist_launcher(procs, *func_args, **func_kwargs)
                    time.sleep(0.5)
            else:
                raise TypeError(
                    'world_size must be an integer or a list of integers.',
                )

        return cast(FuncT, run_func_decorator)

    return cast(Callable[[FuncT], FuncT], dist_wrap)

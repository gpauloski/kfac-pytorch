from __future__ import annotations

import time
from typing import Any
from typing import Callable
from typing import TypeVar

import torch

RT = TypeVar('RT')

_func_traces: dict[str, list[float]] = {}


def clear_trace() -> None:
    _func_traces.clear()


def get_trace(
    average: bool = True,
    max_history: int | None = None,
) -> dict[str, float]:
    out = {}
    for fname, times in _func_traces.items():
        if max_history is not None and len(times) > max_history:
            times = times[-max_history:]
        out[fname] = sum(times)
        if average:
            out[fname] /= len(times)
    return out


def print_trace(average: bool = True, max_history: int | None = None) -> None:
    """Print function execution times recorded with @trace

    To trace function execution times, use the @kfac.utils.trace()
    decorator on all functions to be traced. Then to get the average
    execution times, call kfac.utils.print_trace().

    Args:
        average (bool): if true, average the times otherwise print sum of
            times.
        max_history (int, optional): most recent `max_history` times to use
            for average. If None, all are used.
    """
    if len(_func_traces) == 0:
        return
    for fname, times in get_trace(average, max_history).items():
        print(f'{fname}: {times}')


def trace(
    sync: bool = False,
) -> Callable[[Callable[..., RT]], Callable[..., RT]]:
    def decorator(func: Callable[..., RT]) -> Callable[..., RT]:
        def func_timer(*args: list[Any], **kwargs: dict[str, Any]) -> Any:
            if sync:
                torch.distributed.barrier()
            t = time.time()
            out = func(*args, **kwargs)
            if sync:
                torch.distributed.barrier()
            t = time.time() - t

            if func.__name__ not in _func_traces:
                _func_traces[func.__name__] = [t]
            else:
                _func_traces[func.__name__].append(t)
            return out

        return func_timer

    return decorator

import time
import torch

_FUNC_TRACES = {}


def clear_trace():
    _FUNC_TRACES = {}  # noqa: F841


def get_trace(average=True, max_history=None):
    out = {}
    for fname, times in _FUNC_TRACES.items():
        if max_history is not None and len(times) > max_history:
            times = times[-max_history:]
        out[fname] = sum(times)
        if average:
            out[fname] /= len(times)
    return out


def print_trace(average=True, max_history=None):
    """Print function execution times recorded with @trace

    To trace function execution times, use the @kfac.utils.trace()
    decorator on all functions to be traced. Then to get the average
    execution times, call kfac.utils.print_trace().

    Args:
        max_history (int, optional): most recent `max_history` times to use
            for average. If None, all are used.
    """
    if len(_FUNC_TRACES) == 0:
        return
    for fname, times in get_trace(average, max_history).items():
        print("{}: {}".format(fname, times))


def trace(sync=False):
    def decorator(func):
        def func_timer(*args, **kwargs):
            if sync:
                torch.distributed.barrier()
            t = time.time()
            out = func(*args, **kwargs)
            if sync:
                torch.distributed.barrier()
            t = time.time() - t

            if func.__name__ not in _FUNC_TRACES:
                _FUNC_TRACES[func.__name__] = [t]
            else:
                _FUNC_TRACES[func.__name__].append(t)
            return out

        return func_timer

    return decorator

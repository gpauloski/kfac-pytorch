"""Unit Tests for kfac/tracing.py."""

from __future__ import annotations

import time

from kfac.tracing import clear_trace
from kfac.tracing import get_trace
from kfac.tracing import log_trace
from kfac.tracing import trace
from testing.distributed import distributed_test


def test_trace() -> None:
    """Test tracing function execution times."""

    @trace()
    def a(t: float) -> None:
        time.sleep(t)

    @trace()
    def b(t: float) -> None:
        time.sleep(t)

    assert len(get_trace()) == 0
    # Check log raises no errors... we won't bother verifying the output
    log_trace()

    a(0.01)
    traces = get_trace()
    assert len(traces) == 1
    assert 'a' in traces
    assert traces['a'] >= 0.01

    a(0.0)
    new_traces = get_trace()
    assert new_traces['a'] < traces['a']

    b(0.01)
    traces = get_trace()
    assert len(traces) == 2
    assert 'b' in traces

    traces = get_trace(average=False)
    assert traces['a'] > new_traces['a']

    traces = get_trace(average=False, max_history=1)
    assert traces['a'] < 0.01  # should only use the 0 second sleep call
    # Check log raises no errors... we won't bother verifying the output
    log_trace()

    clear_trace()
    assert len(get_trace()) == 0


@distributed_test(world_size=2)
def test_synced_trace() -> None:
    """Test syncing function executions in distributed training."""

    @trace(sync=True)
    def a(t: float) -> None:
        time.sleep(t)

    a(0.01)

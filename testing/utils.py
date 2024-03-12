"""Fixtures and utilities for testing."""

from __future__ import annotations

import socket

_used_ports: set[int] = set()


def open_port() -> int:
    """Return open port.

    Sources:
        https://stackoverflow.com/questions/2838244
        https://github.com/proxystore/proxystore/blob/598b26072784c0d38e034fd8e73ef615a19974a9/testing/utils.py
    """
    while True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
        s.close()
        if port not in _used_ports:  # pragma: no branch
            _used_ports.add(port)
            return port

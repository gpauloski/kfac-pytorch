"""Custom GPT NeoX Eigen Layer."""
from __future__ import annotations

from kfac.layers.eigen import KFACEigenLayer


class GPTNeoXKFACEigenLayer(KFACEigenLayer):
    """Custom Eigen Layer that is model parallel aware."""

    pass

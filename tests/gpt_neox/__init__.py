from __future__ import annotations

import sys

import pytest

# DeepSpeed currently only supports Python 3.10 and older so skip
# this entire test module in Python 3.11 or later
if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
    pytest.skip(
        'DeepSpeed does not support Python 3.11 and later.',
        allow_module_level=True,
    )

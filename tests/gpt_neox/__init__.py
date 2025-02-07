from __future__ import annotations

import sys

import pytest

# DeepSpeed currently only supports Python 3.12 and older so skip
# this entire test module in Python 3.12 or later
if sys.version_info >= (3, 13):  # pragma: >=3.13 cover
    pytest.skip(
        'DeepSpeed does not support Python 3.13 and later.',
        allow_module_level=True,
    )

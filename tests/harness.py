"""Test helpers: gate the model-backed embedding tests on model availability."""

from __future__ import annotations

import functools

from vgi_aisql import models


@functools.cache
def model_available() -> bool:
    """Whether the default fastembed model can be loaded (cached / online)."""
    try:
        models.get_model(None)
    except Exception:  # noqa: BLE001 - any load failure means "skip the model tests"
        return False
    return True

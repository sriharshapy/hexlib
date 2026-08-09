"""Importing this package registers every op definition as a side effect.

The package is created together with its first definition module so it never
exists in an unimportable state.
"""
from __future__ import annotations

from hexlib.graph.opdefs import elementwise, fused, structural  # noqa: F401

__all__ = ["elementwise", "fused", "structural"]

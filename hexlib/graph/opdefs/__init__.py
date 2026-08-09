"""Importing this package registers every op definition as a side effect.

Task 4 adds `structural` to the import below. The package is created together
with its first definition module so it never exists in an unimportable state.
"""
from __future__ import annotations

from hexlib.graph.opdefs import elementwise  # noqa: F401

__all__ = ["elementwise"]

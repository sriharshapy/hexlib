"""The encoder graph IR, op registry, and reference executor.

Everything in this package is pure host computation over plain data: no
hardware, no SDK, no simulator. That is deliberate -- see the spec's
"separate deciding from doing, ruthlessly".
"""
from __future__ import annotations

from hexlib.graph.ir import DTYPES, Graph, Op, Tensor, nbytes

__all__ = ["DTYPES", "Graph", "Op", "Tensor", "nbytes"]

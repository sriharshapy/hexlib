"""Executing a compiled plan.

`interpreter` replays a plan on the host through its own VTCM allocation, with
per-op backends so a real kernel can replace a reference one at a time.
"""
from hexlib.exec.interpreter import Backend, ExecReport, run
from hexlib.exec.vtcm import VtcmError, VtcmImage, overlapping_slots

__all__ = [
    "Backend",
    "ExecReport",
    "run",
    "VtcmError",
    "VtcmImage",
    "overlapping_slots",
]

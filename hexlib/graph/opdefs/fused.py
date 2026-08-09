"""The fused matmul epilogue.

A real registry entry with a real reference, not an annotation. That is what
lets the M0 eager oracle run a FUSED graph and prove fusion did not change the
answer -- an annotation-only fusion pass could not be checked that way.
"""
from __future__ import annotations

import numpy as np

from hexlib.graph.ops import OpDef, get, register

ACTIVATIONS = ("none", "gelu_tanh", "gelu_erf")


def _infer(inputs, attrs):
    if attrs.get("act", "none") not in ACTIVATIONS:
        raise ValueError(
            f"matmul_epilogue act must be one of {ACTIVATIONS}, got {attrs.get('act')!r}"
        )
    a, b, bias = inputs[0], inputs[1], inputs[2]
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(
            f"matmul_epilogue inner dimensions disagree: {a.name} {a.shape} vs "
            f"{b.name} {b.shape}"
        )
    if bias.shape != (b.shape[-1],):
        raise ValueError(
            f"matmul_epilogue bias {bias.name} is {bias.shape}, expected "
            f"({b.shape[-1]},)"
        )
    batch = tuple(np.broadcast_shapes(a.shape[:-2], b.shape[:-2]))
    return ((batch + (a.shape[-2], b.shape[-1]), a.dtype),)


def _working_set(inputs, outputs, attrs) -> int:
    return sum(t.nbytes for t in inputs) + sum(t.nbytes for t in outputs)


def _reference(arrays, attrs):
    # Cast back to the activation's dtype: numpy promotes, so fp16 @ fp32 + fp32
    # would return fp32 and contradict what _infer declares for this same op.
    # Every reference honours its own infer -- see Global Constraints.
    out = (arrays[0] @ arrays[1] + arrays[2]).astype(arrays[0].dtype)
    act = attrs.get("act", "none")
    if act == "none":
        return (out,)
    # Delegate to the unfused definition so the two can never disagree.
    return get(act).reference((out,), {})


register(
    OpDef(
        kind="matmul_epilogue",
        infer=_infer,
        working_set=_working_set,
        reference=_reference,
    )
)

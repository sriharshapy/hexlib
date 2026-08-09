"""Elementwise and normalisation op definitions.

Formulas are the ones the target model actually uses, cited to the file and
line they were read from. In particular gelu_tanh and gelu_erf are two
different functions used in two different places, not one op with a flag.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from hexlib.graph.ir import Tensor, nbytes
from hexlib.graph.ops import OpDef, register

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)

# math.erf is scalar-only and there is no numpy erf. The oracle is allowed to
# be slow -- spec 7: "The eager executor is the oracle and must stay obviously
# correct. It is never optimized."
_erf = np.vectorize(math.erf, otypes=[np.float64])


def _sum_bytes(tensors: Sequence[Tensor]) -> int:
    return sum(t.nbytes for t in tensors)


def _elementwise_working_set(inputs, outputs, attrs) -> int:
    """Every input and every output must be resident."""
    return _sum_bytes(inputs) + _sum_bytes(outputs)


def _broadcast_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(np.broadcast_shapes(a, b))


# --- add ---------------------------------------------------------------

register(
    OpDef(
        kind="add",
        infer=lambda inputs, attrs: (
            (_broadcast_shape(inputs[0].shape, inputs[1].shape), inputs[0].dtype),
        ),
        working_set=_elementwise_working_set,
        reference=lambda arrays, attrs: (arrays[0] + arrays[1],),
    )
)

# --- scale -------------------------------------------------------------

register(
    OpDef(
        kind="scale",
        infer=lambda inputs, attrs: ((inputs[0].shape, inputs[0].dtype),),
        working_set=_elementwise_working_set,
        reference=lambda arrays, attrs: (arrays[0] * attrs["factor"],),
    )
)

# --- layernorm ---------------------------------------------------------


def _layernorm_reference(arrays, attrs):
    """torch.nn.LayerNorm over the last axis: biased variance, eps inside the sqrt.

    The vision tower uses nn.LayerNorm(hidden_size, eps=1e-6) with the default
    elementwise_affine=True, so there is a learned weight AND a learned bias --
    modeling_qwen3_5.py:991-992 (blocks) and :880 (merger). The text tower uses
    RMSNorm instead; do not conflate them.
    """
    x, weight, bias = arrays[0].astype(np.float64), arrays[1], arrays[2]
    mean = x.mean(axis=-1, keepdims=True)
    centred = x - mean
    var = np.mean(centred * centred, axis=-1, keepdims=True)
    normed = centred / np.sqrt(var + attrs["eps"])
    return ((normed * weight + bias).astype(arrays[0].dtype),)


register(
    OpDef(
        kind="layernorm",
        infer=lambda inputs, attrs: ((inputs[0].shape, inputs[0].dtype),),
        working_set=_elementwise_working_set,
        reference=_layernorm_reference,
    )
)

# --- activations -------------------------------------------------------


def _gelu_tanh_reference(arrays, attrs):
    """ACT2FN["gelu_pytorch_tanh"] -- the blocks' MLP, modeling_qwen3_5.py:849."""
    x = arrays[0].astype(np.float64)
    inner = _SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)
    return ((0.5 * x * (1.0 + np.tanh(inner))).astype(arrays[0].dtype),)


def _gelu_erf_reference(arrays, attrs):
    """Plain nn.GELU() -- the merger, modeling_qwen3_5.py:882.

    No `approximate` argument is passed there, so PyTorch's default
    approximate='none' applies: the exact erf form, NOT the tanh
    approximation the blocks use.
    """
    x = arrays[0].astype(np.float64)
    return ((0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))).astype(arrays[0].dtype),)


for _kind, _fn in (("gelu_tanh", _gelu_tanh_reference), ("gelu_erf", _gelu_erf_reference)):
    register(
        OpDef(
            kind=_kind,
            infer=lambda inputs, attrs: ((inputs[0].shape, inputs[0].dtype),),
            working_set=_elementwise_working_set,
            reference=_fn,
        )
    )

# --- softmax -----------------------------------------------------------


def _softmax_reference(arrays, attrs):
    axis = attrs.get("axis", -1)
    x = arrays[0].astype(np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return ((e / np.sum(e, axis=axis, keepdims=True)).astype(arrays[0].dtype),)


register(
    OpDef(
        kind="softmax",
        infer=lambda inputs, attrs: ((inputs[0].shape, inputs[0].dtype),),
        working_set=_elementwise_working_set,
        reference=_softmax_reference,
    )
)

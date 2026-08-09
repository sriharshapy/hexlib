from __future__ import annotations

import math

import numpy as np
import pytest

import hexlib.graph.opdefs  # noqa: F401  -- registers everything
from hexlib.graph.ops import get
from hexlib.graph.ir import Tensor


def _ref(kind, arrays, attrs=None):
    return get(kind).reference(tuple(arrays), attrs or {})


def test_add_broadcasts_a_bias_row():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    (out,) = _ref("add", (a, b))
    np.testing.assert_allclose(out, [[10, 21, 32], [13, 24, 35]])


def test_add_preserves_first_input_dtype_with_mixed_inputs():
    # The encoder's bias adds: fp16 activations + fp32 biases -> fp16.
    # Output dtype must match the first input (fp16), not numpy's promotion rules.
    a = np.array([1.0, 2.0], dtype=np.float16)
    b = np.array([0.5, 0.5], dtype=np.float32)
    (out,) = _ref("add", (a, b))
    assert out.dtype == np.float16, f"Expected fp16, got {out.dtype}"
    np.testing.assert_allclose(out, [1.5, 2.5], rtol=1e-3)


def test_add_infers_broadcast_shape():
    shapes = get("add").infer(
        (Tensor("a", "fp32", (2, 3)), Tensor("b", "fp32", (3,))), {}
    )
    assert shapes == (((2, 3), "fp32"),)


def test_scale_multiplies():
    (out,) = _ref("scale", (np.array([1.0, 2.0], dtype=np.float32),), {"factor": 0.125})
    np.testing.assert_allclose(out, [0.125, 0.25])


def test_layernorm_matches_the_closed_form():
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    w = np.ones(4, dtype=np.float32)
    b = np.zeros(4, dtype=np.float32)
    (out,) = _ref("layernorm", (x, w, b), {"eps": 1e-6})
    mean = 2.5
    var = np.mean((x - mean) ** 2)  # biased, as torch.nn.LayerNorm uses
    expected = (x - mean) / np.sqrt(var + 1e-6)
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_layernorm_applies_weight_and_bias():
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    w = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    b = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    (plain,) = _ref("layernorm", (x, np.ones(4, np.float32), np.zeros(4, np.float32)), {"eps": 1e-6})
    (out,) = _ref("layernorm", (x, w, b), {"eps": 1e-6})
    np.testing.assert_allclose(out, plain * 2.0 + 1.0, rtol=1e-6, atol=1e-6)


def test_layernorm_eps_is_inside_the_sqrt():
    # A constant row has zero variance. With eps inside the sqrt the result is
    # exactly zero and finite; a wrong placement gives inf or nan.
    x = np.full((1, 8), 3.0, dtype=np.float32)
    (out,) = _ref("layernorm", (x, np.ones(8, np.float32), np.zeros(8, np.float32)), {"eps": 1e-6})
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, 0.0, atol=1e-6)


def test_gelu_tanh_matches_the_published_formula():
    x = np.linspace(-3, 3, 13).astype(np.float32)
    (out,) = _ref("gelu_tanh", (x,))
    expected = 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_gelu_erf_matches_the_exact_formula():
    x = np.linspace(-3, 3, 13).astype(np.float32)
    (out,) = _ref("gelu_erf", (x,))
    expected = np.array([0.5 * v * (1 + math.erf(v / math.sqrt(2))) for v in x])
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_gelu_tanh_and_gelu_erf_are_actually_different_ops():
    # If these ever agree to within 1e-9 someone has aliased one to the other.
    # The merger uses erf and the blocks use tanh; conflating them is a silent
    # numerical error, which is why they are separate kinds.
    x = np.linspace(-3, 3, 101).astype(np.float64)
    (a,) = _ref("gelu_tanh", (x,))
    (b,) = _ref("gelu_erf", (x,))
    assert np.max(np.abs(a - b)) > 1e-5


def test_softmax_rows_sum_to_one():
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    (out,) = _ref("softmax", (x,), {"axis": -1})
    np.testing.assert_allclose(out.sum(axis=-1), [1.0, 1.0], rtol=1e-6)


def test_softmax_is_stable_on_large_inputs():
    x = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
    (out,) = _ref("softmax", (x,), {"axis": -1})
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out.sum(), 1.0, rtol=1e-6)


@pytest.mark.parametrize("kind", ["add", "scale", "layernorm", "gelu_tanh", "gelu_erf", "softmax"])
def test_working_set_is_a_positive_int(kind):
    ins = (Tensor("a", "fp16", (8, 32)), Tensor("b", "fp16", (32,)), Tensor("c", "fp16", (32,)))
    outs = (Tensor("o", "fp16", (8, 32)),)
    ws = get(kind).working_set(ins, outs, {"eps": 1e-6, "factor": 1.0, "axis": -1})
    assert isinstance(ws, int) and ws > 0


def test_layernorm_infer_preserves_shape_and_dtype():
    shapes = get("layernorm").infer(
        (Tensor("x", "fp16", (8, 32)), Tensor("w", "fp32", (32,)), Tensor("b", "fp32", (32,))),
        {"eps": 1e-6},
    )
    assert shapes == (((8, 32), "fp16"),)

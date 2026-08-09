"""Finding 8 (final whole-branch review): make the dtype rule structural.

"Every `OpDef.reference` returns the dtype its own `infer` declares" was
violated three times in M0 (`add`, `scale`, `matmul`) before being promoted to
a written rule, and enforced only by three ad-hoc, per-op tests. This is the
one test that checks the rule itself, over every registered op kind, so a
future op cannot ship the same bug a fourth time without a hand-written test
for that specific op.

Inputs are deliberately built with mixed dtypes (a fp16 activation alongside
fp32 bias/weight/table inputs) wherever an op takes more than one input --
that is the exact shape of the historical bug: numpy's own promotion rules
silently widen `fp16 @ fp32` to `fp32` unless `reference` calls `.astype()`
explicitly. A test where every input shares one dtype cannot exercise that
promotion at all and would prove nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

import hexlib.graph.opdefs  # noqa: F401  -- registers everything
from hexlib.graph.eager import NUMPY_DTYPE as _EAGER_NUMPY_DTYPE
from hexlib.graph.ir import Tensor
from hexlib.graph.ops import all_kinds, get

# The "physical" numpy dtype a hexlib dtype string means when a test
# constructs a real array by hand. Every op EXCEPT `cast` derives its output's
# real numpy dtype from one of its own input arrays (`.astype(arrays[i].dtype)`),
# so the array a test hands it must carry the real numpy dtype its Tensor
# declaration claims, or the check below proves nothing.
_PHYSICAL_DTYPE = {"fp32": np.float32, "fp16": np.float16, "int32": np.int32}

# `cast` is the one op whose output dtype is NOT "whatever numpy dtype one of
# its inputs already has" -- it converts to the numpy dtype `eager.NUMPY_DTYPE`
# maps its target hexlib dtype string to (finding 1's fix), exactly mirroring
# the coercion `eager.run` performs at every op boundary. `NUMPY_DTYPE["fp16"]`
# is `float32` (fp16 never physically materializes in this oracle), so `cast`
# must be checked against that mapping, not the physical one above.
_KINDS_USING_EAGER_DTYPE_MAP = frozenset({"cast"})


def _expected_numpy_dtype(kind: str, declared: str) -> np.dtype:
    table = _EAGER_NUMPY_DTYPE if kind in _KINDS_USING_EAGER_DTYPE_MAP else _PHYSICAL_DTYPE
    return np.dtype(table[declared])


def _case(tensors, arrays, attrs=None):
    return (tensors, arrays, attrs or {})


# kind -> (input Tensors, input arrays, attrs). One valid, callable case per
# registered op kind -- shapes and attrs are the smallest that satisfy each
# op's own `infer` validation.
_CASES = {
    "add": _case(
        (Tensor("x", "fp16", (2, 3)), Tensor("b", "fp32", (3,))),
        (np.arange(6, dtype=np.float16).reshape(2, 3), np.array([1.0, 2.0, 3.0], dtype=np.float32)),
    ),
    "cast": _case(
        (Tensor("x", "fp32", (2, 3)),),
        (np.arange(6, dtype=np.float32).reshape(2, 3),),
        {"dtype": "fp16"},
    ),
    "gelu_erf": _case(
        (Tensor("x", "fp16", (4,)),),
        (np.linspace(-1, 1, 4).astype(np.float16),),
    ),
    "gelu_tanh": _case(
        (Tensor("x", "fp16", (4,)),),
        (np.linspace(-1, 1, 4).astype(np.float16),),
    ),
    "layernorm": _case(
        (Tensor("x", "fp16", (2, 4)), Tensor("w", "fp32", (4,)), Tensor("b", "fp32", (4,))),
        (
            np.arange(8, dtype=np.float16).reshape(2, 4),
            np.ones(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        ),
        {"eps": 1e-6},
    ),
    "matmul": _case(
        (Tensor("a", "fp16", (2, 3)), Tensor("b", "fp32", (3, 4))),
        (np.arange(6, dtype=np.float16).reshape(2, 3), np.arange(12, dtype=np.float32).reshape(3, 4)),
    ),
    "matmul_epilogue": _case(
        (
            Tensor("a", "fp16", (2, 3)),
            Tensor("b", "q4_0", (3, 4)),
            Tensor("bias", "fp32", (4,)),
        ),
        (
            np.arange(6, dtype=np.float16).reshape(2, 3),
            np.arange(12, dtype=np.float32).reshape(3, 4),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        ),
        {"act": "none"},
    ),
    "patchify": _case(
        (Tensor("img", "fp32", (1, 1, 4, 4)),),
        (np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4),),
        {"patch": 2, "temporal_patch": 1, "merge": 1, "grid_h": 2, "grid_w": 2},
    ),
    "reshape": _case(
        (Tensor("x", "fp16", (2, 3)),),
        (np.arange(6, dtype=np.float16).reshape(2, 3),),
        {"shape": (3, 2)},
    ),
    "rope_2d": _case(
        (Tensor("x", "fp16", (2, 1, 4)), Tensor("c", "fp32", (2, 4)), Tensor("s", "fp32", (2, 4))),
        (
            np.arange(8, dtype=np.float16).reshape(2, 1, 4),
            np.ones((2, 4), dtype=np.float32),
            np.zeros((2, 4), dtype=np.float32),
        ),
    ),
    "scale": _case(
        (Tensor("x", "fp16", (3,)),),
        (np.arange(3, dtype=np.float16),),
        {"factor": 0.5},
    ),
    "softmax": _case(
        (Tensor("x", "fp16", (2, 3)),),
        (np.arange(6, dtype=np.float16).reshape(2, 3),),
        {"axis": -1},
    ),
    "transpose": _case(
        (Tensor("x", "fp16", (2, 3, 4)),),
        (np.arange(24, dtype=np.float16).reshape(2, 3, 4),),
        {"perm": (1, 0, 2)},
    ),
}


def test_case_table_covers_every_registered_kind():
    # Fail closed: a kind added to the registry without a matching entry here
    # must not let the parametrized test below silently skip it.
    known = set(all_kinds())
    assert known, "op registry is empty; this test would pass vacuously"
    assert known == set(_CASES), (
        f"kind/table mismatch: registry has {sorted(known - set(_CASES))} that "
        f"the table doesn't cover, table has {sorted(set(_CASES) - known)} that "
        "the registry doesn't know"
    )


@pytest.mark.parametrize("kind", sorted(_CASES))
def test_reference_dtype_matches_what_infer_declared(kind):
    tensors, arrays, attrs = _CASES[kind]
    opdef = get(kind)

    declared = opdef.infer(tensors, attrs)
    results = opdef.reference(arrays, attrs)

    assert isinstance(results, tuple)
    assert len(results) == len(declared), (
        f"{kind}: infer declares {len(declared)} output(s) but reference returned {len(results)}"
    )
    for (shape, dtype), value in zip(declared, results):
        value = np.asarray(value)
        expected = _expected_numpy_dtype(kind, dtype)
        assert value.dtype == expected, (
            f"{kind}: infer declared dtype {dtype!r} ({expected}) but reference "
            f"returned {value.dtype}"
        )
        assert tuple(value.shape) == shape, (
            f"{kind}: infer declared shape {shape} but reference returned {value.shape}"
        )

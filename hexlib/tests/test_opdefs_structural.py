from __future__ import annotations

import numpy as np
import pytest

import hexlib.graph.opdefs  # noqa: F401
from hexlib.graph.ops import get
from hexlib.graph.ir import Tensor


def _ref(kind, arrays, attrs=None):
    return get(kind).reference(tuple(arrays), attrs or {})


def test_matmul_2d():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.arange(12, dtype=np.float32).reshape(3, 4)
    (out,) = _ref("matmul", (a, b))
    np.testing.assert_allclose(out, a @ b)


def test_matmul_batched():
    a = np.random.RandomState(0).randn(5, 2, 3).astype(np.float32)
    b = np.random.RandomState(1).randn(5, 3, 4).astype(np.float32)
    (out,) = _ref("matmul", (a, b))
    np.testing.assert_allclose(out, a @ b, rtol=1e-5)


def test_matmul_with_mixed_dtypes_returns_activation_dtype():
    # Weights (second input) are fp32 but activation is fp16. The output must be
    # fp16, not promoted to fp32. This catches numpy's dtype promotion.
    a = np.arange(6, dtype=np.float16).reshape(2, 3)
    b = np.arange(12, dtype=np.float32).reshape(3, 4)
    (out,) = _ref("matmul", (a, b))
    assert out.dtype == np.float16, f"expected fp16 but got {out.dtype}"
    np.testing.assert_allclose(out, a.astype(np.float32) @ b, rtol=1e-2)


def test_matmul_infer_2d():
    shapes = get("matmul").infer(
        (Tensor("a", "fp16", (8, 768)), Tensor("b", "q4_0", (768, 3072))), {}
    )
    assert shapes == (((8, 3072), "fp16"),)


def test_matmul_infer_takes_dtype_from_the_activation_not_the_weight():
    # Weights are q4_0 and dequantized in-kernel; the output is an activation.
    shapes = get("matmul").infer(
        (Tensor("a", "fp16", (8, 768)), Tensor("w", "q4_0", (768, 768))), {}
    )
    assert shapes[0][1] == "fp16"


def test_matmul_infer_batched():
    shapes = get("matmul").infer(
        (Tensor("a", "fp16", (12, 64, 64)), Tensor("b", "fp16", (12, 64, 32))), {}
    )
    assert shapes == (((12, 64, 32), "fp16"),)


def test_matmul_infer_rejects_inner_dim_mismatch():
    with pytest.raises(ValueError) as e:
        get("matmul").infer((Tensor("a", "fp16", (8, 768)), Tensor("b", "fp16", (512, 32))), {})
    assert "768" in str(e.value) and "512" in str(e.value)


def test_transpose():
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    (out,) = _ref("transpose", (x,), {"perm": (1, 0, 2)})
    np.testing.assert_allclose(out, x.transpose(1, 0, 2))
    shapes = get("transpose").infer((Tensor("x", "fp16", (2, 3, 4)),), {"perm": (1, 0, 2)})
    assert shapes == (((3, 2, 4), "fp16"),)


def test_reshape():
    x = np.arange(24, dtype=np.float32).reshape(2, 12)
    (out,) = _ref("reshape", (x,), {"shape": (6, 4)})
    assert out.shape == (6, 4)
    np.testing.assert_allclose(out.ravel(), x.ravel())


def test_reshape_infer_rejects_element_count_change():
    with pytest.raises(ValueError) as e:
        get("reshape").infer((Tensor("x", "fp16", (2, 12)),), {"shape": (5, 5)})
    assert "24" in str(e.value) and "25" in str(e.value)


def test_patchify_emits_merge_block_order_not_raster_order():
    # 4x4 patch grid, merge 2 -> 4 blocks of 4. Give every patch a unique
    # constant value equal to its raster index, then check the token order.
    patch, tp, c, merge = 1, 1, 1, 2
    grid_h = grid_w = 4
    image = np.arange(grid_h * grid_w, dtype=np.float32).reshape(1, 1, grid_h, grid_w)
    (out,) = _ref(
        "patchify",
        (image,),
        {"patch": patch, "temporal_patch": tp, "merge": merge,
         "grid_h": grid_h, "grid_w": grid_w},
    )
    assert out.shape == (16, 1)
    order = out[:, 0].astype(int).tolist()
    # Block (0,0) is raster patches 0,1,4,5; block (0,1) is 2,3,6,7; etc.
    assert order == [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]


def test_patchify_feature_order_is_c_t_ph_pw():
    # Conv3d weight is [embed, C, T, ph, pw] (modeling_qwen3_5.py:864), so the
    # flattened per-patch feature order must be (C, T, ph, pw).
    c, tp, patch = 2, 2, 2
    grid_h = grid_w = 2
    image = np.arange(c * tp * (grid_h * patch) * (grid_w * patch), dtype=np.float32)
    image = image.reshape(c, tp, grid_h * patch, grid_w * patch)
    (out,) = _ref(
        "patchify",
        (image,),
        {"patch": patch, "temporal_patch": tp, "merge": 2,
         "grid_h": grid_h, "grid_w": grid_w},
    )
    assert out.shape == (4, c * tp * patch * patch)
    # First token is block (0,0) position (0,0) -> raster patch 0 -> rows 0:2, cols 0:2.
    expected = image[:, :, 0:2, 0:2].reshape(-1)
    np.testing.assert_allclose(out[0], expected)


def test_patchify_infer():
    shapes = get("patchify").infer(
        (Tensor("img", "fp32", (3, 2, 32, 32)),),
        {"patch": 16, "temporal_patch": 2, "merge": 2, "grid_h": 2, "grid_w": 2},
    )
    assert shapes == (((4, 3 * 2 * 16 * 16), "fp32"),)


def test_patchify_infer_rejects_a_grid_not_divisible_by_merge():
    # image is 3x2x48x32, tiling correctly into a 3x2 patch grid at patch=16 --
    # but grid_h=3 is not divisible by merge=2, so merge-block ordering is
    # undefined.
    with pytest.raises(ValueError) as e:
        get("patchify").infer(
            (Tensor("img", "fp32", (3, 2, 48, 32)),),
            {"patch": 16, "temporal_patch": 2, "merge": 2, "grid_h": 3, "grid_w": 2},
        )
    assert "merge" in str(e.value)
    assert "3" in str(e.value)


def test_patchify_infer_rejects_a_temporal_patch_mismatch():
    with pytest.raises(ValueError) as e:
        get("patchify").infer(
            (Tensor("img", "fp32", (3, 2, 32, 32)),),
            {"patch": 16, "temporal_patch": 5, "merge": 2, "grid_h": 2, "grid_w": 2},
        )
    assert "temporal_patch" in str(e.value)
    assert "5" in str(e.value)


def test_rope_2d_matches_the_reference_formula():
    rs = np.random.RandomState(7)
    n, heads, d = 6, 3, 8
    x = rs.randn(n, heads, d).astype(np.float32)
    cos = rs.randn(n, d).astype(np.float32)
    sin = rs.randn(n, d).astype(np.float32)
    (out,) = _ref("rope_2d", (x, cos, sin))

    def rotate_half(a):
        h = a.shape[-1] // 2
        return np.concatenate([-a[..., h:], a[..., :h]], axis=-1)

    expected = x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_rope_2d_with_identity_tables_is_the_identity():
    x = np.random.RandomState(3).randn(4, 2, 8).astype(np.float32)
    cos = np.ones((4, 8), dtype=np.float32)
    sin = np.zeros((4, 8), dtype=np.float32)
    (out,) = _ref("rope_2d", (x, cos, sin))
    np.testing.assert_allclose(out, x, rtol=1e-6, atol=1e-6)


def test_rope_2d_infer_preserves_shape():
    shapes = get("rope_2d").infer(
        (Tensor("q", "fp16", (64, 12, 64)), Tensor("c", "fp32", (64, 64)),
         Tensor("s", "fp32", (64, 64))),
        {},
    )
    assert shapes == (((64, 12, 64), "fp16"),)


def test_rope_2d_infer_rejects_cos_sin_shape_mismatch():
    with pytest.raises(ValueError) as e:
        get("rope_2d").infer(
            (Tensor("q", "fp16", (64, 12, 64)), Tensor("c", "fp32", (64, 64)),
             Tensor("s", "fp32", (64, 32))),
            {},
        )
    assert "cos" in str(e.value) and "sin" in str(e.value)


def test_rope_2d_infer_rejects_table_shape_mismatch_with_x():
    # cos/sin agree with each other but not with (tokens, head_dim) from x.
    with pytest.raises(ValueError) as e:
        get("rope_2d").infer(
            (Tensor("q", "fp16", (64, 12, 64)), Tensor("c", "fp32", (32, 64)),
             Tensor("s", "fp32", (32, 64))),
            {},
        )
    assert "64" in str(e.value)


@pytest.mark.parametrize("kind", ["matmul", "transpose", "reshape", "rope_2d"])
def test_working_set_is_a_positive_int(kind):
    ins = (Tensor("a", "fp16", (8, 32)), Tensor("b", "fp16", (32, 16)),
           Tensor("c", "fp16", (8, 32)))
    outs = (Tensor("o", "fp16", (8, 16)),)
    ws = get(kind).working_set(ins, outs, {"perm": (1, 0), "shape": (32, 8)})
    assert isinstance(ws, int) and ws > 0

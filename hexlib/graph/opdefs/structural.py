"""Structural op definitions: matmul, layout moves, patchify, 2D RoPE.

Weights arrive as [k, n] -- already transposed from PyTorch's Linear.weight
[out, in]. That transpose happens once when the graph is built, so no runtime
transpose op ever appears for a projection.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from hexlib.graph.ir import Tensor
from hexlib.graph.ops import OpDef, register


def _sum_bytes(tensors: Sequence[Tensor]) -> int:
    return sum(t.nbytes for t in tensors)


def _default_working_set(inputs, outputs, attrs) -> int:
    return _sum_bytes(inputs) + _sum_bytes(outputs)


# --- matmul ------------------------------------------------------------


def _matmul_infer(inputs, attrs):
    a, b = inputs[0], inputs[1]
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ValueError(
            f"matmul needs 2-D or higher operands, got {a.shape} and {b.shape}"
        )
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(
            f"matmul inner dimensions disagree: {a.name} is {a.shape} (inner "
            f"{a.shape[-1]}) but {b.name} is {b.shape} (inner {b.shape[-2]})"
        )
    batch = tuple(np.broadcast_shapes(a.shape[:-2], b.shape[:-2]))
    # The output is an activation, so it takes the activation's dtype. Quantized
    # weights are dequantized in-kernel and never determine the output dtype.
    return ((batch + (a.shape[-2], b.shape[-1]), a.dtype),)


register(
    OpDef(
        kind="matmul",
        infer=_matmul_infer,
        working_set=_default_working_set,
        reference=lambda arrays, attrs: ((arrays[0] @ arrays[1]).astype(arrays[0].dtype),),
    )
)

# --- transpose ---------------------------------------------------------


def _transpose_infer(inputs, attrs):
    perm = tuple(attrs["perm"])
    shape = inputs[0].shape
    if sorted(perm) != list(range(len(shape))):
        raise ValueError(
            f"transpose perm {perm} is not a permutation of axes 0..{len(shape) - 1}"
        )
    return ((tuple(shape[i] for i in perm), inputs[0].dtype),)


register(
    OpDef(
        kind="transpose",
        infer=_transpose_infer,
        working_set=_default_working_set,
        reference=lambda arrays, attrs: (np.transpose(arrays[0], tuple(attrs["perm"])),),
    )
)

# --- reshape -----------------------------------------------------------


def _reshape_infer(inputs, attrs):
    shape = tuple(attrs["shape"])
    have = math.prod(inputs[0].shape)
    want = math.prod(shape)
    if have != want:
        raise ValueError(
            f"reshape of {inputs[0].name} from {inputs[0].shape} ({have} elements) "
            f"to {shape} ({want} elements) changes the element count"
        )
    return ((shape, inputs[0].dtype),)


register(
    OpDef(
        kind="reshape",
        infer=_reshape_infer,
        working_set=_default_working_set,
        reference=lambda arrays, attrs: (
            arrays[0].reshape(tuple(attrs["shape"])),
        ),
    )
)

# --- patchify ----------------------------------------------------------


def _patchify_infer(inputs, attrs):
    img = inputs[0]
    if len(img.shape) != 4:
        raise ValueError(
            f"patchify expects a [C, T, H, W] image, got {img.name} with shape {img.shape}"
        )
    c, t, h, w = img.shape
    patch = int(attrs["patch"])
    merge = int(attrs["merge"])
    grid_h, grid_w = int(attrs["grid_h"]), int(attrs["grid_w"])
    if t != int(attrs["temporal_patch"]):
        raise ValueError(
            f"patchify: image T is {t} but temporal_patch is {attrs['temporal_patch']}"
        )
    if grid_h * patch != h or grid_w * patch != w:
        raise ValueError(
            f"patchify: {h}x{w} image does not tile into {grid_h}x{grid_w} patches "
            f"of size {patch}"
        )
    if grid_h % merge or grid_w % merge:
        raise ValueError(
            f"patchify: patch grid {grid_h}x{grid_w} is not divisible by merge {merge}; "
            "merge-block ordering would be undefined"
        )
    return (((grid_h * grid_w, c * t * patch * patch), img.dtype),)


def _patchify_reference(arrays, attrs):
    """[C, T, H, W] -> [grid_h*grid_w, C*T*patch*patch] in merge-block order.

    Merge-block order, not raster order: the patch merger is a pure reshape
    (modeling_qwen3_5.py:886), so consecutive runs of merge*merge tokens must
    already BE the 2x2 spatial blocks. transformers/vision_utils.py:76-81
    establishes the same ordering for the RoPE position ids with
    reshape(h//m, m, w//m, m).transpose(1, 2).flatten().

    Per-patch feature order is (C, T, ph, pw), matching the Conv3d weight
    layout [embed, C, T, ph, pw] at modeling_qwen3_5.py:864.

    This ordering was independently confirmed against the actual image
    processor qwen3_5 declares: `transformers/models/auto/image_processing_auto.py:128`
    maps `"qwen3_5"` to `Qwen2VLImageProcessor`
    (`transformers/models/qwen2_vl/image_processing_qwen2_vl.py:196-218`,
    `Qwen2VLImageProcessor._preprocess`), whose own
    `reshape(...).permute(0, 2, 5, 3, 6, 1, 4, 7)` chain produces the same
    (grid_h/merge, grid_w/merge, merge_h, merge_w) token order and (channel,
    T, patch_h, patch_w) feature order implemented below -- see
    `scripts/gen_vision_oracle.py::_patches_from_image` for the full citation
    and the independent numpy re-derivation this was checked against.
    """
    img = arrays[0]
    c, t, h, w = img.shape
    patch = int(attrs["patch"])
    merge = int(attrs["merge"])
    grid_h, grid_w = int(attrs["grid_h"]), int(attrs["grid_w"])

    # Split H and W into patch grid and within-patch offsets.
    x = img.reshape(c, t, grid_h, patch, grid_w, patch)
    # Split the patch grid into merge blocks: (bh, mh, bw, mw).
    x = x.reshape(c, t, grid_h // merge, merge, patch, grid_w // merge, merge, patch)
    # Token axes in merge-block order: bh, bw, mh, mw. Feature axes: c, t, ph, pw.
    x = x.transpose(2, 5, 3, 6, 0, 1, 4, 7)
    return (x.reshape(grid_h * grid_w, c * t * patch * patch),)


register(
    OpDef(
        kind="patchify",
        infer=_patchify_infer,
        working_set=_default_working_set,
        reference=_patchify_reference,
    )
)

# --- rope_2d -----------------------------------------------------------


def _rope_2d_infer(inputs, attrs):
    x, cos, sin = inputs[0], inputs[1], inputs[2]
    if len(x.shape) != 3:
        raise ValueError(
            f"rope_2d expects [tokens, heads, head_dim], got {x.name} {x.shape}"
        )
    if cos.shape != sin.shape:
        raise ValueError(f"rope_2d cos {cos.shape} and sin {sin.shape} must agree")
    if cos.shape != (x.shape[0], x.shape[2]):
        raise ValueError(
            f"rope_2d table shape {cos.shape} does not match "
            f"(tokens={x.shape[0]}, head_dim={x.shape[2]})"
        )
    if x.shape[2] % 2:
        raise ValueError(f"rope_2d head_dim {x.shape[2]} must be even")
    return ((x.shape, x.dtype),)


def _rope_2d_reference(arrays, attrs):
    """apply_rotary_pos_emb_vision, modeling_qwen3_5.py:891-902.

    The reference casts q, k, cos and sin to float32 before the rotation and
    casts the result back, so the arithmetic is float regardless of the
    activation dtype. cos/sin are unsqueezed on the head axis (:897).
    rotate_half is modeling_qwen3_5.py:562.
    """
    x, cos, sin = arrays[0], arrays[1], arrays[2]
    xf = x.astype(np.float32)
    cosf = cos.astype(np.float32)[:, None, :]
    sinf = sin.astype(np.float32)[:, None, :]
    half = xf.shape[-1] // 2
    rotated = np.concatenate([-xf[..., half:], xf[..., :half]], axis=-1)
    return ((xf * cosf + rotated * sinf).astype(x.dtype),)


register(
    OpDef(
        kind="rope_2d",
        infer=_rope_2d_infer,
        working_set=_default_working_set,
        reference=_rope_2d_reference,
    )
)

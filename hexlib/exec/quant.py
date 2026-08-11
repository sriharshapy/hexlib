# hexlib/exec/quant.py
"""Reference q4_0 quantization, in numpy, matching llama.cpp block for block.

WHY THIS EXISTS AND WHY IT IS "REFERENCE". 75 of the encoder's 259 real-work plan
steps are `matmul_epilogue` with a q4_0 weight, and hexlib has no checkpoint
loader: nothing in the tree can produce a q4_0 buffer. `RunnerSpec`'s own error
message says "hexlib does not quantize here", and that remains true of the
serializer — a `RawTensor` is bytes somebody else already quantized. This module
is that somebody, for tests and for the end-to-end encoder run. It is not on any
hot path and makes no attempt to be fast.

THE FORMAT, from `ggml-common.h:194-199` and `quantize_row_q4_0_ref`
(`ggml-quants.c:113-146`), MIT — see ATTRIBUTION.md. Read in place from
`../llama.cpp`; nothing is imported from or built against it.

    #define QK4_0 32
    typedef struct {
        ggml_half d;             // fp16 scale
        uint8_t qs[QK4_0 / 2];   // 16 bytes, two 4-bit quants each
    } block_q4_0;                // 18 bytes total

    d  = max / -8        where `max` is the SIGNED element of largest magnitude
    id = d ? 1/d : 0
    for j in 0..15:
        xi0 = min(15, (int8_t)(x[j]      * id + 8.5f))
        xi1 = min(15, (int8_t)(x[j + 16] * id + 8.5f))
        qs[j] = xi0 | (xi1 << 4)

THREE THINGS THAT ARE EASY TO GET WRONG, each producing a plausible wrong answer
rather than an error:

1. THE NIBBLE PAIRING IS j WITH j+16, NOT j WITH j+1. The LOW nibble of byte j
   holds element j and the HIGH nibble holds element j + 16 — the two halves of
   the block, not adjacent elements. Pairing adjacently round-trips to a
   permutation of the right values, which has the right norm and the right
   histogram and is wrong everywhere.

2. `d` IS SIGNED, and dividing by -8 rather than by 8 is deliberate. `max` keeps
   the sign of the largest-magnitude element, so for a block whose extreme value
   is positive `d` is negative. Using `amax / 8` instead flips the sign of every
   dequantized value in that block.

3. THE CAST TRUNCATES TOWARD ZERO and the clamp is one-sided. `(int8_t)(x + 8.5f)`
   is C truncation, not rounding, and only the upper end is clamped (to 15). The
   `+ 8.5` is what makes truncation behave as round-half-up over the expected
   range. `np.trunc` is used here rather than `np.round` for exactly this reason;
   `np.round` is banker's rounding and disagrees on every exact .5.
"""
from __future__ import annotations

import numpy as np

from hexlib.graph import ir

# Named from the same place `hexlib.graph.ir` takes them, rather than respelled:
# ir is the authority on the block size and the byte count, and a second copy of
# either is a wrong answer and not a crash.
QK4_0 = ir.Q4_0_BLOCK
BLOCK_BYTES = ir.Q4_0_BLOCK_BYTES


def quantize_q4_0(x: np.ndarray) -> bytes:
    """`x` (any shape, last axis a multiple of 32) -> packed q4_0 blocks.

    Blocks run along the LAST axis, in C order, which is what makes the byte
    stream match `ir.nbytes(x.shape, "q4_0")` and what a row-major q4_0 weight
    means: row 0's blocks, then row 1's.
    """
    a = np.ascontiguousarray(x, dtype=np.float32)
    if a.ndim == 0 or a.shape[-1] % QK4_0 != 0:
        raise ValueError(
            f"q4_0 needs a last axis that is a multiple of {QK4_0}; got shape "
            f"{tuple(a.shape)}"
        )
    blocks = a.reshape(-1, QK4_0)
    n = blocks.shape[0]

    # `max` is the SIGNED element of largest magnitude, so argmax over |v| and
    # then take the value at that index -- not amax, whose sign is always +.
    idx = np.abs(blocks).argmax(axis=1)
    mx = blocks[np.arange(n), idx].astype(np.float32)

    d = (mx / np.float32(-8.0)).astype(np.float32)
    # THE SCALE IS ROUND-TRIPPED THROUGH fp16 BEFORE IT IS USED. It is stored as
    # fp16 in the block, so the dequantizer will see the narrowed value; scaling
    # by the fp32 original here would quantize against a scale that does not
    # exist on the wire and make this function's own round trip look better than
    # the kernel's can be.
    d16 = d.astype(np.float16)
    d_used = d16.astype(np.float32)
    # `np.divide(where=...)` and not `np.where(cond, 1/d, 0)`: the latter
    # evaluates BOTH branches, so an all-zero block (d == 0, which llama.cpp
    # guards with `id = d ? 1/d : 0`) raises a divide-by-zero warning and puts an
    # inf in the array before the select discards it. Correct either way here,
    # but a warning that is always emitted is a warning nobody reads.
    inv = np.zeros_like(d_used)
    np.divide(np.float32(1.0), d_used, out=inv, where=(d_used != 0.0))

    scaled = blocks * inv[:, None]
    # Truncation toward zero, one-sided clamp at 15 -- see the module docstring.
    q = np.trunc(scaled + np.float32(8.5)).astype(np.int32)
    q = np.clip(q, 0, 15).astype(np.uint8)

    lo = q[:, : QK4_0 // 2]                 # elements 0..15  -> low nibbles
    hi = q[:, QK4_0 // 2 :]                 # elements 16..31 -> high nibbles
    qs = (lo | (hi << 4)).astype(np.uint8)

    out = np.empty((n, BLOCK_BYTES), dtype=np.uint8)
    out[:, :2] = d16.view(np.uint8).reshape(n, 2)
    out[:, 2:] = qs
    return out.tobytes()


def dequantize_q4_0(data: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """The inverse, to fp32: `v[j] = (nibble - 8) * d`.

    Provided so a test can check the round trip and so a reference matmul can be
    computed from the SAME bytes the kernel is given -- which is the only way to
    separate "the kernel's arithmetic is wrong" from "the kernel decoded the
    blocks differently".
    """
    want = ir.nbytes(tuple(shape), "q4_0")
    if len(data) != want:
        raise ValueError(
            f"a q4_0 tensor of shape {tuple(shape)} is {want} bytes; got {len(data)}"
        )
    raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, BLOCK_BYTES)
    d = raw[:, :2].copy().view(np.float16).reshape(-1).astype(np.float32)
    qs = raw[:, 2:]

    lo = (qs & 0x0F).astype(np.int32)
    hi = (qs >> 4).astype(np.int32)
    q = np.concatenate([lo, hi], axis=1)     # back to element order 0..31
    v = (q - 8).astype(np.float32) * d[:, None]
    return v.reshape(shape)

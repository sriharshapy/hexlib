/* kernels/rope_2d_fp16/kernel_api.h */
#ifndef HEXLIB_ROPE_2D_FP16_API_H
#define HEXLIB_ROPE_2D_FP16_API_H

typedef __fp16 hexlib_hf;

/* 2-D rotary position embedding applied to the vision-tower's attention Q/K.
 * 24 ops in the encoder, all one shape: x fp16 [256, 12, 64] (tokens, heads,
 * head_dim), cos/sin fp32 CONST [256, 64], output fp16 [256, 12, 64].
 *
 * SPECIFICATION SOURCE: hexlib/graph/opdefs/structural.py:222-264, the
 * `rope_2d` OpDef. `_rope_2d_infer` (222-237) fixes the shapes; the actual
 * math is `_rope_2d_reference` (240-254), itself a transcription of
 * `apply_rotary_pos_emb_vision`, modeling_qwen3_5.py:891-902, with
 * `rotate_half` at modeling_qwen3_5.py:562 (cited at structural.py:246).
 *
 *   for each token t in [0, T), head h in [0, H), let half = D / 2:
 *     for i in [0, half):
 *       y[t,h,i]      = x[t,h,i]      * cos[t,i]      - x[t,h,i+half] * sin[t,i]
 *       y[t,h,i+half] = x[t,h,i+half] * cos[t,i+half] + x[t,h,i]      * sin[t,i+half]
 *
 * (Equivalently, structural.py:253's form:
 *   rotated = concat([-x[..., half:], x[..., :half]], axis=-1)
 *   y = x * cos + rotated * sin
 *  -- the two are the same formula written two ways; the derivation above is
 *  just the concat/slice unrolled per index.)
 *
 * ==========================================================================
 * THE PAIRING CONVENTION IS SPLIT-HALF (i, i+D/2), NOT ADJACENT (2i, 2i+1).
 * ==========================================================================
 * Read directly from structural.py:253:
 *     rotated = np.concatenate([-xf[..., half:], xf[..., :half]], axis=-1)
 * `xf[..., :half]` is columns [0, half); `xf[..., half:]` is [half, D). That
 * is GPT-NeoX-style rotation (element i pairs with element i+D/2), NOT GPT-J's
 * interleaved pairing (element 2i pairs with 2i+1). CONFIRMED against two
 * independent sources, not merely consistent with them:
 *
 *   - ../HVX-clean/run_artifacts/hexlib_encoder/enc_rope2d_fp16/kernel.cpp
 *     (forge2's scalar reference for this EXACT op and shape, verified
 *     against a PyTorch golden -- see that directory's PROVENANCE.md).
 *     Lines 40-72: v_slice_1 reads x[.., 32:64] (v_x[i0*768+i1*64+32+i2]),
 *     v_neg negates it, v_slice_2 reads x[.., 0:32], v_cat writes
 *     NEGATED-SECOND-HALF into columns [0,32) and the ORIGINAL FIRST HALF
 *     into columns [32,64) -- i.e. `linalg.mlir`'s own concat, lines 13-21:
 *     `tensor.extract_slice %1[0, 0, 32] ... : ... to tensor<...x32xf32>`
 *     (the SECOND half, negated) concatenated with the extract at `[0,0,0]`
 *     (the FIRST half). Same split-half formula, unit for unit.
 *   - ../llama.cpp/ggml/src/ggml-hexagon/htp/rope-ops.c: mode
 *     HTP_ROPE_TYPE_VISION (line 27) routes through the NEOX-style pairing
 *     path (`is_vision`/`is_neox` at 474-476), whose HVX kernel
 *     `hvx_rope_neox_f32_aa` (lines 285-330) reads `v0` from the FIRST half
 *     (`src0[i]`) and `v1` from the SECOND half (`src0[he+i]`, `he = ne/2`)
 *     and computes `dst[i] = v0*cos - v1*sin`, `dst[he+i] = v0*sin + v1*cos`
 *     -- the identical split-half pairing, sign for sign, with `dst[i]`
 *     matching this file's `y[t,h,i]` and `dst[he+i]` matching
 *     `y[t,h,i+half]`.
 *
 * ALL THREE SOURCES AGREE. There is no disagreement to report.
 *
 * ==========================================================================
 * THE TABLE IS INDEXED OVER THE FULL D=64, NOT D/2=32 REUSED TWICE.
 * ==========================================================================
 * cos/sin have shape (T, D), enforced at structural.py:230 (`cos.shape !=
 * (x.shape[0], x.shape[2])` raises). cos[t,i] and cos[t,i+half] are two
 * DIFFERENT stored floats read at two different offsets, never the same
 * value read twice. Some HF rotary tables happen to satisfy
 * cos[t,i] == cos[t,i+half] (their `emb = cat(freqs, freqs)` convention), but
 * this kernel must not assume that, and the harness deliberately uses a table
 * where the two halves differ so a kernel that assumes the HF convention (and
 * reads only the low half's frequency for both) is caught.
 *
 * ==========================================================================
 * NO HEAD AXIS ON THE TABLE.
 * ==========================================================================
 * cos/sin are (T, D) -- structural.py:250-251 unsqueeze them on the HEAD axis
 * (`cosf[:, None, :]`) before the broadcast multiply, meaning the SAME row
 * cos[t,:], sin[t,:] is used for every head at token t. The table has no head
 * axis at all, which makes "index by head instead of by token" a plausible
 * stride slip: cos[h,:] is a legal, in-bounds read whenever h < T, and is
 * simply the wrong row.
 *
 * ==========================================================================
 * WHAT "2-D" MEANS HERE.
 * ==========================================================================
 * The encoder is a vision tower and each token's position is a (row, column)
 * pair over a patch grid. In the model that builds cos/sin, the two halves of
 * head_dim typically carry the two spatial axes -- e.g. columns [0, half)
 * derive their frequency from the row position, columns [half, D) from the
 * column position -- so cos[t,i] and cos[t,i+half] can differ not just in
 * value but in WHICH axis of the 2-D position they encode. That split,
 * however, happens UPSTREAM of this op: by the time cos and sin reach
 * `rope_2d`, they are already flat (T, D) tables, and `_rope_2d_infer`/
 * `_rope_2d_reference` have no row/column-specific logic at all (structural.
 * py:222-254) -- neither does this kernel. "2-D" describes where the table's
 * values came from, not anything this op computes differently per axis; it is
 * one 1-D rotation over the full head_dim, applied per token.
 *
 * ==========================================================================
 * DTYPES AND ROUNDING.
 * ==========================================================================
 * x and y are fp16 (hexlib_hf); cos and sin are fp32. Per structural.py:
 * 249-254 (`xf = x.astype(np.float32)`, ... , `.astype(x.dtype)` at the very
 * end), the rotation is computed in float32 regardless of the activation
 * dtype; only the STORED result is rounded to fp16. This kernel's HVX path
 * keeps the arithmetic in Hexagon's qf32 the whole way through and narrows
 * once, at the very end, matching that contract.
 *
 * D must be even (required by the op registry's own infer, structural.py:
 * 235-236). The fast HVX path below additionally requires D == 64 exactly:
 * that puts the entire pairing (i, i+32) inside ONE 64-lane fp16 vector,
 * split by the widen step into two 32-lane fp32 halves that line up 1:1 with
 * cos/sin's own 32-lane fp32 vectors -- and D=64 is the ONLY head_dim this op
 * is ever called with in the encoder. Other D fall back to a scalar loop:
 * correct, not vectorised. See kernel.c's header comment.
 */
#define ROPE_T 6
#define ROPE_H 3
#define ROPE_D 64

void rope_2d_fp16(const hexlib_hf *x, const float *costab, const float *sintab,
                  hexlib_hf *y, int T, int H, int D);

#endif

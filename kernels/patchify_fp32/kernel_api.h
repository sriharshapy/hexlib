/* kernels/patchify_fp32/kernel_api.h
 *
 * The vision encoder's patch-embedding rearrangement:
 *
 *   [C, T, H, W]  ->  [grid_h*grid_w, C*T*patch*patch]
 *
 * The encoder's only shape: C=3, T=2, H=W=256, patch=16, merge=2,
 * grid_h=grid_w=16 -- so [3, 2, 256, 256] fp32 -> [256, 1536] fp32
 * (1536 = 3*2*16*16, 256 = 16*16).
 *
 * THE INDEX ARITHMETIC IS TAKEN VERBATIM FROM THE OP REGISTRY, NOT DERIVED
 * HERE. `hexlib/graph/opdefs/structural.py`'s `patchify` OpDef is the
 * authority on what this op means; the eager numpy implementation at
 * `_patchify_reference` (structural.py:172-207) is the specification this
 * kernel and its baseline must both match bit-for-bit. Restated as scalar
 * index arithmetic:
 *
 *   gh = h / patch,  ph = h % patch        (structural.py:202's first reshape
 *   gw = w / patch,  pw = w % patch         splits H into grid_h*patch and
 *                                           W into grid_w*patch)
 *
 *   bh = gh / merge,  mh = gh % merge      (structural.py:204's second
 *   bw = gw / merge,  mw = gw % merge       reshape splits the grid into
 *                                           merge blocks)
 *
 *   Bw = grid_w / merge
 *   token = ((bh * Bw + bw) * merge + mh) * merge + mw
 *                                          (structural.py:206's
 *                                           .transpose(2, 5, 3, 6, 0, 1, 4, 7)
 *                                           puts token axes in (bh, bw, mh,
 *                                           mw) order, then the trailing
 *                                           .reshape flattens them row-major)
 *
 *   feat = ((c * T + t) * patch + ph) * patch + pw
 *                                          (the same transpose puts feature
 *                                           axes in (c, t, ph, pw) order)
 *
 *   out[token][feat] = img[c][t][h][w]
 *
 * MERGE DOES AFFECT THE OUTPUT ROW ORDER -- it is not decorative metadata.
 * structural.py:172-179's own docstring is explicit about why: the patch
 * merger downstream is a pure reshape (cited there as
 * modeling_qwen3_5.py:886), so consecutive runs of merge*merge=4 rows must
 * already BE the 2x2 spatial block the merger will fold together. Per-patch
 * FEATURE order, in contrast, is untouched by merge: each output row still
 * holds exactly one patch's (C, T, ph, pw) pixels, never several patches'
 * pixels concatenated into one row. Getting this backwards -- reordering
 * features by merge instead of rows, or dropping the reordering and emitting
 * plain raster order -- produces a correctly SHAPED wrong answer that no
 * shape check catches; see nearmiss_merge_ignored.c.
 *
 * WHY IT IS FAST (movement_only -- no arithmetic anywhere in this op).
 * grid_w * patch == W exactly (16 * 16 == 256), so a full source image row,
 * for one fixed (c, t, h), is 256 contiguous fp32 = 8 whole 128-byte HVX
 * vectors with NO remainder. The kernel bulk-loads that row through the
 * vector unit (32 elements per instruction instead of one) into a local
 * staging buffer, then redistributes its 16 patch-width (16-float = 64-byte,
 * sub-vector) slices to their 16 different destination rows with scalar
 * stores -- the redistribution has no contiguous run longer than one patch
 * row in either operand, so there is nothing left to vectorise there. This
 * mirrors transpose_th_fp16's rule: vectorise whatever genuinely maps onto
 * whole aligned vectors, scalar for what does not.
 *
 * ALIGNMENT / SIZE CONTRACT. `img` must be 128-byte aligned (HEXLIB_ALIGN)
 * and W <= PATCHIFY_ROWBUF_MAX (256, this op's only width, defined in
 * kernel.c). A W not a multiple of 32 floats still works (via the scalar
 * tail path over the last few elements of the row) but is never exercised
 * on the encoder's actual shape, where W == 256 divides evenly.
 *
 * TOLERANCE. Pure data movement, no arithmetic anywhere -- the comparison
 * against the baseline is EXACT, matching transpose_th_fp16's rationale.
 */
#ifndef HEXLIB_PATCHIFY_FP32_API_H
#define HEXLIB_PATCHIFY_FP32_API_H

/* The encoder's one real shape. */
#define PF_C        3
#define PF_T        2
#define PF_H        256
#define PF_W        256
#define PF_PATCH    16
#define PF_MERGE    2
#define PF_GRID_H   16
#define PF_GRID_W   16

/* A second, deliberately small shape whose W is NOT a multiple of the
 * 32-lane fp32 vector, so the harness exercises kernel.c's scalar tail path
 * (never hit by the encoder's own W=256) as well as a non-trivial merge
 * block (merge=2 with a grid taller/wider than one block) and a genuine
 * channel/temporal asymmetry (C != T, so a stride swap between them is
 * detectable -- see nearmiss_channel_temporal_swap.c).
 */
#define PF2_C        2
#define PF2_T        3
#define PF2_H        12
#define PF2_W        12
#define PF2_PATCH    3
#define PF2_MERGE    2
#define PF2_GRID_H   4
#define PF2_GRID_W   4

/*
 * img: [C, T, H, W], row-major, 128-byte aligned.
 * out: [grid_h*grid_w, C*T*patch*patch], row-major.
 *
 * Caller-supplied C, T, H, W, patch, merge, grid_h, grid_w must satisfy
 * grid_h*patch == H, grid_w*patch == W, T given == the T the image was
 * packed with, and grid_h % merge == grid_w % merge == 0 -- exactly the
 * invariants `_patchify_infer` (structural.py:145-169) checks in the
 * registry. This kernel does not re-validate them at runtime (a DSP kernel
 * is not the place for the error path the graph builder already owns) but
 * degrades safely (returns without writing) if the obviously-fatal ones
 * (non-positive dims, W larger than the row-buffer's fixed 256-float
 * capacity) are violated.
 */
void patchify_fp32(const float *img, float *out,
                    int C, int T, int H, int W,
                    int patch, int merge, int grid_h, int grid_w);

#endif

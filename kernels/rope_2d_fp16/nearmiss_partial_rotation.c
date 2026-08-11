/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: rotating only PART of head_dim and copying the rest straight
 * through, unrotated -- when this op's contract rotates the FULL head_dim,
 * always (kernel_api.h; structural.py's `_rope_2d_reference` has no partial-
 * rotary logic at all).
 *
 * WHY ANYONE WOULD WRITE IT. Partial rotary IS a real, shipped mechanism
 * elsewhere -- ../llama.cpp/ggml/src/ggml-hexagon/htp/rope-ops.c's
 * `rope_neox_f32` (lines 422-435) rotates only `rctx->n_dims` columns and then
 * explicitly copies the remaining channels through unchanged when
 * `n_dims < ne0` (line 432: "fill the remain channels with data from src
 * tensor"). Someone porting a rope kernel FROM that codebase, or simply
 * remembering that "some rope variants only rotate part of the head", could
 * carry that guard into an op whose registry entry never asked for it.
 *
 * WHAT THIS KERNEL DOES: treats only the first D/2 columns as the "rotary
 * dimension" and rotates WITHIN that half using split-half pairing at HALF
 * the correct distance (D/4 instead of D/2), then copies columns [D/2, D)
 * straight from x. Both halves of the mistake are structural: half the
 * output columns are not rotated by any angle at all, and the columns that
 * ARE rotated use the wrong pairing distance too.
 */
#include "kernel_api.h"

void rope_2d_fp16(const hexlib_hf *x, const float *costab, const float *sintab,
                  hexlib_hf *y, int T, int H, int D) {
    if (T <= 0 || H <= 0 || D <= 0) {
        return;
    }
    const int rotary_dim = D / 2;   /* WRONG: should be D, the whole head. */
    const int r_half = rotary_dim / 2;

    for (int t = 0; t < T; ++t) {
        const float *cr = costab + (long) t * D;
        const float *sr = sintab + (long) t * D;
        for (int h = 0; h < H; ++h) {
            const hexlib_hf *xr = x + ((long) t * H + h) * D;
            hexlib_hf *yr = y + ((long) t * H + h) * D;

            /* Rotate only the first `rotary_dim` columns, paired at
             * distance r_half within that sub-range. */
            for (int i = 0; i < r_half; ++i) {
                const float x0 = (float) xr[i];
                const float x1 = (float) xr[i + r_half];
                yr[i]          = (hexlib_hf) (x0 * cr[i]          - x1 * sr[i]);
                yr[i + r_half] = (hexlib_hf) (x1 * cr[i + r_half] + x0 * sr[i + r_half]);
            }
            /* WRONG: the remaining columns are copied through, unrotated --
             * this op has no such pass-through range. */
            for (int i = rotary_dim; i < D; ++i) {
                yr[i] = xr[i];
            }
        }
    }
}

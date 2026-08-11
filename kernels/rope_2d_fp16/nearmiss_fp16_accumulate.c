/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: rounding each intermediate PRODUCT to fp16 before combining
 * them, instead of accumulating the whole rotation in float (kernel_api.h /
 * structural.py:249-254: the reference casts x, cos AND sin to float32
 * before the rotation and rounds only the FINAL result to fp16).
 *
 * WHY ANYONE WOULD WRITE IT. x and y are `hexlib_hf`; cos and sin are the odd
 * ones out at `float`. A kernel author minimising type conversions -- or one
 * who has just written a kernel where x, w and y were ALL fp16 and copies
 * that shape of code -- writes each partial product straight into a
 * `hexlib_hf` local "since that's what the surrounding types are", rounding
 * it immediately rather than carrying it through as float. Nothing in the
 * function signature stops this: `(hexlib_hf) (x0 * costab[i])` and
 * `(float) ((hexlib_hf) (x0 * costab[i]))` differ only in when the fp16 round
 * trip happens, and only the SECOND is wrong here.
 *
 * WHY THIS IS THE QUIET ONE. On generic small values this costs roughly one
 * extra fp16 ULP on top of the correct kernel's own narrowing noise -- the
 * SAME order of magnitude, not obviously separable by a loose per-element
 * tolerance. See harness.c's header comment for the worked arithmetic: this
 * harness catches it not by tightening the tolerance but by including one
 * token (t = ROPE_T-1) where two large (~500) intermediate products nearly
 * cancel to a tiny (~0.1) true result. fp32/qf32 keeps that cancellation
 * accurate; fp16 rounds BOTH large products to the SAME grid point (fp16's
 * ULP at magnitude 500 is ~0.49, far coarser than the 0.1-0.2 gap between
 * them) and reports 0.0, losing the entire signal. That is what this near-
 * miss is built to be rejected on.
 */
#include "kernel_api.h"

void rope_2d_fp16(const hexlib_hf *x, const float *costab, const float *sintab,
                  hexlib_hf *y, int T, int H, int D) {
    if (T <= 0 || H <= 0 || D <= 0) {
        return;
    }
    const int half = D / 2;

    for (int t = 0; t < T; ++t) {
        const float *cr = costab + (long) t * D;
        const float *sr = sintab + (long) t * D;
        for (int h = 0; h < H; ++h) {
            const hexlib_hf *xr = x + ((long) t * H + h) * D;
            hexlib_hf *yr = y + ((long) t * H + h) * D;
            for (int i = 0; i < half; ++i) {
                const float x0 = (float) xr[i];
                const float x1 = (float) xr[i + half];

                /* WRONG: each product rounded to fp16 BEFORE combining,
                 * instead of staying in float through the whole rotation. */
                hexlib_hf p0_lo = (hexlib_hf) (x0 * cr[i]);
                hexlib_hf p1_lo = (hexlib_hf) (x1 * sr[i]);
                yr[i] = (hexlib_hf) ((float) p0_lo - (float) p1_lo);

                hexlib_hf p1_hi = (hexlib_hf) (x1 * cr[i + half]);
                hexlib_hf p0_hi = (hexlib_hf) (x0 * sr[i + half]);
                yr[i + half] = (hexlib_hf) ((float) p1_hi + (float) p0_hi);
            }
        }
    }
}

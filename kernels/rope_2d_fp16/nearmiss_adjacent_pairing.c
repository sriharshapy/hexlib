/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: pairing ADJACENT columns (2i, 2i+1) instead of SPLIT-HALF
 * columns (i, i+D/2).
 *
 * WHY ANYONE WOULD WRITE IT. Rotary embedding has two conventions in the
 * wild, and they are both called "RoPE" in casual writing: GPT-J's
 * interleaved pairing (2i, 2i+1) and GPT-NeoX's split-half pairing
 * (i, i+D/2). They are both real, both shipped in major model families, and
 * nothing about the OP NAME `rope_2d` or its SHAPES distinguishes them -- a
 * kernel author who has implemented the interleaved form before (it is, if
 * anything, the more commonly taught one) will reach for it here too. Only
 * reading the reference formula (structural.py:253's slice-negate-concat, or
 * equivalently kernel_api.h's derivation) settles which one this op needs.
 *
 * WHY IT SURVIVES A SHAPE CHECK. Both conventions consume the same x, cos and
 * sin tensors, of the same shapes, and produce a same-shaped output where
 * every value is a plausible rotated float. There is no dimension mismatch,
 * no NaN, nothing a shape assertion or an "is it finite" check would catch --
 * only comparison against a reference computed with the RIGHT pairing catches
 * it, and only if the input data is rich enough that the two pairings
 * actually diverge (which this harness's asymmetric x and cos/sin ensure).
 */
#include "kernel_api.h"

void rope_2d_fp16(const hexlib_hf *x, const float *costab, const float *sintab,
                  hexlib_hf *y, int T, int H, int D) {
    if (T <= 0 || H <= 0 || D <= 0) {
        return;
    }

    for (int t = 0; t < T; ++t) {
        const float *cr = costab + (long) t * D;
        const float *sr = sintab + (long) t * D;
        for (int h = 0; h < H; ++h) {
            const hexlib_hf *xr = x + ((long) t * H + h) * D;
            hexlib_hf *yr = y + ((long) t * H + h) * D;
            /* WRONG: pairs (2i, 2i+1), the GPT-J interleaved convention,
             * instead of (i, i+D/2). */
            for (int i = 0; i + 1 < D; i += 2) {
                const float x0 = (float) xr[i];
                const float x1 = (float) xr[i + 1];
                yr[i]     = (hexlib_hf) (x0 * cr[i]     - x1 * sr[i]);
                yr[i + 1] = (hexlib_hf) (x1 * cr[i + 1] + x0 * sr[i + 1]);
            }
        }
    }
}

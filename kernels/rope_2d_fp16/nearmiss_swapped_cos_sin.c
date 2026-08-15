/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: swapping the cos and sin table arguments -- using sin where
 * cos belongs and cos where sin belongs, with the pairing and the signs
 * otherwise correct.
 *
 * WHY ANYONE WOULD WRITE IT. `rope_2d_fp16(x, costab, sintab, y, T, H, D)`
 * takes cos before sin; a transcription slip (copying from a reference that
 * lists them the other way, or simply mistyping two adjacent identifiers
 * that are the same shape and dtype) produces a function that compiles
 * cleanly -- costab and sintab are both `const float *`, so nothing in the
 * type system notices the swap.
 *
 * WHY IT SURVIVES A SHAPE CHECK. Same shapes, same dtypes, a same-shaped
 * plausible-looking rotated output. It is caught only by comparison against
 * the real cos/sin assignment, and only because this harness's cos and sin
 * values are NOT symmetric (cos[t,i] != sin[t,i] almost everywhere) -- a
 * harness built from, say, a 45-degree-only angle table (cos == sin) would
 * let this bug pass by coincidence.
 */
#include "kernel_api.h"

void rope_2d_fp16(const hexlib_hf *x, const float *costab, const float *sintab,
                  hexlib_hf *y, int T, int H, int D) {
    if (T <= 0 || H <= 0 || D <= 0) {
        return;
    }
    const int half = D / 2;

    for (int t = 0; t < T; ++t) {
        /* WRONG: cr reads sintab, sr reads costab. */
        const float *cr = sintab + (long) t * D;
        const float *sr = costab + (long) t * D;
        for (int h = 0; h < H; ++h) {
            const hexlib_hf *xr = x + ((long) t * H + h) * D;
            hexlib_hf *yr = y + ((long) t * H + h) * D;
            for (int i = 0; i < half; ++i) {
                const float x0 = (float) xr[i];
                const float x1 = (float) xr[i + half];
                yr[i]        = (hexlib_hf) (x0 * cr[i]        - x1 * sr[i]);
                yr[i + half] = (hexlib_hf) (x1 * cr[i + half] + x0 * sr[i + half]);
            }
        }
    }
}

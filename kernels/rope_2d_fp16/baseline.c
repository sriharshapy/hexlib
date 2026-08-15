/* kernels/rope_2d_fp16/baseline.c */
#include "kernel_api.h"

/* Scalar reference. Correct and obvious, never fast.
 *
 * A direct transcription of the per-index form derived in kernel_api.h from
 * structural.py's `_rope_2d_reference` -- see that header for the full
 * derivation and the split-half pairing citation (structural.py:253, cross-
 * checked against the forge2 reference and llama.cpp's HTP_ROPE_TYPE_VISION
 * kernel).
 *
 * Everything is accumulated in `float`, matching the reference's cast to
 * float32 before the rotation; only the store into `yr[...]` rounds to fp16.
 * cos/sin are read at the SAME token row for every head (no head axis on the
 * table), and at their OWN column -- cos[t,i] and cos[t,i+half] are two
 * distinct reads, never the same value reused.
 */
void rope_2d_fp16_baseline(const hexlib_hf *x, const float *costab,
                           const float *sintab, hexlib_hf *y,
                           int T, int H, int D) {
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
                yr[i]        = (hexlib_hf) (x0 * cr[i]        - x1 * sr[i]);
                yr[i + half] = (hexlib_hf) (x1 * cr[i + half] + x0 * sr[i + half]);
            }
        }
    }
}

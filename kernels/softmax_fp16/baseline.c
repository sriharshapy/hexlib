#include "kernel_api.h"

#include <math.h>

/* Scalar reference. Correct and obvious, never fast.
 *
 * Three passes over the row, matching hexlib/graph/opdefs/elementwise.py:141-146
 * exactly in ORDER OF OPERATIONS: find the max first, subtract it before the
 * exponential (never after), sum the exponentials, then divide. Every
 * intermediate (`m`, `e[c]`, `s`) is float32; only the stored result is rounded
 * to fp16 -- see kernel_api.h's PRECISION note for why float32 rather than the
 * reference's float64 is the right thing for this kernel to be held to.
 *
 * expf() here is the toolchain's own libm, not a polynomial approximation --
 * this file is the reference the polynomial in kernel.c is checked against, so
 * it must not share the same approximation error.
 */
void softmax_fp16_baseline(const hexlib_hf *x, hexlib_hf *y, int R, int C) {
    for (int r = 0; r < R; ++r) {
        const hexlib_hf *xr = x + (long) r * C;
        hexlib_hf *yr = y + (long) r * C;

        float m = (float) xr[0];
        for (int c = 1; c < C; ++c) {
            float v = (float) xr[c];
            if (v > m) m = v;
        }

        float e[C > 0 ? C : 1];  /* VLA: baseline is plain scalar C, no HVX
                                   * alignment constraint to respect. */
        float s = 0.0f;
        for (int c = 0; c < C; ++c) {
            float v = expf((float) xr[c] - m);
            e[c] = v;
            s += v;
        }

        for (int c = 0; c < C; ++c) {
            yr[c] = (hexlib_hf) (e[c] / s);
        }
    }
}

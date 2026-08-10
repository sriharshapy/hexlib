#include "kernel_api.h"

#include <math.h>

/* Scalar reference. Correct and obvious, never fast.
 *
 * Two passes over the row: the mean first, then the sum of squared deviations
 * FROM THAT MEAN. The one-pass alternative (E[x^2] - mean^2) is algebraically
 * equal and numerically worse -- it subtracts two large nearly-equal numbers --
 * so the reference uses the form that does not lose precision, and the kernel is
 * held to it. */
void layernorm_fp16_baseline(const hexlib_hf *x, const float *w, const float *b,
                             hexlib_hf *y, int R, int C, float eps) {
    for (int r = 0; r < R; ++r) {
        const hexlib_hf *xr = x + (long) r * C;
        hexlib_hf *yr = y + (long) r * C;

        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            sum += (float) xr[c];
        }
        const float mean = sum / (float) C;

        float sq = 0.0f;
        for (int c = 0; c < C; ++c) {
            const float d = (float) xr[c] - mean;
            sq += d * d;
        }
        const float inv = 1.0f / sqrtf(sq / (float) C + eps);

        for (int c = 0; c < C; ++c) {
            yr[c] = (hexlib_hf) (((float) xr[c] - mean) * inv * w[c] + b[c]);
        }
    }
}

/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: dividing the sum of squared deviations by C-1 instead of C.
 *
 * WHY ANYONE WOULD WRITE IT. C-1 is the SAMPLE variance, and it is the one every
 * statistics course teaches as "the" variance -- the unbiased estimator of a
 * population variance from a sample. LayerNorm is not estimating anything. It is
 * normalising a fixed vector, so the divisor is C. Reaching for the statistics
 * habit rather than the definition is a single, natural slip.
 *
 * WHY IT IS EASY TO MISS: the error is a factor of sqrt(C/(C-1)) on every output,
 * which at C=768 is 1.00065 -- about 0.065%. That is well inside the eye's
 * tolerance for "looks normalised", it will not show up as a NaN or an obviously
 * broken image, and it is small enough that a loose relative tolerance would
 * accept it. The harness compares against a reference computed the right way,
 * which is the only thing that catches an error this size.
 */
#include "kernel_api.h"

#include <math.h>

void layernorm_fp16(const hexlib_hf *x, const float *w, const float *b,
                    hexlib_hf *y, int R, int C, float eps) {
    for (int r = 0; r < R; ++r) {
        const hexlib_hf *xr = x + (long) r * C;
        hexlib_hf *yr = y + (long) r * C;

        float sum = 0.0f;
        for (int c = 0; c < C; ++c) sum += (float) xr[c];
        const float mean = sum / (float) C;

        float sq = 0.0f;
        for (int c = 0; c < C; ++c) {
            const float d = (float) xr[c] - mean;
            sq += d * d;
        }
        /* WRONG: C-1. LayerNorm's variance is biased -- divide by C. */
        const float inv = 1.0f / sqrtf(sq / (float) (C - 1) + eps);

        for (int c = 0; c < C; ++c) {
            yr[c] = (hexlib_hf) (((float) xr[c] - mean) * inv * w[c] + b[c]);
        }
    }
}

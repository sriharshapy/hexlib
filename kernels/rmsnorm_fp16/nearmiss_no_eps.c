/* kernels/rmsnorm_fp16/nearmiss_no_eps.c
 * Plausible bug: eps dropped from the denominator. Correct on ordinary rows,
 * catastrophic on the near-zero row 0. */
#include "kernel_api.h"

#include <math.h>

void rmsnorm_fp16(const hexlib_hf *x, const hexlib_hf *w,
                  hexlib_hf *y, int R, int C, float eps) {
    (void) eps;
    for (int r = 0; r < R; ++r) {
        float acc = 0.0f;
        for (int c = 0; c < C; ++c) {
            float v = (float) x[r * C + c];
            acc += v * v;
        }
        float inv = 1.0f / sqrtf(acc / (float) C);   /* eps dropped */
        for (int c = 0; c < C; ++c) {
            y[r * C + c] = (hexlib_hf) ((float) x[r * C + c] * inv * (float) w[c]);
        }
    }
}

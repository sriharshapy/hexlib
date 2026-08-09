/* kernels/rmsnorm_fp16/nearmiss_mean_not_rms.c
 * Plausible bug: normalising by the mean of x rather than the root mean square
 * of x — the single most common confusion between LayerNorm and RMSNorm. */
#include "kernel_api.h"

#include <math.h>

void rmsnorm_fp16(const hexlib_hf *x, const hexlib_hf *w,
                  hexlib_hf *y, int R, int C, float eps) {
    for (int r = 0; r < R; ++r) {
        float acc = 0.0f;
        for (int c = 0; c < C; ++c) {
            acc += (float) x[r * C + c];         /* not squared */
        }
        float inv = 1.0f / sqrtf(acc / (float) C + eps);
        for (int c = 0; c < C; ++c) {
            y[r * C + c] = (hexlib_hf) ((float) x[r * C + c] * inv * (float) w[c]);
        }
    }
}

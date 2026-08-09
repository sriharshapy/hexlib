/* kernels/rmsnorm_fp16/baseline.c — correct and obvious, never fast. */
#include "kernel_api.h"

#include <math.h>

void rmsnorm_fp16_baseline(const hexlib_hf *x, const hexlib_hf *w,
                           hexlib_hf *y, int R, int C, float eps) {
    for (int r = 0; r < R; ++r) {
        float acc = 0.0f;
        for (int c = 0; c < C; ++c) {
            float v = (float) x[r * C + c];
            acc += v * v;
        }
        float inv = 1.0f / sqrtf(acc / (float) C + eps);
        for (int c = 0; c < C; ++c) {
            y[r * C + c] = (hexlib_hf) ((float) x[r * C + c] * inv * (float) w[c]);
        }
    }
}

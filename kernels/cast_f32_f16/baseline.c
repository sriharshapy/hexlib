#include "kernel_api.h"

/* Scalar reference. Correct and obvious, never fast.
 *
 * The C cast is IEEE round-to-nearest-even. The vector path's narrowing is not
 * guaranteed to be, so the harness compares with a tolerance of one fp16 ULP
 * rather than exactly -- see kernel_api.h. */
void cast_f32_f16_baseline(const float *x, hexlib_hf *y, int n) {
    for (int i = 0; i < n; ++i) {
        y[i] = (hexlib_hf) x[i];
    }
}

#include "kernel_api.h"

/* Scalar reference. Correct and obvious, never fast.
 *
 * The multiply is done in float and rounded once on store, which is what the
 * vector kernel's qf32 intermediate also does -- so the two agree on where
 * rounding happens and the comparison is about the arithmetic rather than
 * about rounding position. */
void scale_fp16_baseline(const hexlib_hf *x, hexlib_hf *y, int n, float factor) {
    for (int i = 0; i < n; ++i) {
        y[i] = (hexlib_hf) ((float) x[i] * factor);
    }
}

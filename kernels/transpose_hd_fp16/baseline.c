#include "kernel_api.h"

/* Scalar reference. Correct and obvious, never fast.
 *
 * No arithmetic, so there is no rounding anywhere and the comparison against
 * the kernel is EXACT: the output must be a permutation of the input's exact
 * bytes. That makes this the one kernel in the set where a tolerance would
 * hide a bug rather than accommodate the hardware. */
void transpose_hd_fp16_baseline(const hexlib_hf *x, hexlib_hf *y,
                                int B, int T, int D) {
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int d = 0; d < D; ++d) {
                y[((long) b * D + d) * T + t] = x[((long) b * T + t) * D + d];
            }
        }
    }
}

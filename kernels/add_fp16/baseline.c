#include "kernel_api.h"

/* Scalar reference. Correct and obvious, never fast.
 *
 * Widens to float, adds, rounds once on store. For a single add of two fp16
 * values this is bit-identical to adding in fp16 -- the exact sum is always
 * representable in fp32 -- so this reference does not favour either
 * implementation strategy. */
void add_fp16_baseline(const hexlib_hf *a, const hexlib_hf *b,
                       hexlib_hf *y, int n) {
    for (int i = 0; i < n; ++i) {
        y[i] = (hexlib_hf) ((float) a[i] + (float) b[i]);
    }
}

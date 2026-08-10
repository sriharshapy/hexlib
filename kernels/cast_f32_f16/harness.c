/* kernels/cast_f32_f16/harness.c
 *
 * TWO SHAPES, ONE VERDICT: a multiple of 64 for the vector path, and 4133 for
 * the scalar remainder. Accumulated into a single verdict because the driver
 * requires exactly one.
 *
 * THE INPUTS ARE BUILT TO CATCH A SWAPPED HALF. Q6_W_vcombine_VV's argument order
 * decides which input vector lands in which half of the narrowed output, and
 * getting it backwards produces an output of exactly the right length with 32-lane
 * blocks transposed. Values are therefore MONOTONIC in the index, so a swap
 * shows up as an obvious discontinuity rather than as two similar numbers that a
 * tolerance might accept.
 *
 * Magnitudes stay inside fp16's normal range. Values that overflow fp16 are a
 * real concern for a narrowing op, but the C cast's behaviour on overflow is
 * implementation-defined, so making the reference disagree with the hardware
 * there would test the reference rather than the kernel.
 */
#include "hexlib/hexlib_harness.h"
#include "kernel_api.h"

void cast_f32_f16_baseline(const float *, hexlib_hf *, int);

static float X[CAST_N]        HEXLIB_ALIGN;
static hexlib_hf Y[CAST_N]    HEXLIB_ALIGN;
static hexlib_hf REF[CAST_N]  HEXLIB_ALIGN;

/* One fp16 ULP at magnitude m is about m * 2^-10. The tolerance is relative, so
 * 1e-3 covers a single ULP anywhere in the range with a little room, and would
 * still reject a swapped half or a dropped tail by orders of magnitude. */
#define CAST_REL 1e-3f

static void fill(int n) {
    for (int i = 0; i < n; ++i) {
        /* Monotonic, both signs, spanning three orders of magnitude. */
        X[i] = -600.0f + (float) i * 0.29f;
    }
    for (int i = 0; i < n; ++i) {
        Y[i] = (hexlib_hf) 12345.0f;
    }
}

static void compare(int n, int *n_wrong, double *max_err) {
    for (int i = 0; i < n; ++i) {
        if (!hexlib_close_f16((float) Y[i], (float) REF[i], CAST_REL, 1e-4f)) {
            ++(*n_wrong);
        }
        double d = (double) (float) Y[i] - (double) (float) REF[i];
        if (d < 0.0) d = -d;
        if (d > *max_err) *max_err = d;
    }
}

int main(void) {
    int n_wrong = 0;
    double max_err = 0.0;
    unsigned long long kcyc = 0;

    fill(CAST_N);
    cast_f32_f16_baseline(X, REF, CAST_N);
    HEXLIB_TIME_KERNEL(kcyc, cast_f32_f16(X, Y, CAST_N));
    compare(CAST_N, &n_wrong, &max_err);

    /* Scalar remainder path. */
    fill(CAST_N_ODD);
    cast_f32_f16_baseline(X, REF, CAST_N_ODD);
    cast_f32_f16(X, Y, CAST_N_ODD);
    compare(CAST_N_ODD, &n_wrong, &max_err);

    hexlib_report(n_wrong == 0, n_wrong, max_err, kcyc);
    return 0;
}

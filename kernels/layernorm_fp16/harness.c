/* kernels/layernorm_fp16/harness.c
 *
 * ROWS ARE DELIBERATELY DISSIMILAR. Row 0 has a large nonzero mean, row 1 is
 * near-zero-mean, row 2 has a tiny spread so eps and the variance matter, row 3
 * is asymmetric. A LayerNorm bug that only shows up when the mean is far from
 * zero -- forgetting to subtract it, say -- is invisible on centred data.
 *
 * w AND b VARY PER COLUMN AND b IS NOT ZERO. A kernel that applied the affine
 * params to the wrong columns, or dropped the bias, passes on constant w and
 * b=0. One of the near-misses is exactly a column misalignment.
 *
 * TWO WIDTHS, AND THE SECOND ONE IS NOT OPTIONAL. The encoder's C is 768, and at
 * that width a biased-vs-unbiased variance error is a factor of
 * sqrt(768/767) = 1.00065 -- 0.065%. fp16's own relative precision is about
 * 0.05%, so at C=768 that bug is the same size as the storage noise and CANNOT be
 * separated from it by comparing fp16 outputs. It was in fact wrongly accepted
 * here before this pass existed.
 *
 * The error grows as C shrinks: at C=64 it is sqrt(64/63) = 1.0079, or 0.79%,
 * comfortably above fp16 noise. So the harness checks C=64 as well, with a
 * tolerance tight enough to separate the two (0.4% relative, which the correct
 * kernel's ~0.05% clears and the unbiased variant's 0.79% does not).
 *
 * The lesson generalises: a tolerance wide enough for the widest shape can be
 * wider than the bug you are trying to catch, and the fix is a shape where the
 * bug is bigger, not a tolerance you have talked yourself into.
 */
#include "hexlib/hexlib_harness.h"
#include "kernel_api.h"

void layernorm_fp16_baseline(const hexlib_hf *, const float *, const float *,
                            hexlib_hf *, int, int, float);

static hexlib_hf X[LN_R * LN_C]   HEXLIB_ALIGN;
static float W[LN_C]              HEXLIB_ALIGN;
static float B[LN_C]              HEXLIB_ALIGN;
static hexlib_hf Y[LN_R * LN_C]   HEXLIB_ALIGN;
static hexlib_hf REF[LN_R * LN_C] HEXLIB_ALIGN;

/* The narrow width that makes a biased-vs-unbiased variance error visible.
 * A multiple of 64 so the vector path still runs. */
#define LN_C_NARROW 64

static void fill(int C) {
    for (int r = 0; r < LN_R; ++r) {
        for (int c = 0; c < C; ++c) {
            float v;
            switch (r) {
                case 0:  v = 20.0f + 0.5f * (float) ((c % 13) - 6); break;
                case 1:  v = 0.25f * (float) ((c % 17) - 8);        break;
                case 2:  v = 1e-3f * (float) ((c % 7) - 3);         break;
                default: v = (float) (c % 31) * 0.125f;             break;
            }
            X[r * C + c] = (hexlib_hf) v;
        }
    }
    for (int i = 0; i < LN_R * C; ++i) {
        Y[i] = (hexlib_hf) 12345.0f;
    }
}

static void compare(int C, float rel, float abs_tol, int *n_wrong,
                    double *max_err) {
    for (int i = 0; i < LN_R * C; ++i) {
        if (!hexlib_close_f16((float) Y[i], (float) REF[i], rel, abs_tol)) {
            ++(*n_wrong);
        }
        double d = (double) (float) Y[i] - (double) (float) REF[i];
        if (d < 0.0) d = -d;
        if (d > *max_err) *max_err = d;
    }
}

int main(void) {
    for (int c = 0; c < LN_C; ++c) {
        W[c] = 0.75f + 0.01f * (float) (c % 23);
        B[c] = -0.5f + 0.02f * (float) (c % 11);
    }

    for (int r = 0; r < LN_R; ++r) {
        for (int c = 0; c < LN_C; ++c) {
            float v;
            switch (r) {
                case 0:  v = 20.0f + 0.5f * (float) ((c % 13) - 6); break;
                case 1:  v = 0.25f * (float) ((c % 17) - 8);        break;
                case 2:  v = 1e-3f * (float) ((c % 7) - 3);         break;
                default: v = (float) (c % 31) * 0.125f;             break;
            }
            X[r * LN_C + c] = (hexlib_hf) v;
        }
    }

    for (int i = 0; i < LN_R * LN_C; ++i) {
        Y[i] = (hexlib_hf) 12345.0f;
    }

    layernorm_fp16_baseline(X, W, B, REF, LN_R, LN_C, LN_EPS);

    unsigned long long kcyc = 0;
    HEXLIB_TIME_KERNEL(kcyc, layernorm_fp16(X, W, B, Y, LN_R, LN_C, LN_EPS));

    int n_wrong = 0;
    double max_err = 0.0;
    compare(LN_C, 0.02f, 1e-3f, &n_wrong, &max_err);

    /* Narrow width, tight tolerance: the pass that can actually see a
     * biased-vs-unbiased variance error. See the header comment. */
    fill(LN_C_NARROW);
    layernorm_fp16_baseline(X, W, B, REF, LN_R, LN_C_NARROW, LN_EPS);
    layernorm_fp16(X, W, B, Y, LN_R, LN_C_NARROW, LN_EPS);
    compare(LN_C_NARROW, 0.004f, 1e-4f, &n_wrong, &max_err);

    hexlib_report(n_wrong == 0, n_wrong, max_err, kcyc);
    return 0;
}

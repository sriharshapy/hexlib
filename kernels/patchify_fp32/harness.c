/* kernels/patchify_fp32/harness.c
 *
 * TWO SHAPES, ONE VERDICT. The encoder shape (PF_*) is W=256=8*32, so it
 * exercises kernel.c's whole-vector row-staging path and is what the cycle
 * count is measured on. The small shape (PF2_*) has W=12 -- not a multiple
 * of the 32-lane fp32 vector -- so it exercises the scalar tail path that
 * PF_* never reaches, and it uses C != T (2 vs 3) so a channel/temporal
 * stride swap is actually detectable (see nearmiss_channel_temporal_swap.c's
 * own note on why this matters).
 *
 * Every input element gets its own exact integer value (its flat index --
 * fp32 represents integers up to 2^24 exactly, and both shapes here are far
 * smaller than that), so a comparison cannot pass by two different positions
 * happening to hold equal numbers. The check is exact: this op does no
 * arithmetic, so any difference at all is a bug.
 */
#include "hexlib/hexlib_harness.h"
#include "kernel_api.h"

void patchify_fp32_baseline(const float *, float *, int, int, int, int,
                             int, int, int, int);

#define PF_IN_N  (PF_C * PF_T * PF_H * PF_W)
#define PF_OUT_N (PF_GRID_H * PF_GRID_W * PF_C * PF_T * PF_PATCH * PF_PATCH)

#define PF2_IN_N  (PF2_C * PF2_T * PF2_H * PF2_W)
#define PF2_OUT_N (PF2_GRID_H * PF2_GRID_W * PF2_C * PF2_T * PF2_PATCH * PF2_PATCH)

#define MAXN (PF_IN_N > PF_OUT_N ? PF_IN_N : PF_OUT_N)

static float X[MAXN]   HEXLIB_ALIGN;
static float Y[MAXN]   HEXLIB_ALIGN;
static float REF[MAXN] HEXLIB_ALIGN;

static int check(int C, int T, int H, int W, int patch, int merge,
                  int grid_h, int grid_w, int n_in, int n_out,
                  int *n_wrong, double *max_err) {
    for (int i = 0; i < n_in; ++i) {
        X[i] = (float) i;   /* distinct, exactly representable */
    }
    for (int i = 0; i < n_out; ++i) {
        Y[i] = 12345.0f;
        REF[i] = 0.0f;
    }

    patchify_fp32_baseline(X, REF, C, T, H, W, patch, merge, grid_h, grid_w);
    patchify_fp32(X, Y, C, T, H, W, patch, merge, grid_h, grid_w);

    for (int i = 0; i < n_out; ++i) {
        double d = (double) Y[i] - (double) REF[i];
        if (d != 0.0) {
            ++(*n_wrong);
        }
        if (d < 0.0) d = -d;
        if (d > *max_err) *max_err = d;
    }
    return n_out;
}

int main(void) {
    int n_wrong = 0;
    double max_err = 0.0;

    unsigned long long kcyc = 0;
    /* Timed call is the encoder's own shape (the whole-vector row path). The
     * small shape is checked for correctness but deliberately not folded
     * into the cycle count, which would make the number mean nothing. */
    {
        for (int i = 0; i < PF_IN_N; ++i) {
            X[i] = (float) i;
        }
        for (int i = 0; i < PF_OUT_N; ++i) {
            Y[i] = 12345.0f;
        }
        patchify_fp32_baseline(X, REF, PF_C, PF_T, PF_H, PF_W, PF_PATCH,
                                PF_MERGE, PF_GRID_H, PF_GRID_W);
        HEXLIB_TIME_KERNEL(kcyc, patchify_fp32(X, Y, PF_C, PF_T, PF_H, PF_W,
                                                PF_PATCH, PF_MERGE,
                                                PF_GRID_H, PF_GRID_W));
        for (int i = 0; i < PF_OUT_N; ++i) {
            double d = (double) Y[i] - (double) REF[i];
            if (d != 0.0) ++n_wrong;
            if (d < 0.0) d = -d;
            if (d > max_err) max_err = d;
        }
    }

    /* Scalar-tail + non-trivial-merge shape. */
    check(PF2_C, PF2_T, PF2_H, PF2_W, PF2_PATCH, PF2_MERGE,
          PF2_GRID_H, PF2_GRID_W, PF2_IN_N, PF2_OUT_N, &n_wrong, &max_err);

    hexlib_report(n_wrong == 0, n_wrong, max_err, kcyc);
    return 0;
}

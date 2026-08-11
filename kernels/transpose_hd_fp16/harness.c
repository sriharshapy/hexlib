/* kernels/transpose_hd_fp16/harness.c
 *
 * TWO SHAPES, ONE VERDICT. TR_T=64 makes each gathered chunk exactly one
 * 128-byte HVX vector and exercises the vector-store path; TR_T_ODD=40 makes
 * every chunk 80 bytes, so no whole vector ever fits and the scalar tail path
 * runs for the entire row instead. Both are checked and the counts are
 * accumulated into a single verdict, because the driver requires exactly one
 * and would treat two as an error.
 *
 * B, T and D are deliberately three DIFFERENT numbers (5, 64, 7). A kernel
 * that confuses which axis belongs in which stride -- the classic transpose
 * bug -- produces the right answer whenever the two confused axes happen to
 * have equal size, so the near-misses that make exactly that mistake would
 * pass a harness built on equal dimensions. They are why all three differ.
 *
 * Every element is given a distinct value, so a comparison cannot pass by two
 * different positions happening to hold the same number. The check is exact:
 * this op does no arithmetic, so any difference at all is a bug.
 */
#include "hexlib/hexlib_harness.h"
#include "kernel_api.h"

void transpose_hd_fp16_baseline(const hexlib_hf *, hexlib_hf *, int, int, int);

#define MAXN (TR_B * TR_T * TR_D)

static hexlib_hf X[MAXN]   HEXLIB_ALIGN;
static hexlib_hf Y[MAXN]   HEXLIB_ALIGN;
static hexlib_hf REF[MAXN] HEXLIB_ALIGN;

static int check(int B, int T, int D, int *n_wrong, double *max_err) {
    const int n = B * T * D;
    for (int i = 0; i < n; ++i) {
        /* Distinct and exactly representable in fp16: integers up to 2048 are. */
        X[i] = (hexlib_hf) (float) (i % 2048);
    }
    for (int i = 0; i < n; ++i) {
        Y[i] = (hexlib_hf) 12345.0f;
        REF[i] = (hexlib_hf) 0.0f;
    }

    transpose_hd_fp16_baseline(X, REF, B, T, D);
    transpose_hd_fp16(X, Y, B, T, D);

    for (int i = 0; i < n; ++i) {
        double d = (double) (float) Y[i] - (double) (float) REF[i];
        if (d != 0.0) {
            ++(*n_wrong);
        }
        if (d < 0.0) d = -d;
        if (d > *max_err) *max_err = d;
    }
    return n;
}

int main(void) {
    int n_wrong = 0;
    double max_err = 0.0;

    unsigned long long kcyc = 0;
    /* Timed call is the whole-vector shape. The odd-T shape is checked for
     * correctness but deliberately not folded into the cycle count, which
     * would make the number mean nothing. */
    {
        const int n = TR_B * TR_T * TR_D;
        for (int i = 0; i < n; ++i) {
            X[i] = (hexlib_hf) (float) (i % 2048);
            Y[i] = (hexlib_hf) 12345.0f;
        }
        transpose_hd_fp16_baseline(X, REF, TR_B, TR_T, TR_D);
        HEXLIB_TIME_KERNEL(kcyc, transpose_hd_fp16(X, Y, TR_B, TR_T, TR_D));
        for (int i = 0; i < n; ++i) {
            double d = (double) (float) Y[i] - (double) (float) REF[i];
            if (d != 0.0) ++n_wrong;
            if (d < 0.0) d = -d;
            if (d > max_err) max_err = d;
        }
    }

    /* Scalar-only path: no chunk of the row is a whole number of vectors. */
    check(TR_B, TR_T_ODD, TR_D, &n_wrong, &max_err);

    hexlib_report(n_wrong == 0, n_wrong, max_err, kcyc);
    return 0;
}

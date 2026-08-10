/* kernels/transpose_th_fp16/harness.c
 *
 * TWO SHAPES, ONE VERDICT. D=64 makes each run exactly one 128-byte vector and
 * exercises the vector path; D=40 makes it 80 bytes, so no whole vector fits and
 * the scalar path runs instead. Both are checked and the counts are accumulated
 * into a single verdict, because the driver requires exactly one and would treat
 * two as an error.
 *
 * T and H are deliberately DIFFERENT (8 and 3). With T == H a kernel that
 * confuses the two strides produces the right answer, and one of the near-misses
 * is exactly that confusion.
 *
 * Every element is given a distinct value, so a comparison cannot pass by two
 * different positions happening to hold the same number. The check is exact:
 * this op does no arithmetic, so any difference at all is a bug.
 */
#include "hexlib/hexlib_harness.h"
#include "kernel_api.h"

void transpose_th_fp16_baseline(const hexlib_hf *, hexlib_hf *, int, int, int);

#define MAXN (TR_T * TR_H * TR_D)

static hexlib_hf X[MAXN]   HEXLIB_ALIGN;
static hexlib_hf Y[MAXN]   HEXLIB_ALIGN;
static hexlib_hf REF[MAXN] HEXLIB_ALIGN;

static int check(int T, int H, int D, int *n_wrong, double *max_err) {
    const int n = T * H * D;
    for (int i = 0; i < n; ++i) {
        /* Distinct and exactly representable in fp16: integers up to 2048 are. */
        X[i] = (hexlib_hf) (float) (i % 2048);
    }
    for (int i = 0; i < n; ++i) {
        Y[i] = (hexlib_hf) 12345.0f;
        REF[i] = (hexlib_hf) 0.0f;
    }

    transpose_th_fp16_baseline(X, REF, T, H, D);
    transpose_th_fp16(X, Y, T, H, D);

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
    /* Timed call is the encoder's shape (the vector path). The odd-D shape is
     * checked for correctness but deliberately not folded into the cycle count,
     * which would make the number mean nothing. */
    {
        const int n = TR_T * TR_H * TR_D;
        for (int i = 0; i < n; ++i) {
            X[i] = (hexlib_hf) (float) (i % 2048);
            Y[i] = (hexlib_hf) 12345.0f;
        }
        transpose_th_fp16_baseline(X, REF, TR_T, TR_H, TR_D);
        HEXLIB_TIME_KERNEL(kcyc, transpose_th_fp16(X, Y, TR_T, TR_H, TR_D));
        for (int i = 0; i < n; ++i) {
            double d = (double) (float) Y[i] - (double) (float) REF[i];
            if (d != 0.0) ++n_wrong;
            if (d < 0.0) d = -d;
            if (d > max_err) max_err = d;
        }
    }

    /* Scalar path: run length is not a whole number of vectors. */
    check(TR_T, TR_H, TR_D_ODD, &n_wrong, &max_err);

    hexlib_report(n_wrong == 0, n_wrong, max_err, kcyc);
    return 0;
}

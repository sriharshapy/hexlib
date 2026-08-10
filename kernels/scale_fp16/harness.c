/* kernels/scale_fp16/harness.c
 *
 * Builds inputs, runs the baseline for reference, times ONLY the kernel call,
 * compares with tolerance, and prints the two lines the driver parses.
 *
 * TWO THINGS THIS HARNESS DOES DELIBERATELY:
 *
 *  1. SCALE_N is 4100, not a multiple of the 64-lane vector, so the tail path
 *     runs. The encoder's real shape is a multiple of 64, so a harness sized
 *     from the caller would leave the tail untested -- and one of the two
 *     near-misses is exactly a missing tail.
 *
 *  2. The output is poisoned before the call with a value no correct result can
 *     produce. A kernel that writes nothing, or writes only part of the array,
 *     then fails on the poison rather than passing on a buffer that happened to
 *     contain zeros.
 */
#include "hexlib/hexlib_harness.h"
#include "kernel_api.h"

void scale_fp16_baseline(const hexlib_hf *, hexlib_hf *, int, float);

static hexlib_hf X[SCALE_N]   HEXLIB_ALIGN;
static hexlib_hf Y[SCALE_N]   HEXLIB_ALIGN;
static hexlib_hf REF[SCALE_N] HEXLIB_ALIGN;

int main(void) {
    /* A spread of magnitudes and both signs, so a sign error or a saturation
     * bug has somewhere to show up. Values stay well inside fp16's range: the
     * kernel under test is a scale, and overflow behaviour is not what is being
     * measured here. */
    for (int i = 0; i < SCALE_N; ++i) {
        float v = (float) ((i % 37) - 18) * 0.375f;
        if ((i % 5) == 0) {
            v *= 8.0f;
        }
        X[i] = (hexlib_hf) v;
    }

    for (int i = 0; i < SCALE_N; ++i) {
        Y[i] = (hexlib_hf) 12345.0f;
    }

    scale_fp16_baseline(X, REF, SCALE_N, SCALE_FACTOR);

    unsigned long long kcyc = 0;
    HEXLIB_TIME_KERNEL(kcyc, scale_fp16(X, Y, SCALE_N, SCALE_FACTOR));

    int n_wrong = 0;
    double max_err = 0.0;
    for (int i = 0; i < SCALE_N; ++i) {
        if (!hexlib_close_f16((float) Y[i], (float) REF[i], 0.02f, 1e-3f)) {
            ++n_wrong;
        }
        double d = (double) (float) Y[i] - (double) (float) REF[i];
        if (d < 0.0) d = -d;
        if (d > max_err) max_err = d;
    }

    hexlib_report(n_wrong == 0, n_wrong, max_err, kcyc);
    return 0;
}

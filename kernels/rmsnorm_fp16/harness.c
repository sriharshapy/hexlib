/* kernels/rmsnorm_fp16/harness.c
 *
 * Builds inputs, runs the baseline for reference, times ONLY the kernel call,
 * compares with tolerance, and prints the two lines the driver parses.
 *
 * Row 0 is deliberately near-degenerate (all values tiny) so that eps actually
 * matters: without it, 1/sqrt(ms) explodes. nearmiss_no_eps.c must fail here.
 */
#include "hexlib/hexlib_harness.h"
#include "kernel_api.h"

void rmsnorm_fp16_baseline(const hexlib_hf *, const hexlib_hf *,
                           hexlib_hf *, int, int, float);

static hexlib_hf X[RMSNORM_R * RMSNORM_C] HEXLIB_ALIGN;
static hexlib_hf W[RMSNORM_C]             HEXLIB_ALIGN;
static hexlib_hf Y[RMSNORM_R * RMSNORM_C] HEXLIB_ALIGN;
static hexlib_hf REF[RMSNORM_R * RMSNORM_C] HEXLIB_ALIGN;

int main(void) {
    for (int c = 0; c < RMSNORM_C; ++c) {
        W[c] = (hexlib_hf) (0.5f + 0.01f * (float) (c % 17));
    }
    for (int r = 0; r < RMSNORM_R; ++r) {
        for (int c = 0; c < RMSNORM_C; ++c) {
            float v;
            if (r == 0) {
                v = 1e-4f * (float) ((c % 5) - 2);   /* forces eps to matter */
            } else {
                v = (float) ((r * 7 + c * 3) % 23) * 0.125f - 1.4375f;
            }
            X[r * RMSNORM_C + c] = (hexlib_hf) v;
        }
    }
    /* Poison the output so a kernel that writes nothing cannot pass. */
    for (int i = 0; i < RMSNORM_R * RMSNORM_C; ++i) {
        Y[i] = (hexlib_hf) 12345.0f;
    }

    rmsnorm_fp16_baseline(X, W, REF, RMSNORM_R, RMSNORM_C, RMSNORM_EPS);

    unsigned long long kcyc = 0;
    HEXLIB_TIME_KERNEL(kcyc,
        rmsnorm_fp16(X, W, Y, RMSNORM_R, RMSNORM_C, RMSNORM_EPS));

    int n_wrong = 0;
    double max_err = 0.0;
    for (int i = 0; i < RMSNORM_R * RMSNORM_C; ++i) {
        if (!hexlib_close_f16(Y[i], REF[i], 0.02f, 1e-3f)) {
            ++n_wrong;
        }
        double d = (double) (float) Y[i] - (double) (float) REF[i];
        if (d < 0.0) d = -d;
        if (d > max_err) max_err = d;
    }

    hexlib_report(n_wrong == 0, n_wrong, max_err, kcyc);
    return 0;
}

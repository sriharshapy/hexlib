/* kernels/softmax_fp16/harness.c
 *
 * Builds inputs, runs the baseline for reference, times ONLY the kernel call,
 * compares with tolerance, and prints the two lines the driver parses.
 *
 * SHAPE: SOFTMAX_R=6, SOFTMAX_C=256. C matches the encoder's real row width
 * (fp16 (12, 256, 256), axis=-1); R is a small representative sample of the
 * encoder's 3072 independent rows, same as layernorm_fp16 sampling R=4 of 256.
 * R != C ON PURPOSE: the encoder's own last two dims ARE square (256x256), so
 * a wrong-axis softmax there produces a same-shape, same-size output that a
 * shape check cannot see. Making this harness's R and C UNEQUAL means a
 * row-softmax and a column-softmax are reducing over different-sized groups
 * (6 vs 256) no matter what the data looks like, so nearmiss_wrong_axis.c
 * cannot pass here by accident -- see kernel_api.h's SHAPE note.
 *
 * ROW 0 IS THE "no max subtraction" TRIGGER. x[0][0] = 90.0f. expf(90) alone
 * overflows float32 (FLT_MAX's ln is ~88.72), so a kernel that exponentiates
 * before subtracting the row max produces +inf, then inf/inf = NaN.
 * nearmiss_no_max_subtraction.c must fail here, loudly, via NaN != anything.
 *
 * ROW 1 IS THE "sum accumulated in fp16" TRIGGER, and it is the one that
 * matters. Read this carefully, because a per-element tolerance loose enough
 * to admit the real kernel's own legitimate noise CAN be looser than a real
 * bug -- that already happened once in this repo (layernorm_fp16's
 * unbiased-variance near-miss was wrongly accepted on its first run because
 * the bug's size, ~0.065% at C=768, was smaller than fp16's own ~0.05% ULP
 * noise). The same risk applies here: summing 256 fp16-rounded exp() values
 * one at a time, rounding to fp16 AFTER EACH addition, can differ from a
 * float32 accumulation by well under one ULP PER ELEMENT on "friendly" data
 * (measured in Python: uniform small scores, max relative difference only
 * ~0.09% -- indistinguishable from ordinary rounding noise, and a tolerance
 * tight enough to catch it there would also reject the correct kernel).
 *
 * So row 1 is not friendly. It is one dominant score (20.0) and 255 IDENTICAL
 * followers at 20.0 - 6.5 = 13.5 -- every follower's exp(shifted) is
 * ~exp(-6.5) = 0.0015, individually tiny against a running sum near 1.0-2.0,
 * which is exactly the shape of input that makes fp16 accumulation lose mass:
 * many increments each close to the accumulator's own ULP, added one at a
 * time, round away a little every single step, 255 times in a row. Measured
 * in Python (float32 exp, float32 vs fp16-per-step accumulation, same
 * algorithm nearmiss_sum_fp16.c implements): the two summed to 1.383384 vs.
 * 1.498047 -- an 8.3% difference in the DENOMINATOR -- which lands as a 7.7%
 * relative / 0.0552 absolute error on the dominant output element alone. That
 * is not a rounding-noise near-miss: it is 90-150x the size of the real
 * kernel's own error on the same row (measured max_abs_err ~4.9e-4, ~1 fp16
 * ULP at that magnitude, from the single unavoidable narrow-to-fp16 rounding
 * every correct implementation pays exactly once). The tolerance below (1%
 * relative, 1e-3 absolute) sits in between with more than an order of
 * magnitude of headroom on each side -- it is not a number arrived at by
 * relaxing it until the kernel passed.
 *
 * ROWS 2-5 are generic, mutually dissimilar fp16 attention-score-shaped data
 * (mixed sign, mixed magnitude) -- coverage, not a discriminator on their own.
 */
#include "hexlib/hexlib_harness.h"
#include "kernel_api.h"

void softmax_fp16_baseline(const hexlib_hf *, hexlib_hf *, int, int);

static hexlib_hf X[SOFTMAX_R * SOFTMAX_C]   HEXLIB_ALIGN;
static hexlib_hf Y[SOFTMAX_R * SOFTMAX_C]   HEXLIB_ALIGN;
static hexlib_hf REF[SOFTMAX_R * SOFTMAX_C] HEXLIB_ALIGN;

static void fill(void) {
    for (int c = 0; c < SOFTMAX_C; ++c) {
        /* Row 0: overflow trigger for "no max subtraction". */
        float v0 = (c == 0) ? 90.0f : 0.1f * (float) ((c % 13) - 6);
        X[0 * SOFTMAX_C + c] = (hexlib_hf) v0;

        /* Row 1: fp16-sum trigger -- one dominant score, 255 identical
         * followers 6.5 below it. See the header comment for the numbers. */
        float v1 = (c == 0) ? 20.0f : (20.0f - 6.5f);
        X[1 * SOFTMAX_C + c] = (hexlib_hf) v1;

        /* Rows 2-5: generic coverage, mutually dissimilar. */
        X[2 * SOFTMAX_C + c] = (hexlib_hf) (2.0f * (float) ((c % 17) - 8) * 0.25f);
        X[3 * SOFTMAX_C + c] = (hexlib_hf) (0.01f * (float) (c % 31));
        X[4 * SOFTMAX_C + c] = (hexlib_hf) (-1.5f + 0.03f * (float) (c % 41));
        X[5 * SOFTMAX_C + c] = (hexlib_hf) (5.0f * (float) ((c * 7) % 19) / 19.0f - 2.5f);
    }
    for (int i = 0; i < SOFTMAX_R * SOFTMAX_C; ++i) {
        Y[i] = (hexlib_hf) 12345.0f;   /* poison: a no-op kernel cannot pass */
    }
}

int main(void) {
    fill();

    softmax_fp16_baseline(X, REF, SOFTMAX_R, SOFTMAX_C);

    unsigned long long kcyc = 0;
    HEXLIB_TIME_KERNEL(kcyc, softmax_fp16(X, Y, SOFTMAX_R, SOFTMAX_C));

    int n_wrong = 0;
    double max_err = 0.0;
    for (int i = 0; i < SOFTMAX_R * SOFTMAX_C; ++i) {
        if (!hexlib_close_f16((float) Y[i], (float) REF[i], 0.01f, 1e-3f)) {
            ++n_wrong;
        }
        double d = (double) (float) Y[i] - (double) (float) REF[i];
        if (d < 0.0) d = -d;
        if (d > max_err) max_err = d;
    }

    hexlib_report(n_wrong == 0, n_wrong, max_err, kcyc);
    return 0;
}

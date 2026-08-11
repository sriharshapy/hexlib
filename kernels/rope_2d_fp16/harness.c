/* kernels/rope_2d_fp16/harness.c
 *
 * T, H, D ARE ALL DIFFERENT NUMBERS (6, 3, 64), for the reason
 * kernels/transpose_th_fp16 gives for its own T=8, H=3: with any two of the
 * three axes equal, a kernel that confuses their strides -- indexing the
 * cos/sin table by head instead of by token is exactly this op's version of
 * that mistake, since the table has no head axis at all -- can still produce
 * the right answer by accident. D=64 is additionally fixed by the encoder's
 * only real shape and by kernel.c's fast path (see kernel_api.h).
 *
 * THE TABLE'S TWO HALVES GENUINELY DIFFER. cos[t,i] and cos[t,i+half] are
 * built from different, unequal formulas below (not HF's `cat(freqs,freqs)`
 * convention, where they coincide) -- see kernel_api.h's "2-D" note. A kernel
 * that assumes the low half's angle also applies to the high half is caught.
 *
 * EVERY (t,h) GETS A DIFFERENT x ROW, so a kernel that mixes up which head's
 * data meets which token's table entry is visible, not masked by repeated
 * data.
 *
 * ==========================================================================
 * THE DISCRIMINATING CHECK: fp16-vs-fp32 accumulation, made STRUCTURAL.
 * ==========================================================================
 * One real near-miss here is subtle on friendly data: rounding each
 * intermediate product (x0*cos, x1*sin) to fp16 before combining, instead of
 * keeping the whole rotation in float/qf32 and rounding only the final
 * result. On generic small values that costs roughly one extra fp16 ULP
 * (~1e-3 relative) on top of the correct kernel's own ~1e-3 narrowing noise
 * (Q6_Vhf_equals_Wqf32 is not IEEE round-to-nearest and differs from numpy by
 * 1 ULP -- see kernel_api.h / hexlib's HARD CONSTRAINTS). A tolerance loose
 * enough to admit that legitimate noise is already loose enough to admit the
 * bug too -- exactly the failure kernels/layernorm_fp16's
 * nearmiss_unbiased_variance.c was WRONGLY ACCEPTED by once, on friendly
 * data, before that kernel's harness was tightened.
 *
 * Rather than chase a tolerance that happens to sit between the two noise
 * floors, this harness makes the DIFFERENCE STRUCTURAL: one token (the last
 * one, t = ROPE_T - 1) is given a NEAR-CANCELLATION at column i=0 (and its
 * pair i=half): both halves of x are set to the same large value (1000.0,
 * exactly representable in fp16), and cos/sin are chosen so the two products
 * being combined are individually large (~500) but nearly equal in magnitude,
 * so the TRUE result is tiny (~0.1):
 *
 *     x0 = x1 = 1000.0
 *     cos[i] = 0.5,  sin[i] = 0.5001        -> y[i]      = 500.0 - 500.1  = -0.1 (true)
 *     cos[i+half] = 0.5,  sin[i+half] = -0.4999  -> y[i+half] = 500.0 + (-499.9) = 0.1 (true)
 *
 * WHY THIS DISCRIMINATES, WORKED OUT ARITHMETICALLY (not just asserted):
 *
 *   fp32/qf32 path (correct kernel and baseline): 500.0 and 500.1 (or 500.0
 *   and -499.9) are each accurate to ~7 decimal digits, so their difference
 *   is accurate to a few times 1e-5 -- the tiny true result survives the
 *   subtraction essentially intact, and only THEN gets rounded to fp16 (whose
 *   ULP near 0.1 is ~1e-4). Error at this point: ~1e-4, well inside any
 *   tolerance this harness uses elsewhere.
 *
 *   fp16-intermediate path (the near-miss): fp16's ULP at magnitude ~500 is
 *   500 * 2^-10 =~ 0.49 (the representable grid there is spaced by 0.25,
 *   since 500 sits in the [256,512) binade). 500.1 is only 0.1 away from
 *   500.0 and 0.15 away from the next grid point 500.25, so it rounds TO
 *   500.0 -- indistinguishable from the other operand. The same happens to
 *   -499.9, which rounds to -500.0. The near-miss's subtraction/addition then
 *   sees 500.0-500.0=0 and 500.0+(-500.0)=0, LOSING THE ENTIRE SIGNAL: it
 *   reports 0.0 where the true answer is -0.1 / +0.1. Error: 0.1 -- a
 *   thousand times the correct kernel's error at the same point, and far
 *   larger than the loose per-element tolerance (rel 0.02, abs 1e-3) used for
 *   every other element in this harness.
 *
 * This is the same principle kernels/layernorm_fp16's C=64-narrow-width pass
 * uses (pick a regime where the bug is bigger, not a tolerance you have
 * talked yourself into) -- applied here as a single crafted VALUE rather than
 * a second narrower SHAPE, since this op has a value-dependent (cancellation)
 * failure mode rather than a size-dependent (1/C vs 1/(C-1)) one.
 */
#include "hexlib/hexlib_harness.h"
#include "kernel_api.h"

#include <math.h>

void rope_2d_fp16_baseline(const hexlib_hf *, const float *, const float *,
                           hexlib_hf *, int, int, int);

#define ROPE_N (ROPE_T * ROPE_H * ROPE_D)

static hexlib_hf X[ROPE_N]        HEXLIB_ALIGN;
static float     COS[ROPE_T * ROPE_D] HEXLIB_ALIGN;
static float     SIN[ROPE_T * ROPE_D] HEXLIB_ALIGN;
static hexlib_hf Y[ROPE_N]        HEXLIB_ALIGN;
static hexlib_hf REF[ROPE_N]      HEXLIB_ALIGN;

int main(void) {
    const int half = ROPE_D / 2;

    /* cos/sin table: the two halves of each row use DIFFERENT angle formulas
     * (mimicking a real 2-D table where the low half carries one spatial axis
     * and the high half the other -- see kernel_api.h's "2-D" note), so
     * cos[t,i] != cos[t,i+half] in general and a kernel that reuses the low
     * half's angle for the high half is caught. */
    for (int t = 0; t < ROPE_T; ++t) {
        for (int i = 0; i < half; ++i) {
            const float angle_lo = 0.15f * (float) (t + 1) * (float) (i + 1);
            COS[t * ROPE_D + i] = cosf(angle_lo);
            SIN[t * ROPE_D + i] = sinf(angle_lo);

            const float angle_hi =
                0.11f * (float) (t + 1) * (float) (i + 1) + 0.83f;
            COS[t * ROPE_D + half + i] = cosf(angle_hi);
            SIN[t * ROPE_D + half + i] = sinf(angle_hi);
        }
    }

    /* x: every (t,h,i) cell distinct, asymmetric, no repeated pattern that
     * could make a wrong pairing or a wrong stride pass by coincidence. */
    for (int t = 0; t < ROPE_T; ++t) {
        for (int h = 0; h < ROPE_H; ++h) {
            for (int i = 0; i < ROPE_D; ++i) {
                const int idx = (t * ROPE_H + h) * ROPE_D + i;
                const float v =
                    (float) (((t * 7 + h * 13 + i * 3) % 23) - 11) * 0.3f;
                X[idx] = (hexlib_hf) v;
            }
        }
    }

    /* The near-cancellation cell: see the header comment above for the
     * arithmetic. Overwritten AFTER the generic fill, at token
     * t=ROPE_T-1, head 0, columns 0 and half. */
    {
        const int t = ROPE_T - 1;
        const int h = 0;
        const int base = (t * ROPE_H + h) * ROPE_D;
        X[base + 0]    = (hexlib_hf) 1000.0f;
        X[base + half] = (hexlib_hf) 1000.0f;
        COS[t * ROPE_D + 0]        = 0.5f;
        SIN[t * ROPE_D + 0]        = 0.5001f;
        COS[t * ROPE_D + half]     = 0.5f;
        SIN[t * ROPE_D + half]     = -0.4999f;
    }

    /* Poison the output so a kernel that writes nothing cannot pass. */
    for (int i = 0; i < ROPE_N; ++i) {
        Y[i] = (hexlib_hf) 12345.0f;
    }

    rope_2d_fp16_baseline(X, COS, SIN, REF, ROPE_T, ROPE_H, ROPE_D);

    unsigned long long kcyc = 0;
    HEXLIB_TIME_KERNEL(kcyc,
        rope_2d_fp16(X, COS, SIN, Y, ROPE_T, ROPE_H, ROPE_D));

    int n_wrong = 0;
    double max_err = 0.0;
    for (int i = 0; i < ROPE_N; ++i) {
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

/* kernels/matmul_epilogue_fp16/harness.c
 *
 * Builds inputs, runs the baseline for reference, times ONLY the kernel calls,
 * compares with tolerance, and prints the two lines the driver parses.
 *
 * THREE SHAPES, THREE ACTIVATIONS, ALL DIFFERENT (M, K, N), K A MULTIPLE OF 32:
 *
 *   Test 1: M=8,  K=64,  N=128, act=none        -- K fits in one 64-lane vector
 *   Test 2: M=12, K=96,  N=160, act=gelu_tanh    -- K spans 3 blocks of 32
 *   Test 3: M=20, K=128, N=96,  act=gelu_erf     -- K spans 4 blocks of 32
 *
 * M, K, N are pairwise distinct within every test (per-test, not just overall)
 * specifically so a transposed-operand bug cannot return a same-shape,
 * same-size, wrong-content answer the way it could if e.g. M == N.
 *
 * ==========================================================================
 * WHY THESE NUMBERS DISCRIMINATE, NOT JUST "LOOK ADVERSARIAL" -- MEASURED.
 * ==========================================================================
 * This repo has been burned once already (layernorm_fp16's unbiased-variance
 * near-miss, ~0.065% error at C=768, WRONGLY ACCEPTED on its first run because
 * that was SMALLER than fp16's own ~0.05% ULP noise). The tolerance below
 * (rel=1%, abs=3e-4) and the two adversarial cells that follow were derived
 * the same way kernels/softmax_fp16 derived its own: measured in Python
 * against the exact algorithm this file and kernel.c implement, not loosened
 * until something passed.
 *
 * CELL 1 -- Test 1, row m=0, output column n=0: THE fp16-ACCUMULATION-OVER-K
 * TRIGGER. a[0,k] = 1.0 for every k (see fill_test1). Column 0's dequantized
 * weight value at k=0 is exactly 1.0 (scale d=1.0, code 9 -> code-8=1); at
 * every k in [1,64) it is exactly 2^-12 = 0.000244140625 (scale d=2^-12, same
 * code 9). So the true (real-number) dot product at (m=0, n=0) is exactly
 *     1.0 + 63 * 2^-12 = 1.015380859375
 * Measured in Python, running the exact two algorithms:
 *   - fp32-accumulate-then-narrow-once (this kernel's own contract): result
 *     1.015625, error +0.000244140625 (+0.024% relative) -- an ordinary
 *     single narrowing-to-fp16 rounding, 0.25 ULP at this magnitude
 *     (ulp(1.0) = 0.0009765625).
 *   - fp16-running-accumulation (nearmiss_fp16_accumulation.c's bug): every
 *     one of the 63 additions of 2^-12 is below half the accumulator's own
 *     ULP once it reaches 1.0, so EVERY one of them rounds back to exactly
 *     1.0 and the near-miss's running sum never leaves 1.0. Final result:
 *     1.0, error -0.015380859375 (-1.51% relative) -- 63x the correct
 *     kernel's own noise, and both the 1% relative AND the 3e-4 absolute
 *     tolerance branches reject it (1.51% > 1%; 0.0156 > 3e-4).
 *
 * CELL 2 -- Test 3, row m=0: THE gelu_tanh/gelu_erf-SWAP TRIGGER. a[0,k] = 0
 * for every k (all of row 0's activation is zero), so the matmul contributes
 * nothing and every output in that row is exactly act(bias[n]) -- a direct
 * probe of the activation function alone. bias[0] is set to exactly -2.7f.
 * Measured in Python (real math.tanh/math.erf, the same formulas kernel_api.h
 * cites): over x in [-8, 8] scanned at 0.01 resolution, the GLOBAL MAXIMUM
 * absolute difference between gelu_tanh(x) and gelu_erf(x) is only 4.73e-4,
 * at x = -2.70 -- genuinely quiet in absolute terms (kernel_api.h's own
 * warning: "these two agree to within ~5e-4 almost everywhere"). But at that
 * SAME x, gelu_erf(-2.7) = -0.0093608..., i.e. a SMALL output magnitude, so:
 *   - fp16 ULP at that output's own magnitude is only 7.6e-6.
 *   - the tanh/erf formula difference there, in fp16, is 4.73e-4 -- SIXTY-TWO
 *     of those ULPs, and 5.2% relative to the correct (erf) value.
 *   - this kernel's OWN legitimate noise at that point (fp32 tanh/erf via
 *     hvx_vec_exp_f32's measured ~1e-6 relative error, one final narrow) is
 *     at most a couple of ULPs there, i.e. order 1e-5 -- far under both the
 *     1% relative and 3e-4 absolute tolerance branches, while the swapped
 *     formula (5.2% relative, 4.7e-4 absolute) fails BOTH.
 * This is exactly why the tolerance uses an ABSOLUTE bound small enough to
 * bite near zero (3e-4, not the 1e-3-or-looser bound that would admit this
 * bug by absolute value alone) alongside the RELATIVE bound that does the
 * real work away from zero -- see hexlib_close_f16's own OR-of-two-branches
 * definition. A single loose absolute tolerance could not do both jobs at
 * once; that is the exact trap this comment exists to name.
 *
 * All other near-misses (swapped nibble order, bias-after-activation,
 * scale-off-by-one) are NOT given a dedicated adversarial cell: measured in
 * Python against this exact algorithm on ordinary (non-adversarial, formula-
 * generated, mixed-sign) weight and activation data, each already disagrees
 * with the correct reference on 60-85% of ALL output elements, at a max
 * relative error in the hundreds-to-millions-of-percent range (a swapped
 * nibble or a scale from the wrong block is not a rounding-sized mistake).
 * They do not need to be quiet to be real bugs; only the two above do.
 */
#include "hexlib/hexlib_harness.h"
#include "kernel_api.h"

#include <string.h>

void matmul_epilogue_fp16_baseline(const hexlib_hf *, const unsigned char *,
                                   const float *, hexlib_hf *,
                                   int, int, int, int);

/* --- q4_0 quantizer, TEST-DATA CONSTRUCTION ONLY -----------------------
 * Not part of the kernel's contract (the kernel and baseline only ever
 * DEQUANTIZE). Implements the same reference formula kernel_api.h cites
 * (llama.cpp's quantize_row_q4_0_ref, adapted): amax-derived scale,
 * round-half-away-from-zero via the classic "+8.5, truncate" trick, clamp to
 * [0,15]. Used to build "generic" blocks from a chosen set of 32 target
 * float values; the two adversarial cells above are built directly (exact
 * scale and code chosen by hand) rather than through this quantizer, because
 * they need EXACT dequantized values, not "whatever this round-trips to".
 */
static void mm_quantize_block(const float *vals32, unsigned char *blk18) {
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) {
        float av = vals32[j] < 0.0f ? -vals32[j] : vals32[j];
        if (av > amax) amax = av;
    }
    __fp16 dh = (__fp16) (amax / -8.0f);
    float id = (float) dh != 0.0f ? 1.0f / (float) dh : 0.0f;
    memcpy(blk18, &dh, sizeof(dh));
    for (int j = 0; j < 16; ++j) {
        int lo = (int) (vals32[j] * id + 8.5f);
        int hi = (int) (vals32[j + 16] * id + 8.5f);
        if (lo < 0) lo = 0; if (lo > 15) lo = 15;
        if (hi < 0) hi = 0; if (hi > 15) hi = 15;
        blk18[2 + j] = (unsigned char) (((hi & 0x0F) << 4) | (lo & 0x0F));
    }
}

/* Build one block by hand: exact scale, exact 32 codes. Used for the two
 * adversarial cells, where the test needs an EXACT dequantized value rather
 * than whatever mm_quantize_block happens to round a target to. */
static void mm_build_block(float d, const int codes[32], unsigned char *blk18) {
    __fp16 dh = (__fp16) d;
    memcpy(blk18, &dh, sizeof(dh));
    for (int j = 0; j < 16; ++j) {
        int lo = codes[j] & 0x0F;
        int hi = codes[j + 16] & 0x0F;
        blk18[2 + j] = (unsigned char) ((hi << 4) | lo);
    }
}

/* ======================= Test 1: M=8, K=64, N=128, act=none ============ */
#define T1_M 8
#define T1_K 64
#define T1_N 128
#define T1_NBLOCKS (T1_N / MM_Q4_0_BLOCK)
#define T1_ROWSTRIDE (T1_NBLOCKS * MM_Q4_0_BLOCK_BYTES)

static hexlib_hf T1_A[T1_M * T1_K]     HEXLIB_ALIGN;
static unsigned char T1_W[T1_K * T1_ROWSTRIDE] HEXLIB_ALIGN;
static float T1_BIAS[T1_N]             HEXLIB_ALIGN;
static hexlib_hf T1_OUT[T1_M * T1_N]   HEXLIB_ALIGN;
static hexlib_hf T1_REF[T1_M * T1_N]   HEXLIB_ALIGN;

static void fill_test1(void) {
    for (int m = 0; m < T1_M; ++m) {
        for (int k = 0; k < T1_K; ++k) {
            /* Row 0 is the fp16-accumulation stress row: a[0,k] = 1.0 for
             * every k, so the dot product at column 0 is a direct sum of
             * that column's dequantized weight values (see CELL 1 above). */
            float v = (m == 0) ? 1.0f : 0.05f * (float) (((m * 7 + k * 3) % 23) - 11);
            T1_A[m * T1_K + k] = (hexlib_hf) v;
        }
    }
    for (int n = 0; n < T1_N; ++n) {
        T1_BIAS[n] = 0.01f * (float) (((n * 5) % 19) - 9);
    }

    for (int k = 0; k < T1_K; ++k) {
        for (int bb = 0; bb < T1_NBLOCKS; ++bb) {
            unsigned char *blk = T1_W + (long) k * T1_ROWSTRIDE + (long) bb * MM_Q4_0_BLOCK_BYTES;
            if (bb == 0) {
                /* CELL 1: column 0 (index 0 of this block) carries the
                 * stress value; the other 31 codes in the block are a
                 * generic varying pattern sharing the same scale. */
                int codes[32];
                for (int j = 0; j < 32; ++j) {
                    codes[j] = (j * 5 + k * 3) % 16;
                }
                codes[0] = 9; /* code-8 = 1 */
                float d = (k == 0) ? 1.0f : 0.000244140625f; /* 1.0 or 2^-12 */
                mm_build_block(d, codes, blk);
            } else {
                float vals[32];
                for (int j = 0; j < 32; ++j) {
                    vals[j] = 0.3f * (float) (((k * 3 + bb * 7 + j) % 13) - 6);
                }
                mm_quantize_block(vals, blk);
            }
        }
    }
}

/* ================= Test 2: M=12, K=96, N=160, act=gelu_tanh ============ */
#define T2_M 12
#define T2_K 96
#define T2_N 160
#define T2_NBLOCKS (T2_N / MM_Q4_0_BLOCK)
#define T2_ROWSTRIDE (T2_NBLOCKS * MM_Q4_0_BLOCK_BYTES)

static hexlib_hf T2_A[T2_M * T2_K]     HEXLIB_ALIGN;
static unsigned char T2_W[T2_K * T2_ROWSTRIDE] HEXLIB_ALIGN;
static float T2_BIAS[T2_N]             HEXLIB_ALIGN;
static hexlib_hf T2_OUT[T2_M * T2_N]   HEXLIB_ALIGN;
static hexlib_hf T2_REF[T2_M * T2_N]   HEXLIB_ALIGN;

static void fill_test2(void) {
    for (int m = 0; m < T2_M; ++m) {
        for (int k = 0; k < T2_K; ++k) {
            float v = 0.04f * (float) (((m * 11 + k * 5) % 29) - 14);
            T2_A[m * T2_K + k] = (hexlib_hf) v;
        }
    }
    for (int n = 0; n < T2_N; ++n) {
        T2_BIAS[n] = 0.02f * (float) (((n * 7) % 23) - 11);
    }
    for (int k = 0; k < T2_K; ++k) {
        for (int bb = 0; bb < T2_NBLOCKS; ++bb) {
            unsigned char *blk = T2_W + (long) k * T2_ROWSTRIDE + (long) bb * MM_Q4_0_BLOCK_BYTES;
            float vals[32];
            for (int j = 0; j < 32; ++j) {
                vals[j] = 0.25f * (float) (((k * 5 + bb * 3 + j * 2) % 17) - 8);
            }
            mm_quantize_block(vals, blk);
        }
    }
}

/* ================= Test 3: M=20, K=128, N=96, act=gelu_erf ============= */
#define T3_M 20
#define T3_K 128
#define T3_N 96
#define T3_NBLOCKS (T3_N / MM_Q4_0_BLOCK)
#define T3_ROWSTRIDE (T3_NBLOCKS * MM_Q4_0_BLOCK_BYTES)

static hexlib_hf T3_A[T3_M * T3_K]     HEXLIB_ALIGN;
static unsigned char T3_W[T3_K * T3_ROWSTRIDE] HEXLIB_ALIGN;
static float T3_BIAS[T3_N]             HEXLIB_ALIGN;
static hexlib_hf T3_OUT[T3_M * T3_N]   HEXLIB_ALIGN;
static hexlib_hf T3_REF[T3_M * T3_N]   HEXLIB_ALIGN;

static void fill_test3(void) {
    for (int m = 0; m < T3_M; ++m) {
        for (int k = 0; k < T3_K; ++k) {
            /* Row 0 is the gelu_tanh/gelu_erf-swap stress row: a[0,k] = 0 for
             * every k, so every output in that row is exactly act(bias[n])
             * (see CELL 2 above). */
            float v = (m == 0) ? 0.0f : 0.03f * (float) (((m * 13 + k * 3) % 31) - 15);
            T3_A[m * T3_K + k] = (hexlib_hf) v;
        }
    }
    for (int n = 0; n < T3_N; ++n) {
        T3_BIAS[n] = 0.05f * (float) (((n * 9) % 27) - 13);
    }
    T3_BIAS[0] = -2.7f; /* CELL 2: exact probe point, see header comment */

    for (int k = 0; k < T3_K; ++k) {
        for (int bb = 0; bb < T3_NBLOCKS; ++bb) {
            unsigned char *blk = T3_W + (long) k * T3_ROWSTRIDE + (long) bb * MM_Q4_0_BLOCK_BYTES;
            float vals[32];
            for (int j = 0; j < 32; ++j) {
                vals[j] = 0.2f * (float) (((k * 7 + bb * 5 + j * 3) % 19) - 9);
            }
            mm_quantize_block(vals, blk);
        }
    }
}

/* rel=1%, abs=3e-4: derived above from measured numbers, not loosened until
 * something passed. See the header comment for CELL 1 and CELL 2's numbers. */
#define MM_REL_TOL 0.01f
#define MM_ABS_TOL 3e-4f

static void poison(hexlib_hf *buf, int n) {
    for (int i = 0; i < n; ++i) {
        buf[i] = (hexlib_hf) 12345.0f;
    }
}

int main(void) {
    fill_test1();
    fill_test2();
    fill_test3();
    poison(T1_OUT, T1_M * T1_N);
    poison(T2_OUT, T2_M * T2_N);
    poison(T3_OUT, T3_M * T3_N);

    matmul_epilogue_fp16_baseline(T1_A, T1_W, T1_BIAS, T1_REF, T1_M, T1_K, T1_N, MM_ACT_NONE);
    matmul_epilogue_fp16_baseline(T2_A, T2_W, T2_BIAS, T2_REF, T2_M, T2_K, T2_N, MM_ACT_GELU_TANH);
    matmul_epilogue_fp16_baseline(T3_A, T3_W, T3_BIAS, T3_REF, T3_M, T3_K, T3_N, MM_ACT_GELU_ERF);

    unsigned long long kcyc1 = 0, kcyc2 = 0, kcyc3 = 0;
    HEXLIB_TIME_KERNEL(kcyc1, matmul_epilogue_fp16(T1_A, T1_W, T1_BIAS, T1_OUT, T1_M, T1_K, T1_N, MM_ACT_NONE));
    HEXLIB_TIME_KERNEL(kcyc2, matmul_epilogue_fp16(T2_A, T2_W, T2_BIAS, T2_OUT, T2_M, T2_K, T2_N, MM_ACT_GELU_TANH));
    HEXLIB_TIME_KERNEL(kcyc3, matmul_epilogue_fp16(T3_A, T3_W, T3_BIAS, T3_OUT, T3_M, T3_K, T3_N, MM_ACT_GELU_ERF));
    unsigned long long kcyc = kcyc1 + kcyc2 + kcyc3;

    int n_wrong = 0;
    double max_err = 0.0;

    hexlib_hf *outs[3]  = { T1_OUT, T2_OUT, T3_OUT };
    hexlib_hf *refs[3]  = { T1_REF, T2_REF, T3_REF };
    int counts[3] = { T1_M * T1_N, T2_M * T2_N, T3_M * T3_N };

    for (int t = 0; t < 3; ++t) {
        for (int i = 0; i < counts[t]; ++i) {
            float yv = (float) outs[t][i];
            float rv = (float) refs[t][i];
            if (!hexlib_close_f16(yv, rv, MM_REL_TOL, MM_ABS_TOL)) {
                ++n_wrong;
            }
            double d = (double) yv - (double) rv;
            if (d < 0.0) d = -d;
            if (d > max_err) max_err = d;
        }
    }

    hexlib_report(n_wrong == 0, n_wrong, max_err, kcyc);
    return 0;
}

#include "kernel_api.h"

#include <math.h>
#include <string.h>

/* Scalar reference. Correct and obvious, never fast.
 *
 * Same contract as kernel.c (see kernel_api.h): fp32 accumulate over K, bias
 * added BEFORE the activation, activation computed in fp32 via libm's real
 * tanhf/erff (not a polynomial approximation -- this file is what kernel.c's
 * own tanh/erf approximations are checked against, so it must not share their
 * approximation error), one narrow to fp16 at the very end.
 *
 * The q4_0 dequant is its OWN copy of the formula (not a call into kernel.c),
 * exactly like kernels/softmax_fp16/baseline.c reimplements its own expf loop
 * rather than sharing kernel.c's polynomial: a baseline that calls into the
 * thing it is checking would not be an independent reference.
 *
 * THE DEQUANTIZED WEIGHT IS ROUNDED THROUGH fp16 HERE TOO, DELIBERATELY.
 * kernel.c's `wbuf` is `hexlib_hf` -- it MUST be, because the only HVX
 * multiply primitive available (`Q6_Wqf32_vmpy_VhfVhf`) takes two fp16
 * vectors as input, so every dequantized weight value is rounded to fp16
 * before it is ever multiplied, not just at the final output. That is a
 * real extra rounding step this kernel's hardware path pays that a
 * hypothetical float32-weight-storage kernel would not, and it must be
 * reflected here for this file to be a fair reference for THIS kernel's
 * actual numeric contract: a baseline that kept the dequantized weight in
 * float32 the whole way through would be measuring a DIFFERENT, more
 * accurate algorithm than the one kernel.c implements, and would
 * systematically look "wrong" by an amount that has nothing to do with a
 * bug -- exactly what was first measured as up to ~2.2e-3 absolute error at
 * K=128 before this fix (a few ULP of weight-rounding noise per term,
 * accumulated over many terms, is not negligible at the largest K this
 * harness tests).
 */
static void mm_dequant_block_baseline(const unsigned char *blk, float *out32) {
    __fp16 d;
    memcpy(&d, blk, sizeof(d));
    const float df = (float) d;
    const unsigned char *qs = blk + 2;
    for (int j = 0; j < 16; ++j) {
        const int lo = (int) (qs[j] & 0x0F) - 8;
        const int hi = (int) ((qs[j] >> 4) & 0x0F) - 8;
        out32[j]      = (float) (hexlib_hf) (df * (float) lo);
        out32[16 + j] = (float) (hexlib_hf) (df * (float) hi);
    }
}

void matmul_epilogue_fp16_baseline(const hexlib_hf *a, const unsigned char *w,
                                   const float *bias, hexlib_hf *out,
                                   int M, int K, int N, int act) {
    if (M <= 0 || K <= 0 || N <= 0 || N % MM_Q4_0_BLOCK != 0) {
        return;
    }
    const int nblocks = N / MM_Q4_0_BLOCK;
    const long row_stride = (long) nblocks * MM_Q4_0_BLOCK_BYTES;

    float wblock[MM_Q4_0_BLOCK];

    for (int m = 0; m < M; ++m) {
        const hexlib_hf *arow = a + (long) m * K;
        hexlib_hf *orow = out + (long) m * N;

        for (int bb = 0; bb < nblocks; ++bb) {
            float acc[MM_Q4_0_BLOCK];
            for (int j = 0; j < MM_Q4_0_BLOCK; ++j) {
                acc[j] = 0.0f;
            }

            const long bb_off = (long) bb * MM_Q4_0_BLOCK_BYTES;
            for (int k = 0; k < K; ++k) {
                const unsigned char *blk = w + (long) k * row_stride + bb_off;
                mm_dequant_block_baseline(blk, wblock);
                const float av = (float) arow[k];
                for (int j = 0; j < MM_Q4_0_BLOCK; ++j) {
                    acc[j] += av * wblock[j];
                }
            }

            for (int j = 0; j < MM_Q4_0_BLOCK; ++j) {
                float r = acc[j] + bias[(long) bb * MM_Q4_0_BLOCK + j];
                if (act == MM_ACT_GELU_TANH) {
                    const double x = (double) r;
                    /* sqrt(2/pi), literal rather than M_PI: M_PI is not
                     * guaranteed to be declared by <math.h> without a
                     * feature-test macro on every toolchain. */
                    const double inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
                    r = (float) (0.5 * x * (1.0 + tanh(inner)));
                } else if (act == MM_ACT_GELU_ERF) {
                    const double x = (double) r;
                    r = (float) (0.5 * x * (1.0 + erf(x / sqrt(2.0))));
                }
                orow[(long) bb * MM_Q4_0_BLOCK + j] = (hexlib_hf) r;
            }
        }
    }
}

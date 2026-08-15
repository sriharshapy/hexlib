/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: accumulating the K-reduction in __fp16 (rounding to fp16
 * after EVERY multiply-add) instead of carrying it in fp32 the whole way
 * through. Everything else is IDENTICAL to kernel.c: same dequant, same bias
 * order, same activation, same final narrow.
 *
 * WHY ANYONE WOULD WRITE IT. The activation operand and the weight are both
 * fp16, so accumulating the running sum in fp16 "matches" everything else in
 * sight, and on a part where fp16 vectors are twice as wide as fp32 ones, it
 * looks like the natural width to keep a running sum in. Nothing about the C
 * source looks unstable; the loop is the same loop.
 *
 * WHY THIS IS THE INTERESTING NEAR-MISS, WITH THE NUMBERS. See harness.c's
 * header, "CELL 1": row 0, column 0 of Test 1 sums 1.0 (k=0) with 63 copies
 * of 2^-12 (k=1..63). The true sum is 1.015380859375. The correct kernel
 * (fp32 accumulate, one narrow) gets 1.015625, error +0.000244140625
 * (+0.024%) -- an ordinary 0.25-ULP narrowing rounding. THIS near-miss's
 * fp16-per-step running sum rounds every one of those 63 additions back to
 * exactly 1.0 (each increment is below half the accumulator's own ULP at
 * that magnitude), so it never leaves 1.0: final error -0.015380859375
 * (-1.51%), 63x the correct kernel's own noise, and well past both the 1%
 * relative and 3e-4 absolute tolerance branches. Measured in Python running
 * this exact algorithm on this exact data (see harness.c).
 */
#include "kernel_api.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <string.h>

#include "hexlib/hvx/hvx-base.h"
#include "hexlib/hvx/hvx-exp.h"
#include "hexlib/hvx/hvx-inverse.h"

#define LANES_FP16 64
#define LANES_FP32 32

static inline void mm_dequant_block(const unsigned char *blk, hexlib_hf *out32) {
    __fp16 d;
    memcpy(&d, blk, sizeof(d));
    const float df = (float) d;
    const unsigned char *qs = blk + 2;
    for (int j = 0; j < 16; ++j) {
        const int lo = (int) (qs[j] & 0x0F) - 8;
        const int hi = (int) ((qs[j] >> 4) & 0x0F) - 8;
        out32[j]      = (hexlib_hf) (df * (float) lo);
        out32[16 + j] = (hexlib_hf) (df * (float) hi);
    }
}

static inline void mm_widen_mul_ordered(HVX_Vector x_bcast, HVX_Vector y,
                                        HVX_Vector *out) {
    HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(y), x_bcast);
    out[0] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
    out[1] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));
}

static inline HVX_Vector mm_narrow_ordered(HVX_Vector lo, HVX_Vector hi) {
    const HVX_Vector zero = Q6_V_vzero();
    HVX_Vector qlo = Q6_Vqf32_vadd_VsfVsf(lo, zero);
    HVX_Vector qhi = Q6_Vqf32_vadd_VsfVsf(hi, zero);
    return Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(qhi, qlo)));
}

static inline HVX_Vector mm_splat_hf(float v) {
    /* float, not hexlib_hf: hexagon-clang rejects __fp16 as a by-value
     * parameter outright (see include/hexlib/hexlib_harness.h's own note
     * on the same restriction). */
    union { hexlib_hf h; unsigned short u; } bits;
    bits.h = (hexlib_hf) v;
    return Q6_Vh_vsplat_R((int) bits.u);
}

/* WRONG: round the running sum down to fp16 and back up to fp32, every
 * single step -- the running accumulator is effectively __fp16, not fp32. */
static inline HVX_Vector mm_round_trip_fp16(HVX_Vector v32) {
    HVX_Vector as16 = mm_narrow_ordered(v32, Q6_V_vzero());
    HVX_Vector one = mm_splat_hf((hexlib_hf) 1.0f);
    HVX_Vector widened[2];
    mm_widen_mul_ordered(one, as16, widened);
    return widened[0];
}

static inline HVX_Vector mm_tanh_f32(HVX_Vector u) {
    HVX_Vector absu = hvx_vec_abs_f32(u);
    HVX_Vector e = hvx_vec_exp_f32(hvx_vec_mul_f32_f32(absu, hvx_vec_splat_f32(-2.0f)));
    HVX_Vector num = hvx_vec_sub_f32_f32(hvx_vec_splat_f32(1.0f), e);
    HVX_Vector den = hvx_vec_add_f32_f32(hvx_vec_splat_f32(1.0f), e);
    HVX_Vector t = hvx_vec_mul_f32_f32(num, hvx_vec_inverse_f32(den));
    HVX_Vector sign_bits = Q6_V_vand_VV(u, Q6_V_vsplat_R(0x80000000));
    return Q6_V_vor_VV(t, sign_bits);
}

static inline HVX_Vector mm_gelu_tanh_f32(HVX_Vector x) {
    HVX_Vector x2 = hvx_vec_mul_f32_f32(x, x);
    HVX_Vector x3 = hvx_vec_mul_f32_f32(x2, x);
    HVX_Vector inner = hvx_vec_add_f32_f32(
        x, hvx_vec_mul_f32_f32(x3, hvx_vec_splat_f32(0.044715f)));
    inner = hvx_vec_mul_f32_f32(inner, hvx_vec_splat_f32(0.7978845608028654f));
    HVX_Vector t = mm_tanh_f32(inner);
    HVX_Vector one_plus_t = hvx_vec_add_f32_f32(hvx_vec_splat_f32(1.0f), t);
    HVX_Vector half_x = hvx_vec_mul_f32_f32(x, hvx_vec_splat_f32(0.5f));
    return hvx_vec_mul_f32_f32(half_x, one_plus_t);
}

static inline HVX_Vector mm_erf_f32(HVX_Vector x) {
    HVX_Vector absx = hvx_vec_abs_f32(x);
    HVX_Vector denom = hvx_vec_add_f32_f32(
        hvx_vec_splat_f32(1.0f),
        hvx_vec_mul_f32_f32(hvx_vec_splat_f32(0.3275911f), absx));
    HVX_Vector t = hvx_vec_inverse_f32(denom);

    HVX_Vector poly = hvx_vec_splat_f32(1.061405429f);
    poly = hvx_vec_mul_f32_f32(poly, t);
    poly = hvx_vec_add_f32_f32(poly, hvx_vec_splat_f32(-1.453152027f));
    poly = hvx_vec_mul_f32_f32(poly, t);
    poly = hvx_vec_add_f32_f32(poly, hvx_vec_splat_f32(1.421413741f));
    poly = hvx_vec_mul_f32_f32(poly, t);
    poly = hvx_vec_add_f32_f32(poly, hvx_vec_splat_f32(-0.284496736f));
    poly = hvx_vec_mul_f32_f32(poly, t);
    poly = hvx_vec_add_f32_f32(poly, hvx_vec_splat_f32(0.254829592f));
    poly = hvx_vec_mul_f32_f32(poly, t);

    HVX_Vector neg_x2 = hvx_vec_mul_f32_f32(
        hvx_vec_mul_f32_f32(absx, absx), hvx_vec_splat_f32(-1.0f));
    HVX_Vector exp_neg_x2 = hvx_vec_exp_f32(neg_x2);
    HVX_Vector erf_abs = hvx_vec_sub_f32_f32(hvx_vec_splat_f32(1.0f),
                                             hvx_vec_mul_f32_f32(poly, exp_neg_x2));
    HVX_Vector sign_bits = Q6_V_vand_VV(x, Q6_V_vsplat_R(0x80000000));
    return Q6_V_vor_VV(erf_abs, sign_bits);
}

static inline HVX_Vector mm_gelu_erf_f32(HVX_Vector x) {
    HVX_Vector arg = hvx_vec_mul_f32_f32(x, hvx_vec_splat_f32(0.7071067811865476f));
    HVX_Vector e = mm_erf_f32(arg);
    HVX_Vector one_plus_e = hvx_vec_add_f32_f32(hvx_vec_splat_f32(1.0f), e);
    HVX_Vector half_x = hvx_vec_mul_f32_f32(x, hvx_vec_splat_f32(0.5f));
    return hvx_vec_mul_f32_f32(half_x, one_plus_e);
}

void matmul_epilogue_fp16(const hexlib_hf *a, const unsigned char *w,
                          const float *bias, hexlib_hf *out,
                          int M, int K, int N, int act) {
    if (M <= 0 || K <= 0 || N <= 0 || N % MM_Q4_0_BLOCK != 0) {
        return;
    }

    const int nblocks = N / MM_Q4_0_BLOCK;
    const long row_stride = (long) nblocks * MM_Q4_0_BLOCK_BYTES;

    hexlib_hf wbuf[LANES_FP16] __attribute__((aligned(128)));
    for (int j = MM_Q4_0_BLOCK; j < LANES_FP16; ++j) {
        wbuf[j] = (hexlib_hf) 0.0f;
    }
    HVX_Vector *wv_slot = (HVX_Vector *) wbuf;

    for (int m = 0; m < M; ++m) {
        const hexlib_hf *arow = a + (long) m * K;
        hexlib_hf *orow = out + (long) m * N;

        for (int bb = 0; bb < nblocks; ++bb) {
            const long bb_off = (long) bb * MM_Q4_0_BLOCK_BYTES;
            HVX_Vector acc_lo = Q6_V_vzero();

            for (int k = 0; k < K; ++k) {
                const unsigned char *blk = w + (long) k * row_stride + bb_off;
                mm_dequant_block(blk, wbuf);
                HVX_Vector wvec = *wv_slot;
                HVX_Vector abcast = mm_splat_hf(arow[k]);

                HVX_Vector prod[2];
                mm_widen_mul_ordered(abcast, wvec, prod);
                acc_lo = hvx_vec_add_f32_f32(acc_lo, prod[0]);
                /* WRONG: round the running sum through fp16 every step. */
                acc_lo = mm_round_trip_fp16(acc_lo);
            }

            HVX_Vector biasv = hvx_vmemu(bias + (long) bb * MM_Q4_0_BLOCK);
            HVX_Vector r = hvx_vec_add_f32_f32(acc_lo, biasv);

            if (act == MM_ACT_GELU_TANH) {
                r = mm_gelu_tanh_f32(r);
            } else if (act == MM_ACT_GELU_ERF) {
                r = mm_gelu_erf_f32(r);
            }

            HVX_Vector outv = mm_narrow_ordered(r, Q6_V_vzero());
            hvx_vec_store_u(orow + (long) bb * MM_Q4_0_BLOCK,
                            MM_Q4_0_BLOCK * (uint32_t) sizeof(hexlib_hf), outv);
        }
    }
}

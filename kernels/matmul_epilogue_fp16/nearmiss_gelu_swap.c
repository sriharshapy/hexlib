/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: using gelu_tanh's formula where gelu_erf belongs, and vice
 * versa -- the MM_ACT_GELU_TANH branch calls mm_gelu_erf_f32 and the
 * MM_ACT_GELU_ERF branch calls mm_gelu_tanh_f32. Everything else, including
 * both activation implementations themselves, is byte-identical to kernel.c.
 *
 * WHY ANYONE WOULD WRITE IT. Both are "the GELU variant", both take one
 * fp32 vector and return one fp32 vector, and a copy-paste or a mixed-up
 * `if`/`else if` branch is the whole bug -- nothing about either function's
 * signature or body hints which activation attribute it belongs under.
 * kernel_api.h says outright that gelu_tanh and gelu_erf are "two different
 * functions used in two different places in the model... not one op with a
 * flag" precisely because this swap is easy to make and easy to miss.
 *
 * WHY THIS IS THE QUIET NEAR-MISS THIS KERNEL HAS TO WORK TO CATCH. See
 * harness.c's header, "CELL 2": gelu_tanh and gelu_erf agree to within
 * 4.73e-4 absolute EVERYWHERE (measured over x in [-8,8] at 0.01
 * resolution, real math.tanh/math.erf) -- smaller, in absolute terms, than
 * one legitimate fp16 rounding at many magnitudes. What makes it catchable
 * is that harness.c's Test 3 row 0 probes x=-2.7 specifically, where the
 * correct (erf) output's own magnitude is small (~0.0094), so that same
 * 4.73e-4 absolute difference is 62 ULPs and 5.2% relative AT THAT POINT --
 * both tolerance branches (1% relative, 3e-4 absolute) reject it there, even
 * though a naive "how big is the raw number" glance at 4.73e-4 in isolation
 * would suggest otherwise.
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
            }

            HVX_Vector biasv = hvx_vmemu(bias + (long) bb * MM_Q4_0_BLOCK);
            HVX_Vector r = hvx_vec_add_f32_f32(acc_lo, biasv);

            /* WRONG: the two branches call each other's formula. */
            if (act == MM_ACT_GELU_TANH) {
                r = mm_gelu_erf_f32(r);
            } else if (act == MM_ACT_GELU_ERF) {
                r = mm_gelu_tanh_f32(r);
            }

            HVX_Vector outv = mm_narrow_ordered(r, Q6_V_vzero());
            hvx_vec_store_u(orow + (long) bb * MM_Q4_0_BLOCK,
                            MM_Q4_0_BLOCK * (uint32_t) sizeof(hexlib_hf), outv);
        }
    }
}

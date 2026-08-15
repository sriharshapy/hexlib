/* matmul + bias + optional activation, q4_0 block-quantized weight.
 * See kernel_api.h for the full spec, citations, and the layout/precision
 * decisions this file implements.
 *
 * STRUCTURE. One q4_0 block (32 output columns) is processed per inner
 * iteration: for a fixed row m and a fixed block of 32 output columns, the
 * kernel walks all of K, dequantizing that row's block to fp16, multiplying
 * by the broadcast activation scalar a[m,k], and accumulating in fp32; after
 * the K loop it adds the bias, applies the activation, and narrows once. This
 * is deliberately the SIMPLEST correct structure, not the fastest one: the
 * weight block at (k, block) is re-dequantized once per output ROW m rather
 * than once and reused across all M rows. Fixing that (dequantize each (k,
 * block) once, hold the M partial sums live across a re-ordered loop) is the
 * obvious next optimisation and is not attempted here -- see kernel_api.h's
 * "WHY HVX-COMPUTE, NOT HMX" for the same "first rung, not a result" framing
 * kernels/softmax_fp16 and kernels/layernorm_fp16 use for their own
 * unfinished optimisations.
 *
 * LANE ORDER: THE SAME SHUFFLE-BEFORE / DEAL-AFTER CONVENTION AS
 * kernels/layernorm_fp16/kernel.c AND kernels/rope_2d_fp16/kernel.c.
 * `Q6_Wqf32_vmpy_VhfVhf` (the only fp16->qf32 widening primitive) and
 * `Q6_Vhf_equals_Wqf32` (the only qf32->fp16 narrowing primitive) both
 * PERMUTE lanes -- element k does not land in lane k (verified against those
 * two kernels' own header comments, same toolchain, same arch). That is
 * invisible to a PURE elementwise op with no other operand (add_fp16,
 * scale_fp16 use the raw primitives directly and get away with it, because
 * widen-then-narrow with no shuffle in between is its own inverse). It is
 * NOT invisible here: after the K-reduction, this kernel adds a bias vector
 * that was loaded PLAINLY from memory (true column order), so the
 * accumulated qf32 partial sums MUST already be in true column order before
 * that add, or bias[n] would land on the wrong accumulated column -- a
 * correctly-shaped, silently-wrong answer no shape check would catch. So
 * every widen here shuffles its non-uniform operand first (`widen_mul_
 * ordered`, adapted from layernorm_fp16's `widen_ordered`), and the one
 * narrow at the very end deals after narrowing (`narrow_ordered`, copied
 * from the same file, cited there as line-by-line verified in kernels/
 * softmax_fp16/kernel.c's own header comment).
 *
 * EXP / TANH / ERF: hvx_vec_exp_f32, NEVER hvx_vec_exp2_f16. hvx_vec_exp2_f16
 * (hvx-exp.h) has a wrong E5 coefficient (0x5082 where upstream calls for
 * 0x090c, 262% error at fractional input 0.7 -- see kernels/softmax_fp16's
 * header for the same finding) and this kernel's gelu_tanh needs a tanh,
 * which this file builds from `hvx_vec_exp_f32` directly (tanh(u) =
 * (1-exp(-2|u|))/(1+exp(-2|u|)), sign copied back in) rather than via
 * hvx-sigmoid.h's `hvx_vec_fast_sigmoid_f16`/`hvx_vec_tanh_f16`, which route
 * through the broken `hvx_vec_exp2_f16`. `hvx_vec_exp_f32` itself is
 * measured at ~1e-6 relative over the input range that matters (kernels/
 * softmax_fp16/kernel.c's header). gelu_erf uses the Abramowitz-Stegun 7.1.26
 * rational approximation to erf (max absolute error in erf itself: 1.5e-7,
 * a standard published approximation, not derived here), also built on
 * `hvx_vec_exp_f32`. The two activations therefore share the same exp
 * primitive but are otherwise genuinely different formulas, per kernel_api.h
 * -- see nearmiss_gelu_swap.c for what happens if they are swapped.
 *
 * NO Q6_Vhf_vadd_VhfVhf anywhere (does not exist on v75, crashes clang
 * 19.0.04 exit code 70 -- add_fp16/kernel.c's header). All fp16/fp32
 * arithmetic here goes through the qf32 path via hvx-base.h's helpers or the
 * hand-rolled widen/narrow below.
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

/* --- q4_0 dequant, one 32-element block, scalar -----------------------
 *
 * blk: 18 bytes -- fp16 scale d, then 16 bytes of packed nibbles.
 * out32: 32 hexlib_hf, written in full (indices [0,32)).
 *
 * Formula from kernel_api.h's SPEC SOURCE 3 (adapted from llama.cpp's
 * dequantize_row_q4_0, cited there): low nibble of byte j -> index j, high
 * nibble of byte j -> index j+16. NOT interleaved. See
 * nearmiss_swapped_nibble_order.c.
 */
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

/* --- ordered widen/narrow, adapted from kernels/layernorm_fp16/kernel.c ---
 * (that file's own header explains why the shuffle/deal is needed; see this
 * file's header for why it applies here too).
 */

/* (x_bcast_fp16 * y_fp16), widened to fp32, TRUE element order.
 * x is assumed UNIFORM across all 64 lanes (a broadcast scalar), so it needs
 * no shuffle of its own -- shuffling a vector whose lanes are all equal is a
 * no-op, exactly how layernorm_fp16's widen_ordered treats its `one` operand.
 * out[0] = elements [0,32) of x*y as fp32; out[1] = elements [32,64). */
static inline void mm_widen_mul_ordered(HVX_Vector x_bcast, HVX_Vector y,
                                        HVX_Vector *out) {
    HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(y), x_bcast);
    out[0] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
    out[1] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));
}

/* Two IEEE fp32 vectors (TRUE element order) -> one fp16 vector (TRUE order). */
static inline HVX_Vector mm_narrow_ordered(HVX_Vector lo, HVX_Vector hi) {
    const HVX_Vector zero = Q6_V_vzero();
    HVX_Vector qlo = Q6_Vqf32_vadd_VsfVsf(lo, zero);
    HVX_Vector qhi = Q6_Vqf32_vadd_VsfVsf(hi, zero);
    return Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(qhi, qlo)));
}

/* One fp16 value splatted to all 64 lanes, as a bit pattern (scale_fp16's
 * own splat_fp16, same reasoning: Q6_Vh_vsplat_R wants the bit pattern, a
 * numeric cast would splat the wrong thing). */
static inline HVX_Vector mm_splat_hf(float v) {
    /* float, not hexlib_hf: hexagon-clang rejects __fp16 as a by-value
     * parameter outright (see include/hexlib/hexlib_harness.h's own note
     * on the same restriction). */
    union { hexlib_hf h; unsigned short u; } bits;
    bits.h = (hexlib_hf) v;
    return Q6_Vh_vsplat_R((int) bits.u);
}

/* --- tanh and erf, both built on hvx_vec_exp_f32 (fp32, accurate) ------ */

/* tanh(u) = (1 - e)/(1 + e), e = exp(-2*|u|); sign copied back in bitwise.
 * Numerically stable for either sign: e is always in (0, 1]. */
static inline HVX_Vector mm_tanh_f32(HVX_Vector u) {
    HVX_Vector absu = hvx_vec_abs_f32(u);
    HVX_Vector e = hvx_vec_exp_f32(hvx_vec_mul_f32_f32(absu, hvx_vec_splat_f32(-2.0f)));
    HVX_Vector num = hvx_vec_sub_f32_f32(hvx_vec_splat_f32(1.0f), e);
    HVX_Vector den = hvx_vec_add_f32_f32(hvx_vec_splat_f32(1.0f), e);
    HVX_Vector t = hvx_vec_mul_f32_f32(num, hvx_vec_inverse_f32(den));  /* tanh(|u|), >= 0 */
    HVX_Vector sign_bits = Q6_V_vand_VV(u, Q6_V_vsplat_R(0x80000000));
    return Q6_V_vor_VV(t, sign_bits);
}

/* gelu_tanh(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))).
 * kernel_api.h SPEC SOURCE 2 / elementwise.py:110-114. */
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

/* erf(x), x>=0 form via Abramowitz & Stegun 7.1.26 (published rational
 * approximation, max abs error 1.5e-7 in erf itself): t=1/(1+p*x),
 * erf(x) = 1 - (a1*t+a2*t^2+a3*t^3+a4*t^4+a5*t^5)*exp(-x^2). Horner form
 * below; sign copied back in bitwise for x<0 (erf is odd). */
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

/* gelu_erf(x) = 0.5*x*(1 + erf(x/sqrt(2))).
 * kernel_api.h SPEC SOURCE 2 / elementwise.py:117-125. */
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
    if (M <= 0 || K <= 0 || N <= 0) {
        return;
    }
    /* ir.nbytes (ir.py:47) already refused a caller whose N is not a
     * multiple of 32 before this weight buffer could even be constructed;
     * this is a defensive check on the kernel's own contract, not a new
     * requirement. */
    if (N % MM_Q4_0_BLOCK != 0) {
        return;
    }

    const int nblocks = N / MM_Q4_0_BLOCK;
    const long row_stride = (long) nblocks * MM_Q4_0_BLOCK_BYTES;

    /* Scratch for one dequantized 32-column block, upper half of the 64-lane
     * fp16 vector left at zero: only ONE block (32 real columns) is live at a
     * time, so the widen-multiply's high 32 lanes always see 0 * anything =
     * 0 and never contribute to the accumulator that gets read out below.
     * Zeroed once, outside every loop -- mm_dequant_block only ever writes
     * indices [0,32). */
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
                /* prod[1] (elements [32,64), always the product of `abcast`
                 * with the zeroed high half of wbuf) is exactly zero and is
                 * never read -- see the wbuf comment above. */
            }

            HVX_Vector biasv = hvx_vmemu(bias + (long) bb * MM_Q4_0_BLOCK);
            HVX_Vector r = hvx_vec_add_f32_f32(acc_lo, biasv);

            if (act == MM_ACT_GELU_TANH) {
                r = mm_gelu_tanh_f32(r);
            } else if (act == MM_ACT_GELU_ERF) {
                r = mm_gelu_erf_f32(r);
            }
            /* act == MM_ACT_NONE: r is already the value to store. */

            HVX_Vector outv = mm_narrow_ordered(r, Q6_V_vzero());
            hvx_vec_store_u(orow + (long) bb * MM_Q4_0_BLOCK,
                            MM_Q4_0_BLOCK * (uint32_t) sizeof(hexlib_hf), outv);
        }
    }
}

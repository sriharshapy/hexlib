/* Row-wise softmax, fp16 in/out, float32 max/exp/sum -- see kernel_api.h for
 * the exact formula and where it comes from.
 *
 * THREE VECTORISED PASSES OVER EACH ROW, mirroring the scalar reference's own
 * three passes (max, then exp+sum, then divide) rather than trying to fuse
 * them -- the max must be known before any exp() call, and the sum must be
 * known before any division, so there is no way to do this in one pass without
 * re-deriving online-softmax's running-rescale trick, which this "first rung"
 * version does not attempt (compare layernorm_fp16's own honesty about its
 * unvectorised reductions).
 *
 * WIDEN/NARROW: reused, not reimplemented. `hvx_vec_f16_to_f32` and
 * `hvx_vec_f32_to_f16` in the vendored include/hexlib/hvx/hvx-base.h already do
 * exactly the shuffle-widen / narrow-deal dance that kernels/layernorm_fp16/
 * kernel.c hand-rolls as `widen_ordered`/`narrow_ordered` -- verified against
 * that file line by line: hvx_vec_f16_to_f32's low lane group is elements
 * 0..31, high is 32..63 (same as layernorm's out[0]/out[1]), and
 * hvx_vec_f32_to_f16(lo, hi) performs the identical
 * qf32-combine-then-Q6_Vh_vdeal_Vh as layernorm's narrow_ordered(lo, hi). Using
 * the header's own versions instead of a local copy means one thing to keep in
 * sync with the vendored source, not two.
 *
 * EXP: hvx_vec_exp_f32, NEVER hvx_vec_exp2_f16. The latter is in the same
 * vendored header (hvx-exp.h) and IS BROKEN -- its E5 coefficient is 0x5082
 * where it should be 0x090c, 262% error at fractional input 0.7, and it is
 * live in llama.cpp's own fp16 flash-attention softmax upstream. hvx_vec_exp_f32
 * is a different function with a different (natural-log-based, degree-7
 * Taylor) polynomial and does not share that bug. Measured in Python against
 * real exp() (see scratch verification, same coefficients, same algorithm):
 * ~1e-6 relative over shifted inputs in [-20, 0], which is every element that
 * still matters after the max subtraction -- exp_f32's own intentional clamp
 * below x=-88 only touches terms so far below the row's max that they
 * underflow to 0 in fp16 regardless of how accurately they are computed.
 *
 * NO Q6_Vhf_vadd_VhfVhf, NO Vhf-typed accumulate anywhere. Per the repo's own
 * hardware notes, that instruction does not exist on v75 and crashes clang
 * 19.0.04 with exit code 70. All fp32 arithmetic here goes through
 * hvx_vec_{add,sub,mul}_f32_f32 (hvx-base.h), which on this arch are
 * Q6_Vsf_equals_Vqf32(Q6_Vqf32_..._VsfVsf(...)) under the hood -- the qf32
 * path, never a native Vhf op.
 *
 * SUM IS float32, ACCUMULATED FROM THE UNROUNDED exp() VALUES, EXACTLY ONCE
 * ROUNDED TO fp16 AT THE FINAL DIVISION. This is the property the
 * fp16-accumulated near-miss (nearmiss_sum_fp16.c) gets wrong, and the property
 * the harness's adversarial row is built to make visible -- see harness.c's
 * header comment and nearmiss_sum_fp16.c's for the numbers.
 */
#include "kernel_api.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>

#include "hexlib/hvx/hvx-base.h"
#include "hexlib/hvx/hvx-exp.h"
#include "hexlib/hvx/hvx-reduce.h"

#define LANES_FP16 64

/* Scratch for one row's unnormalised exp() values, float32, so the sum can be
 * accumulated from full precision and the divide-then-narrow happens exactly
 * once. Fixed-size and 128-byte aligned so the vectorised store/load into it is
 * never the unaligned path. Sized for the encoder's real C=256 with 4x
 * headroom; a row wider than this falls back to a fully scalar per-row
 * computation below rather than corrupting memory or silently truncating --
 * "a kernel that silently mangles a different C is worse than one that is
 * slow" (kernels/layernorm_fp16/kernel.c's own phrase for the same tradeoff). */
#define SOFTMAX_SCRATCH_CAP 1024

static void softmax_row_scalar(const hexlib_hf *xr, hexlib_hf *yr, int C) {
    float m = (float) xr[0];
    for (int c = 1; c < C; ++c) {
        float v = (float) xr[c];
        if (v > m) m = v;
    }
    float s = 0.0f;
    /* Re-derive e[c] on the second pass rather than storing it: this fallback
     * exists only for C beyond the fast path's scratch cap, so it is not on
     * any measured path and trading a second expf() for zero extra memory is
     * the right call here. */
    for (int c = 0; c < C; ++c) {
        s += expf((float) xr[c] - m);
    }
    for (int c = 0; c < C; ++c) {
        yr[c] = (hexlib_hf) (expf((float) xr[c] - m) / s);
    }
}

void softmax_fp16(const hexlib_hf *x, hexlib_hf *y, int R, int C) {
    if (R <= 0 || C <= 0) {
        return;
    }
    if (C > SOFTMAX_SCRATCH_CAP) {
        for (int r = 0; r < R; ++r) {
            softmax_row_scalar(x + (long) r * C, y + (long) r * C, C);
        }
        return;
    }

    float escratch[SOFTMAX_SCRATCH_CAP] __attribute__((aligned(128)));
    const int nvec16 = C / LANES_FP16;         /* fp16 vectors per row */
    const int vecC = nvec16 * LANES_FP16;      /* elements covered by the vector path */

    for (int r = 0; r < R; ++r) {
        const hexlib_hf *xr = x + (long) r * C;
        hexlib_hf *yr = y + (long) r * C;
        const HVX_Vector *xv = (const HVX_Vector *) xr;
        HVX_Vector *ev = (HVX_Vector *) escratch;

        /* --- pass 1: row max, vectorised over the 64-lane blocks --------- */
        float m;
        if (nvec16 > 0) {
            HVX_Vector accmax;
            for (int i = 0; i < nvec16; ++i) {
                HVX_VectorPair p = hvx_vec_f16_to_f32(xv[i]);
                HVX_Vector blockmax = Q6_Vsf_vmax_VsfVsf(Q6_V_lo_W(p), Q6_V_hi_W(p));
                accmax = (i == 0) ? blockmax : Q6_Vsf_vmax_VsfVsf(accmax, blockmax);
            }
            HVX_Vector redmax = hvx_vec_reduce_max_f32(accmax);
            m = hvx_vec_get_f32(redmax);
        } else {
            m = (float) xr[0];
        }
        /* Scalar tail, C % 64 != 0. When nvec16 == 0 the vector path above
         * never ran and m was already seeded from xr[0]; this loop still
         * starts at vecC == 0 in that case and simply re-compares xr[0]
         * against itself once, which is harmless. The encoder's C=256 has no
         * tail (256 = 4*64), so this loop is untested by the harness's main
         * shape and exists only so a future non-multiple-of-64 C is correct
         * rather than lucky. */
        for (int c = vecC; c < C; ++c) {
            float v = (float) xr[c];
            if (v > m) m = v;
        }

        /* --- pass 2: shift, exp, accumulate the sum in float32 ----------- */
        const HVX_Vector vmax = hvx_vec_splat_f32(m);
        HVX_Vector accsum;
        for (int i = 0; i < nvec16; ++i) {
            HVX_VectorPair p = hvx_vec_f16_to_f32(xv[i]);
            HVX_Vector lo = hvx_vec_sub_f32_f32(Q6_V_lo_W(p), vmax);
            HVX_Vector hi = hvx_vec_sub_f32_f32(Q6_V_hi_W(p), vmax);
            HVX_Vector elo = hvx_vec_exp_f32(lo);
            HVX_Vector ehi = hvx_vec_exp_f32(hi);
            ev[2 * i] = elo;
            ev[2 * i + 1] = ehi;
            HVX_Vector blocksum = hvx_vec_add_f32_f32(elo, ehi);
            accsum = (i == 0) ? blocksum : hvx_vec_add_f32_f32(accsum, blocksum);
        }
        float s = (nvec16 > 0) ? hvx_vec_get_f32(hvx_vec_reduce_sum_f32(accsum)) : 0.0f;
        for (int c = vecC; c < C; ++c) {          /* scalar tail */
            float v = expf((float) xr[c] - m);
            escratch[c] = v;
            s += v;
        }

        /* --- pass 3: divide by the sum, narrow once to fp16 --------------- */
        const HVX_Vector vinv = hvx_vec_splat_f32(1.0f / s);
        HVX_Vector *yv = (HVX_Vector *) yr;
        for (int i = 0; i < nvec16; ++i) {
            HVX_Vector olo = hvx_vec_mul_f32_f32(ev[2 * i], vinv);
            HVX_Vector ohi = hvx_vec_mul_f32_f32(ev[2 * i + 1], vinv);
            yv[i] = hvx_vec_f32_to_f16(olo, ohi);
        }
        for (int c = vecC; c < C; ++c) {          /* scalar tail */
            yr[c] = (hexlib_hf) (escratch[c] / s);
        }
    }
}

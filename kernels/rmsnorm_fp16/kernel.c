/* WINNER of the bake-off recorded in BAKEOFF.md: 2020 kernel cycles vs the
 * scalar baseline's 69438 (34.38x) and vs the other HVX candidate's 10498
 * (5.20x). Adapted from HVX-clean/data/v6/tasks/rmsnorm_gain_fp16/expert.c
 * (solutions/s2.c: vector-native ror-shift horizontal reduce for
 * sum-of-squares, R=6 C=80, recorded 9798 kernel cycles, 2.741x).
 *
 * WHY IT IS FAST. Both HVX candidates vectorize the sum-of-squares reduction
 * and the scale epilogue identically -- the difference that decides this
 * bake-off is how the 64-lane qf16 accumulator is collapsed to one scalar.
 * This kernel finishes the horizontal reduction entirely IN THE VECTOR UNIT:
 * a six-step ror-shift butterfly (`hreduce_qf16`, rotate-by 64/32/16/8/4/2
 * lanes with a qf16 add at each step) leaves every lane holding the full sum,
 * so reading lane 0 into a scalar is the only vector-to-scalar transition in
 * the whole reduction. The losing candidate (fp16_rmsnorm's expert, see
 * BAKEOFF.md) instead unpacks the accumulator to memory and finishes with a
 * 64-iteration SCALAR add loop, once per row. That scalar loop is the
 * expert.c-reported difference between the two mechanisms in the original
 * v6 corpus (solutions/s2.c: 9798 cyc vs. solutions/s1.c-style extract: much
 * higher), and it is amplified here: hexlib's contract calls the kernel once
 * for R=8 rows, so the 64-scalar-add cost is paid eight times instead of the
 * single call the original n=2048 shape amortized it over. The gain multiply
 * is folded into the same per-block vector pass as the scale (x*inv_rms then
 * *w[block], two qf16 vector multiplies, no extra full pass over the data).
 *
 * ADAPTATION NOTES (see BAKEOFF.md for the full discussion):
 *  1. The original's gain is PER-ROW: gain[r] is a single scalar splatted
 *     and folded into inv_rms once per row, then broadcast uniformly across
 *     all C columns. hexlib's rmsnorm_fp16 contract is PER-COLUMN: w[c] is
 *     shared across all rows and varies per column. A scalar splat cannot
 *     express a per-column vector, so the scale epilogue was rewritten to
 *     splat only inv_rms and then do a real per-block VECTOR*VECTOR multiply
 *     against w[block] -- structurally the same move fp16_rmsnorm's expert
 *     already makes for its per-feature gamma.
 *  2. The original's reduction only ever read block 0 (`if (nb > 0) { ...
 *     xv[0] ... }`), which is correct ONLY because R=6,C=80 has exactly one
 *     full 64-lane block. C=128 here is two full blocks, so reading only
 *     block 0 would silently drop half the input. Generalized to accumulate
 *     sum-of-squares across all nb blocks (the multi-block accumulate
 *     fp16_rmsnorm's expert uses) and THEN apply the original's real
 *     differentiator -- the six-step ror-shift butterfly (`hreduce_qf16`)
 *     -- exactly once, to the final accumulated vector.
 *  3. eps was hardcoded to 1e-3f in the original; threaded through as the
 *     `eps` parameter instead, since hexlib's harness varies it and row 0
 *     is deliberately built to make eps matter.
 *  4. `harness_common.h` is not included: nothing in the kernel body used
 *     it (no HVX_ALIGN, no hvx_report, no HVX_TIME_KERNEL) -- only
 *     hexagon_types.h/hexagon_protos.h (Q6 intrinsics, HVX_Vector) and
 *     math.h (sqrtf) were actually load-bearing.
 */
#include "kernel_api.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <math.h>

#define BLK 64

static inline HVX_Vector hf_splat_f(float f) {
    hexlib_hf v = (hexlib_hf) f;
    unsigned short bits = *(const unsigned short *) &v;
    return Q6_Vh_vsplat_R((int) bits);
}

/* Six-step ror-shift butterfly: after this, every lane of the returned
 * vector holds the full horizontal sum of the input vector's 64 lanes. This
 * is candidate A's real distinguishing mechanism -- the v6 corpus measured
 * it at 9798 cycles vs. 16047 for the extract-and-scalar-sum alternative
 * (solutions/s1.c) on the original R=6,C=80 shape. */
static inline HVX_Vector hreduce_qf16(HVX_Vector v) {
    v = Q6_Vqf16_vadd_Vqf16Vqf16(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vqf16_vadd_Vqf16Vqf16(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vqf16_vadd_Vqf16Vqf16(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vqf16_vadd_Vqf16Vqf16(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vqf16_vadd_Vqf16Vqf16(v, Q6_V_vror_VR(v, 4));
    v = Q6_Vqf16_vadd_Vqf16Vqf16(v, Q6_V_vror_VR(v, 2));
    return v;
}

void rmsnorm_fp16(const hexlib_hf *x, const hexlib_hf *w,
                  hexlib_hf *y, int R, int C, float eps) {
    int nb = C / BLK;   /* RMSNORM_C=128 -> nb=2, no remainder */

    const HVX_Vector *wv = (const HVX_Vector *) w;

    for (int r = 0; r < R; ++r) {
        const HVX_Vector *xv = (const HVX_Vector *) (x + (long) r * C);
        HVX_Vector *ov       = (HVX_Vector *) (y + (long) r * C);

        HVX_Vector accSq;
        if (nb > 0) {
            HVX_Vector v0 = xv[0];
            accSq = Q6_Vqf16_vmpy_VhfVhf(v0, v0);
            for (int b = 1; b < nb; ++b) {
                HVX_Vector v = xv[b];
                HVX_Vector sq = Q6_Vqf16_vmpy_VhfVhf(v, v);
                accSq = Q6_Vqf16_vadd_Vqf16Vqf16(accSq, sq);
            }
        } else {
            accSq = Q6_Vqf16_vmpy_VhfVhf(xv[0], xv[0]); /* never hit: C multiple of 64 */
        }

        HVX_Vector red   = hreduce_qf16(accSq);
        HVX_Vector redHf = Q6_Vhf_equals_Vqf16(red);
        const hexlib_hf *rp = (const hexlib_hf *) &redHf;
        float sumsq = (float) rp[0];   /* every lane holds the full sum */

        float ms      = sumsq / (float) C;
        float inv_rms = 1.0f / sqrtf(ms + eps);
        HVX_Vector invVec = hf_splat_f(inv_rms);

        for (int b = 0; b < nb; ++b) {
            HVX_Vector v    = xv[b];
            HVX_Vector sc   = Q6_Vqf16_vmpy_VhfVhf(v, invVec);      /* x * inv_rms */
            HVX_Vector scHf = Q6_Vhf_equals_Vqf16(sc);
            HVX_Vector g    = Q6_Vqf16_vmpy_VhfVhf(scHf, wv[b]);    /* * w[block] */
            ov[b] = Q6_Vhf_equals_Vqf16(g);
        }
    }
}

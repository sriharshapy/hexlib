/* Narrow fp32 to fp16, two input vectors at a time.
 *
 * WHY IT IS FAST. Pure format conversion with no arithmetic, so the work is
 * bounded by bytes touched. Each iteration reads two 128-byte vectors (32 fp32
 * lanes each) and writes one (64 fp16 lanes), converting 64 elements per
 * narrowing instruction instead of one per scalar cast.
 *
 * WHY THE ADD OF ZERO IS NOT WASTE. There is no direct IEEE-single to IEEE-half
 * narrowing available on v75: Q6_Vhf_equals_Wsf does not exist. The narrowing
 * that DOES exist takes a qf32 pair (Q6_Vhf_equals_Wqf32), and the way to get
 * qf32 from sf is an arithmetic op that consumes sf and produces qf32 --
 * Q6_Vqf32_vadd_VsfVsf against a zero vector. So `x + 0.0` is the conversion,
 * not an operation on the value. IEEE +0.0 is all-zero bits, which is what
 * Q6_V_vzero gives, and adding it changes nothing (including for negative zero,
 * where -0.0 + 0.0 is +0.0 -- the sign of zero is not something this op is
 * required to preserve, and the scalar baseline's C cast makes the same choice).
 *
 * THE NARROWING INTERLEAVES ITS LANES, AND THE DEAL IS NOT OPTIONAL.
 * Q6_Vhf_equals_Wqf32 does not concatenate the two input vectors' lanes -- it
 * emits them shuffled, so lane k of the result does not come from element k.
 * Q6_Vh_vdeal_Vh undoes that.
 *
 * This was established by measurement, not by reading: without the deal the
 * harness reported 7835 of 8293 elements wrong with a max error of 9.25, and
 * swapping the two halves did NOT fix it -- which is the tell that the problem is
 * an interleave rather than an ordering. Neither the types nor the intrinsic
 * name says any of this. Both orders produce an output of exactly the right
 * length full of correctly-converted numbers in the wrong positions, so only the
 * values catch it. There is a near-miss for the version without the deal,
 * because that is the mistake this kernel actually shipped with first.
 */
#include "kernel_api.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>

#define LANES_FP32 32
#define LANES_FP16 64

void cast_f32_f16(const float *x, hexlib_hf *y, int n) {
    if (n <= 0) {
        return;
    }

    const HVX_Vector zero = Q6_V_vzero();
    const int nout = n / LANES_FP16;   /* whole fp16 output vectors */
    const HVX_Vector *xv = (const HVX_Vector *) x;
    HVX_Vector *yv = (HVX_Vector *) y;

    for (int i = 0; i < nout; ++i) {
        /* Two fp32 vectors feed one fp16 vector. */
        HVX_Vector lo = Q6_Vqf32_vadd_VsfVsf(xv[2 * i], zero);
        HVX_Vector hi = Q6_Vqf32_vadd_VsfVsf(xv[2 * i + 1], zero);
        HVX_Vector shuffled = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(hi, lo));
        yv[i] = Q6_Vh_vdeal_Vh(shuffled);
    }

    for (int i = nout * LANES_FP16; i < n; ++i) {
        y[i] = (hexlib_hf) x[i];
    }
}

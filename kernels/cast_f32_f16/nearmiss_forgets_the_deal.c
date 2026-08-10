/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: omitting Q6_Vh_vdeal_Vh after the narrowing.
 *
 * THIS IS NOT A HYPOTHETICAL. It is the version this kernel was first written
 * as, and it failed the harness with 7835 of 8293 elements wrong and a max error
 * of 9.25. It is preserved as a near-miss because everything about it looks
 * right: the widening is correct, the two-vectors-in-one-vector-out stepping is
 * correct, the combine order is correct, and the output length is correct. The
 * only defect is that Q6_Vhf_equals_Wqf32 emits its lanes INTERLEAVED, so
 * element k does not land in lane k.
 *
 * Nothing in the intrinsic's name or signature says so. The vendored reference
 * library encodes it only in a function name -- `hvx_vec_f32_to_f16_shuff` is
 * this sequence, and `hvx_vec_f32_to_f16` is the same thing with the deal applied
 * -- so the distinction is available to someone who notices a suffix and invisible
 * to someone who does not.
 *
 * Diagnostically the useful detail is that swapping the two halves does NOT fix
 * it. A wrong ORDER would be repaired by a swap; a wrong INTERLEAVE is not. That
 * is how the real cause was identified.
 */
#include "kernel_api.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>

#define LANES_FP16 64

void cast_f32_f16(const float *x, hexlib_hf *y, int n) {
    if (n <= 0) {
        return;
    }
    const HVX_Vector zero = Q6_V_vzero();
    const int nout = n / LANES_FP16;
    const HVX_Vector *xv = (const HVX_Vector *) x;
    HVX_Vector *yv = (HVX_Vector *) y;

    for (int i = 0; i < nout; ++i) {
        HVX_Vector lo = Q6_Vqf32_vadd_VsfVsf(xv[2 * i], zero);
        HVX_Vector hi = Q6_Vqf32_vadd_VsfVsf(xv[2 * i + 1], zero);
        /* WRONG: the narrowing's output is interleaved and is left that way. */
        yv[i] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(hi, lo));
    }
    for (int i = nout * LANES_FP16; i < n; ++i) {
        y[i] = (hexlib_hf) x[i];
    }
}

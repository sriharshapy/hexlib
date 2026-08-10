/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: Q6_W_vcombine_VV(lo, hi) instead of (hi, lo).
 *
 * WHY ANYONE WOULD WRITE IT. The combine takes two vectors and the narrowing
 * takes the pair; nothing in the types says which argument becomes the high half
 * of the result, and reading left-to-right suggests the first argument is the
 * first (lower) 32 lanes. It is the other way round. The output has exactly the
 * right length and every value in it is a correctly converted fp16 number -- they
 * are just in the wrong 32-lane blocks, which is why no length check, shape check
 * or dtype check anywhere can catch it.
 *
 * The harness's inputs are monotonic in the index precisely so this produces a
 * visible discontinuity rather than two plausible neighbouring values.
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
        /* WRONG: halves swapped. The deal IS present, so this isolates the swap
         * rather than failing for two reasons at once. */
        yv[i] = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(lo, hi)));
    }
    for (int i = nout * LANES_FP16; i < n; ++i) {
        y[i] = (hexlib_hf) x[i];
    }
}

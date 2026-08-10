/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: advancing the input by ONE vector per output vector instead of
 * two.
 *
 * WHY ANYONE WOULD WRITE IT. Every other elementwise kernel in this set steps
 * input and output together, one vector for one vector, because the element width
 * does not change. A narrowing op is the exception: 32 fp32 lanes in, but 64 fp16
 * lanes out, so the input pointer has to move twice as fast. `xv[i]` is the habit
 * and `xv[2*i]` is the correction.
 *
 * The result reads only the first half of the input and writes a full-length
 * output, so the second half of the array is silently never converted and the
 * first half appears twice. The length is right, so only the values catch it.
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
        /* WRONG: one input vector per output vector. */
        HVX_Vector lo = Q6_Vqf32_vadd_VsfVsf(xv[i], zero);
        HVX_Vector hi = Q6_Vqf32_vadd_VsfVsf(xv[i], zero);
        yv[i] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(hi, lo));
    }
    for (int i = nout * LANES_FP16; i < n; ++i) {
        y[i] = (hexlib_hf) x[i];
    }
}

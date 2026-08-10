/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: assuming n is a multiple of the 64-lane fp16 vector, so the last
 * n % 64 elements are never written.
 *
 * WHY IT WOULD SURVIVE A CARELESS HARNESS. Every add in the encoder is
 * [256, 768] = 196608 elements, which IS a multiple of 64. A harness sized from
 * the real caller would never execute the tail, this would pass the gate, and the
 * first caller with a different length would get a partially-written buffer.
 * ADD_N is 4100 = 64*64 + 4 so the tail exists, and the harness poisons the
 * output so the four untouched elements hold an impossible value rather than a
 * plausible one.
 */
#include "kernel_api.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>

#define LANES_FP16 64
#define FP16_ONE   0x3C00

static inline HVX_Vector add_hf(HVX_Vector a, HVX_Vector b, HVX_Vector one) {
    HVX_VectorPair ap = Q6_Wqf32_vmpy_VhfVhf(a, one);
    HVX_VectorPair bp = Q6_Wqf32_vmpy_VhfVhf(b, one);
    HVX_Vector lo = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(ap), Q6_V_lo_W(bp));
    HVX_Vector hi = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(ap), Q6_V_hi_W(bp));
    return Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(hi, lo));
}

void add_fp16(const hexlib_hf *a, const hexlib_hf *b, hexlib_hf *y, int n) {
    if (n <= 0) {
        return;
    }
    const HVX_Vector one = Q6_Vh_vsplat_R(FP16_ONE);
    const int nvec = n / LANES_FP16;
    const HVX_Vector *av = (const HVX_Vector *) a;
    const HVX_Vector *bv = (const HVX_Vector *) b;
    HVX_Vector *yv = (HVX_Vector *) y;

    for (int i = 0; i < nvec; ++i) {
        yv[i] = add_hf(av[i], bv[i], one);
    }
    /* WRONG: no tail loop. */
}

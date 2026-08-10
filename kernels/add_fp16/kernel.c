/* Elementwise fp16 add, 64 lanes at a time.
 *
 * WHY IT IS FAST. Two loads, one add, one store, per 64 elements. There is no
 * reduction, no cross-lane dependency and nothing to reorder, so the kernel is
 * bounded by how many bytes it can move rather than by arithmetic: a 128-byte
 * vector carries 64 fp16 lanes, so one instruction replaces 64 scalar adds, and
 * aligned accesses cost one instruction each rather than the two an unaligned
 * pair needs. Nothing cleverer is available for this op -- an add has no
 * arithmetic to save.
 *
 * THERE IS NO fp16 ADD INSTRUCTION ON v75. Q6_Vhf_vadd_VhfVhf exists only for
 * __HVX_ARCH__ >= 79; on v75 it does not select, and this toolchain
 * (hexagon-clang 19.0.04) does not say so -- it crashes in instruction
 * selection with exit code 70 rather than reporting an unavailable intrinsic.
 * So the qf32 round trip below is FORCED BY THE ISA, not a precision choice.
 *
 * The sequence is: widen each operand to qf32 by multiplying it by 1.0 (the only
 * fp16->qf32 widening available is a multiply), add the two halves in qf32, then
 * narrow the pair back to IEEE fp16. Six instructions per 64 lanes instead of
 * one, which is simply what the part costs.
 *
 * It is also exactly right numerically, which is worth stating so nobody
 * "optimises" it later: the exact sum of two fp16 values is always representable
 * in fp32, so the qf32 add is exact and the single rounding happens on the
 * narrow. An fp16 add instruction, if the part had one, would give bit-identical
 * results -- so this costs speed and nothing else.
 *
 * THE Vhf / Vh DISTINCTION IS THE TRAP HERE. `hf` is 16-bit FLOAT; `h` is 16-bit
 * INTEGER. Q6_Vh_vadd_VhVh does exist on v75, takes and returns the same
 * HVX_Vector type, compiles without a warning, and silently adds the fp16 bit
 * patterns as signed integers. There is a near-miss for exactly this.
 */
#include "kernel_api.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>

#define LANES_FP16 64

/* IEEE fp16 1.0, as a bit pattern for Q6_Vh_vsplat_R. */
#define FP16_ONE 0x3C00

/* a + b, for 64 fp16 lanes, via qf32. */
static inline HVX_Vector add_hf(HVX_Vector a, HVX_Vector b, HVX_Vector one) {
    /* Multiplying by 1.0 is the widening: there is no fp16->qf32 convert, but
     * the fp16 multiply already produces a qf32 PAIR, so x*1.0 is x widened. */
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

    for (int i = nvec * LANES_FP16; i < n; ++i) {
        y[i] = (hexlib_hf) ((float) a[i] + (float) b[i]);
    }
}

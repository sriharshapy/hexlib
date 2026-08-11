/* kernels/rope_2d_fp16/kernel.c
 *
 * WHAT IS VECTORISED AND WHAT IS NOT. This op has no reduction at all -- it is
 * purely elementwise -- so unlike kernels/layernorm_fp16 there is no scalar
 * "next thing to fix". The FAST PATH below is fully vectorised for the one
 * shape the encoder ever calls this with (D == 64). Any other D falls back to
 * a scalar loop, same order of work as layernorm_fp16's documented tail: a
 * kernel that silently mangled an untested shape would be worse than one that
 * is merely slow there.
 *
 * WHY D == 64 IS SPECIAL. The pairing is split-half: column i pairs with
 * column i+D/2 (see kernel_api.h for the citation). For D == 64, half == 32,
 * which is exactly LANES_FP32 -- so the whole pairing lives inside ONE 64-lane
 * fp16 vector, and widening it to two 32-lane fp32 halves via widen_ordered
 * below splits it EXACTLY at the pairing boundary: part[0] holds columns
 * [0,32), part[1] holds columns [32,64). Those line up 1:1 with cos/sin's own
 * 32-lane fp32 vectors (crv[0]/srv[0] = columns [0,32), crv[1]/srv[1] =
 * columns [32,64)), so the whole rotation is four vector multiplies, an add,
 * a subtract and one narrow -- no cross-vector shuffling of the pairing
 * itself is needed. For any other multiple of 64, half would not equal 32 and
 * this alignment would not hold in general, which is why the fast path is
 * gated on D == 64 exactly rather than "D is a multiple of 64".
 *
 * WHY THE CONVERSIONS LOOK LIKE THAT. Adapted VERBATIM from
 * kernels/layernorm_fp16/kernel.c's `widen_ordered`/`narrow_ordered` helpers,
 * for the same reason they exist there: there is no fp16<->qf32 convert
 * instruction (widening is a multiply by 1.0, narrowing is an add of 0.0), and
 * both the widen (Q6_Wqf32_vmpy_VhfVhf) and the narrow (Q6_Vhf_equals_Wqf32)
 * PERMUTE lanes -- element k does not land in lane k. Q6_Vh_vshuff_Vh before
 * the widen and Q6_Vh_vdeal_Vh after the narrow put it back. Lane order is
 * load-bearing HERE, exactly as it is in layernorm_fp16's affine epilogue and
 * for the same reason: cos[t,i] and sin[t,i] are PER-COLUMN, so column i of x
 * must meet column i (not some shuffled position) of cos/sin. Getting that
 * wrong yields a correctly-shaped output where the rotation angle applied to
 * a column is not the one that column was given, which no shape check can
 * see -- exactly the failure mode layernorm_fp16's
 * nearmiss_permuted_affine_lanes.c demonstrates for its own per-column w/b.
 *
 * v75 HAS NO fp16 ADD/SUB INSTRUCTION (Q6_Vhf_vadd_VhfVhf arrives at
 * __HVX_ARCH__ 79 and crashes hexagon-clang 19.0.04's instruction selection
 * with exit code 70 if used here). All arithmetic below is qf32:
 * Q6_Vqf32_vmpy_VsfVsf multiplies two IEEE fp32 vectors into qf32,
 * Q6_Vqf32_vsub_Vqf32Vqf32/Q6_Vqf32_vadd_Vqf32Vqf32 combine two qf32 vectors,
 * and Q6_Vsf_equals_Vqf32 narrows qf32 back to IEEE fp32 before the final
 * fp16 narrow (which itself goes through the qf32-pair narrow, per
 * narrow_ordered).
 */
#include "kernel_api.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>

#define LANES_FP32 32
#define LANES_FP16 64
#define FP16_ONE   0x3C00

/* fp16 vector -> two IEEE fp32 vectors, IN ELEMENT ORDER.
 * out[0] holds elements 0..31, out[1] holds 32..63.
 * Adapted verbatim from kernels/layernorm_fp16/kernel.c. */
static inline void widen_ordered(HVX_Vector v, HVX_Vector one, HVX_Vector *out) {
    HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v), one);
    out[0] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
    out[1] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));
}

/* Two IEEE fp32 vectors -> one fp16 vector, IN ELEMENT ORDER.
 * Adapted verbatim from kernels/layernorm_fp16/kernel.c. */
static inline HVX_Vector narrow_ordered(HVX_Vector lo, HVX_Vector hi) {
    const HVX_Vector zero = Q6_V_vzero();
    HVX_Vector qlo = Q6_Vqf32_vadd_VsfVsf(lo, zero);
    HVX_Vector qhi = Q6_Vqf32_vadd_VsfVsf(hi, zero);
    return Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(qhi, qlo)));
}

void rope_2d_fp16(const hexlib_hf *x, const float *costab, const float *sintab,
                  hexlib_hf *y, int T, int H, int D) {
    if (T <= 0 || H <= 0 || D <= 0) {
        return;
    }
    const int half = D / 2;

    if (D == LANES_FP16) {
        const HVX_Vector one = Q6_Vh_vsplat_R(FP16_ONE);

        for (int t = 0; t < T; ++t) {
            /* cos[t,:] and sin[t,:] loaded once per token, reused for every
             * head -- the table has no head axis (kernel_api.h). */
            const HVX_Vector *crv = (const HVX_Vector *) (costab + (long) t * D);
            const HVX_Vector *srv = (const HVX_Vector *) (sintab + (long) t * D);
            const HVX_Vector vc0 = crv[0];   /* cos[t, 0:32]  */
            const HVX_Vector vc1 = crv[1];   /* cos[t, 32:64] */
            const HVX_Vector vs0 = srv[0];   /* sin[t, 0:32]  */
            const HVX_Vector vs1 = srv[1];   /* sin[t, 32:64] */

            for (int h = 0; h < H; ++h) {
                const HVX_Vector *xv =
                    (const HVX_Vector *) (x + ((long) t * H + h) * D);
                HVX_Vector *yv = (HVX_Vector *) (y + ((long) t * H + h) * D);

                HVX_Vector part[2];
                widen_ordered(xv[0], one, part);
                /* part[0] = x[t,h,0:32], part[1] = x[t,h,32:64] */

                /* y[i]      = x[i]     *cos[i]      - x[i+half]*sin[i]      */
                HVX_Vector lo = Q6_Vsf_equals_Vqf32(
                    Q6_Vqf32_vsub_Vqf32Vqf32(
                        Q6_Vqf32_vmpy_VsfVsf(part[0], vc0),
                        Q6_Vqf32_vmpy_VsfVsf(part[1], vs0)));

                /* y[i+half] = x[i+half]*cos[i+half] + x[i]     *sin[i+half] */
                HVX_Vector hi = Q6_Vsf_equals_Vqf32(
                    Q6_Vqf32_vadd_Vqf32Vqf32(
                        Q6_Vqf32_vmpy_VsfVsf(part[1], vc1),
                        Q6_Vqf32_vmpy_VsfVsf(part[0], vs1)));

                yv[0] = narrow_ordered(lo, hi);
            }
        }
        return;
    }

    /* Scalar fallback for D != 64. Never exercised by the encoder (this op
     * is always called at head_dim=64); kept correct rather than omitted. */
    for (int t = 0; t < T; ++t) {
        const float *cr = costab + (long) t * D;
        const float *sr = sintab + (long) t * D;
        for (int h = 0; h < H; ++h) {
            const hexlib_hf *xr = x + ((long) t * H + h) * D;
            hexlib_hf *yr = y + ((long) t * H + h) * D;
            for (int i = 0; i < half; ++i) {
                const float x0 = (float) xr[i];
                const float x1 = (float) xr[i + half];
                yr[i]        = (hexlib_hf) (x0 * cr[i]        - x1 * sr[i]);
                yr[i + half] = (hexlib_hf) (x1 * cr[i + half] + x0 * sr[i + half]);
            }
        }
    }
}

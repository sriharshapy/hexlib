/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: widening fp16 to fp32 without the shuffle that puts lanes back in
 * element order, so the affine parameters meet the wrong columns.
 *
 * WHY ANYONE WOULD WRITE IT. Q6_Wqf32_vmpy_VhfVhf is the widening, and it looks
 * complete on its own -- it takes an fp16 vector and produces a qf32 pair, which
 * is exactly what was wanted. That it PERMUTES the lanes on the way is recorded
 * nowhere in its name or signature. Upstream marks the distinction only by
 * calling the unshuffled helper `hvx_vec_f16_to_f32_shuff`.
 *
 * WHY IT SURVIVES SO MUCH SCRUTINY. The output has the right shape, the right
 * dtype, the right length, and every value in it is a plausible normalised
 * number. The mean and variance are also still exactly right, because a sum does
 * not care what order it adds in -- so the reduction, which is the part anyone
 * would check first, is correct. Only the per-column affine is wrong, and only
 * because w[c] and b[c] are per-column: with constant w and b=0 this kernel is
 * CORRECT and would ship.
 *
 * That is why the harness varies both w and b along the column axis.
 */
#include "kernel_api.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>

#define LANES_FP16 64
#define FP16_ONE   0x3C00

static inline HVX_Vector splat_f32(float f) {
    union { float f; int i; } bits;
    bits.f = f;
    return Q6_V_vsplat_R(bits.i);
}

void layernorm_fp16(const hexlib_hf *x, const float *w, const float *b,
                    hexlib_hf *y, int R, int C, float eps) {
    if (R <= 0 || C <= 0) return;

    const HVX_Vector one = Q6_Vh_vsplat_R(FP16_ONE);
    const HVX_Vector zero = Q6_V_vzero();
    const int nvec16 = C / LANES_FP16;

    for (int r = 0; r < R; ++r) {
        const hexlib_hf *xr = x + (long) r * C;
        hexlib_hf *yr = y + (long) r * C;

        float sum = 0.0f;
        for (int c = 0; c < C; ++c) sum += (float) xr[c];
        const float mean = sum / (float) C;
        float sq = 0.0f;
        for (int c = 0; c < C; ++c) {
            const float d = (float) xr[c] - mean;
            sq += d * d;
        }
        const float inv = 1.0f / sqrtf(sq / (float) C + eps);

        const HVX_Vector vmean = splat_f32(mean);
        const HVX_Vector vinv = splat_f32(inv);
        const HVX_Vector *xv = (const HVX_Vector *) xr;
        HVX_Vector *yv = (HVX_Vector *) yr;
        const HVX_Vector *wv = (const HVX_Vector *) w;
        const HVX_Vector *bv = (const HVX_Vector *) b;

        for (int i = 0; i < nvec16; ++i) {
            /* WRONG: no Q6_Vh_vshuff_Vh, so lanes are permuted and w/b land on
             * the wrong columns. */
            HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(xv[i], one);
            HVX_Vector part[2];
            part[0] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
            part[1] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));

            HVX_Vector out[2];
            for (int h = 0; h < 2; ++h) {
                const int j = 2 * i + h;
                HVX_Vector d = Q6_Vqf32_vsub_VsfVsf(part[h], vmean);
                d = Q6_Vqf32_vmpy_Vqf32Vqf32(d, Q6_Vqf32_vadd_VsfVsf(vinv, zero));
                d = Q6_Vqf32_vmpy_Vqf32Vqf32(d, Q6_Vqf32_vadd_VsfVsf(wv[j], zero));
                d = Q6_Vqf32_vadd_Vqf32Vsf(d, bv[j]);
                out[h] = Q6_Vsf_equals_Vqf32(d);
            }
            HVX_Vector qlo = Q6_Vqf32_vadd_VsfVsf(out[0], zero);
            HVX_Vector qhi = Q6_Vqf32_vadd_VsfVsf(out[1], zero);
            /* Deal present, so this isolates the missing shuffle. */
            yv[i] = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(qhi, qlo)));
        }
        for (int c = nvec16 * LANES_FP16; c < C; ++c) {
            yr[c] = (hexlib_hf) (((float) xr[c] - mean) * inv * w[c] + b[c]);
        }
    }
}

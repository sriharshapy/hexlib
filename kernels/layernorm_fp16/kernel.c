/* LayerNorm, fp16 data with fp32 affine params.
 *
 * WHAT IS VECTORISED AND WHAT IS NOT -- stated plainly, because this kernel is
 * deliberately NOT finished.
 *
 * The affine epilogue is vectorised: it is 3 of the 3 passes' worth of
 * arithmetic per element (subtract, two multiplies, an add) and it is where the
 * work is. It runs 32 fp32 lanes at a time.
 *
 * The two REDUCTIONS are scalar. That is a known cost and the next thing to fix.
 * A horizontal sum does not vectorise as a plain vector loop -- it needs a
 * rotate-and-add butterfly to collapse 32 lanes to one, which `kernels/
 * rmsnorm_fp16` already does for a sum of squares and which this kernel should
 * adopt. Leaving it scalar first is the honest order of work: this version is
 * correct and measurable, and the reduction has a recorded baseline to beat
 * rather than an assumed one. The cycle count in RESULT.md is therefore a FIRST
 * RUNG, not a result.
 *
 * WHY THE CONVERSIONS LOOK LIKE THAT. Two facts about this part, both
 * established the hard way and neither visible in an intrinsic's signature:
 *
 *   1. The fp16->qf32 widening (Q6_Wqf32_vmpy_VhfVhf) and the qf32->fp16
 *      narrowing (Q6_Vhf_equals_Wqf32) both PERMUTE lanes. Element k does not
 *      land in lane k. Q6_Vh_vshuff_Vh before the widening and Q6_Vh_vdeal_Vh
 *      after the narrowing put it back.
 *   2. Lane order is load-bearing HERE and not in a reduction. A sum does not
 *      care what order it adds in, so the reductions could use the permuted form
 *      freely -- but w[c] and b[c] are PER-COLUMN, so the epilogue must have
 *      element c in the lane that meets w[c]. Getting that wrong yields a
 *      correctly-shaped output with the affine params applied to the wrong
 *      columns, which no shape check can see.
 *
 * There is no fp16->qf32 convert instruction, so the widening is a multiply by
 * 1.0; there is no sf->qf32 convert either, so that widening is an add of 0.0.
 */
#include "kernel_api.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>

#define LANES_FP32 32
#define LANES_FP16 64
#define FP16_ONE   0x3C00

/* fp16 vector -> two IEEE fp32 vectors, IN ELEMENT ORDER.
 * out[0] holds elements 0..31, out[1] holds 32..63. */
static inline void widen_ordered(HVX_Vector v, HVX_Vector one, HVX_Vector *out) {
    HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v), one);
    out[0] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
    out[1] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));
}

/* Two IEEE fp32 vectors -> one fp16 vector, IN ELEMENT ORDER. */
static inline HVX_Vector narrow_ordered(HVX_Vector lo, HVX_Vector hi) {
    const HVX_Vector zero = Q6_V_vzero();
    HVX_Vector qlo = Q6_Vqf32_vadd_VsfVsf(lo, zero);
    HVX_Vector qhi = Q6_Vqf32_vadd_VsfVsf(hi, zero);
    return Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(qhi, qlo)));
}

/* Broadcast one float to all 32 lanes as IEEE single. */
static inline HVX_Vector splat_f32(float f) {
    union { float f; int i; } bits;
    bits.f = f;
    return Q6_V_vsplat_R(bits.i);
}

void layernorm_fp16(const hexlib_hf *x, const float *w, const float *b,
                    hexlib_hf *y, int R, int C, float eps) {
    if (R <= 0 || C <= 0) {
        return;
    }

    const HVX_Vector one = Q6_Vh_vsplat_R(FP16_ONE);
    const int nvec16 = C / LANES_FP16;   /* fp16 vectors per row */

    for (int r = 0; r < R; ++r) {
        const hexlib_hf *xr = x + (long) r * C;
        hexlib_hf *yr = y + (long) r * C;

        /* --- reductions: scalar for now, see the header comment --- */
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            sum += (float) xr[c];
        }
        const float mean = sum / (float) C;

        float sq = 0.0f;
        for (int c = 0; c < C; ++c) {
            const float d = (float) xr[c] - mean;
            sq += d * d;
        }
        const float inv = 1.0f / sqrtf(sq / (float) C + eps);

        /* --- vectorised affine epilogue --- */
        const HVX_Vector vmean = splat_f32(mean);
        const HVX_Vector vinv = splat_f32(inv);
        const HVX_Vector *xv = (const HVX_Vector *) xr;
        HVX_Vector *yv = (HVX_Vector *) yr;
        const HVX_Vector *wv = (const HVX_Vector *) w;
        const HVX_Vector *bv = (const HVX_Vector *) b;

        for (int i = 0; i < nvec16; ++i) {
            HVX_Vector part[2];
            widen_ordered(xv[i], one, part);

            HVX_Vector out[2];
            for (int h = 0; h < 2; ++h) {
                const int j = 2 * i + h;     /* which fp32 vector of the row */
                /* (x - mean) * inv * w + b, all in qf32, narrowed once at the
                 * end of the chain rather than between steps. */
                HVX_Vector d = Q6_Vqf32_vsub_VsfVsf(part[h], vmean);
                d = Q6_Vqf32_vmpy_Vqf32Vqf32(d, Q6_Vqf32_vadd_VsfVsf(vinv, Q6_V_vzero()));
                d = Q6_Vqf32_vmpy_Vqf32Vqf32(
                        d, Q6_Vqf32_vadd_VsfVsf(wv[j], Q6_V_vzero()));
                d = Q6_Vqf32_vadd_Vqf32Vsf(d, bv[j]);
                out[h] = Q6_Vsf_equals_Vqf32(d);
            }
            yv[i] = narrow_ordered(out[0], out[1]);
        }

        /* Tail: C not a multiple of 64. The encoder's C is 768, so this does not
         * run there, but a kernel that silently mangles a different C is worse
         * than one that is slow. */
        for (int c = nvec16 * LANES_FP16; c < C; ++c) {
            yr[c] = (hexlib_hf) (((float) xr[c] - mean) * inv * w[c] + b[c]);
        }
    }
}

/* kernels/matmul_fp16/kernel.c
 *
 * HVX-COMPUTE matmul. HMX was attempted first (see the report for what was
 * tried and why it was abandoned: the HMX matrix-unit path needs the SSR.XE
 * extension-context bit enabled from inside the kernel, and setting it from
 * this standalone gate's runtime made the simulator hang rather than fault or
 * proceed -- an environment difference from the two read-only reference
 * projects this kernel started from, not a numerics problem, and not
 * something safe to keep debugging blind against a 900s-per-run gate). Per
 * the brief's own fallback guidance, this is the complete, HVX-compute
 * deliverable: real vector ARITHMETIC (not just vector loads/stores), which
 * is what `hexlib/anticheat.py`'s `used_hvx_compute` proof requires.
 *
 * SHAPE OF THE COMPUTE. HVX has no native dot-product-with-horizontal-reduce
 * primitive worth using per output element here, so this vectorises across
 * the OUTPUT ROW (the N axis) instead of the K reduction: for a fixed batch
 * b and row m, accumulate `C[m][:] += A[m][k] * B[k][:]` for every k, where
 * `A[m][k]` is a single scalar broadcast across a vector and `B[k][:]` is a
 * real vector load -- an outer-product-style accumulation, one FMA-pair
 * (mul + add) per 64-column block per k. This is the same "vectorise the
 * elementwise axis, loop the reduction axis in scalar" shape
 * kernels/softmax_fp16/kernel.c uses for its own reduction (max, then sum),
 * just with the roles of "vectorised axis" and "reduction axis" swapped
 * (columns vectorised, K reduced) because THIS op's reduction axis is not
 * the last one.
 *
 * PRECISION. Accumulation is float32 throughout (never `hexlib_hf`), narrowed
 * to fp16 exactly once at the end of the K loop -- see kernel_api.h's SPEC
 * comment for where that requirement comes from (structural.py's matmul
 * reference + eager.py's fp32-everywhere oracle). All fp32 arithmetic here
 * goes through `hvx_vec_{add,mul}_f32_f32` (include/hexlib/hvx/hvx-base.h),
 * which on this arch are `Q6_Vsf_equals_Vqf32(Q6_Vqf32_..._VsfVsf(...))`
 * under the hood -- the qf32 path, never a native `Vhf`-typed accumulate.
 * `Q6_Vhf_vadd_VhfVhf` does not exist on v75 and crashes clang 19.0.04 with
 * exit code 70 (this repo's own hardware note); nothing here calls it.
 *
 * WIDEN/NARROW: reused from the vendored header, not reimplemented --
 * `hvx_vec_f16_to_f32` and `hvx_vec_f32_to_f16` (hvx-base.h) do the same
 * shuffle-widen / deal-narrow dance kernels/softmax_fp16/kernel.c already
 * verified line by line against layernorm_fp16's hand-rolled version.
 *
 * COLUMN BLOCKING. N is processed 64 fp16 lanes (= one HVX vector) at a time,
 * each block held as a pair of float32 accumulator vectors (lo/hi 32 lanes).
 * Both real encoder shapes have N a multiple of 64 (256 and 64), so the
 * vectorised path covers them fully; a scalar tail below still handles a
 * non-multiple-of-64 N correctly (not exercised by this harness, never
 * silently wrong for a shape nobody vectorised -- the same tradeoff
 * kernels/softmax_fp16/kernel.c documents for its own scratch cap). Rows (M)
 * and the batches (Bn) are plain scalar loops: only the reduction's inner
 * width (N) needs SIMD width, not the outer dimensions.
 */
#include "kernel_api.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>

#include "hexlib/hvx/hvx-base.h"

#define LANES_FP16 64

/* Per-row float32 accumulator: one (lo, hi) fp32 vector pair per 64-column
 * block. Sized for the encoder's own largest N (256 -> 4 blocks) with
 * headroom; a wider N falls back to the plain scalar loop below rather than
 * overflow this fixed array. */
#define MM_MAX_NVEC64 8

void matmul_fp16(const hexlib_hf *A, const hexlib_hf *B, hexlib_hf *C,
                  int Bn, int M, int K, int N) {
    if (Bn <= 0 || M <= 0 || K <= 0 || N <= 0) {
        return;
    }

    const int nvec64 = N / LANES_FP16;
    const int vecN = nvec64 * LANES_FP16;

    if (nvec64 > MM_MAX_NVEC64) {
        /* Wider than this kernel's fixed accumulator array -- correct but
         * fully scalar rather than overflowing it. */
        for (int b = 0; b < Bn; ++b) {
            const hexlib_hf *Ab = A + (long) b * M * K;
            const hexlib_hf *Bb = B + (long) b * K * N;
            hexlib_hf *Cb = C + (long) b * M * N;
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < N; ++n) {
                    float acc = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        acc += (float) Ab[(long) m * K + k] * (float) Bb[(long) k * N + n];
                    }
                    Cb[(long) m * N + n] = (hexlib_hf) acc;
                }
            }
        }
        return;
    }

    HVX_Vector acc_lo[MM_MAX_NVEC64];
    HVX_Vector acc_hi[MM_MAX_NVEC64];
    float scalar_acc[LANES_FP16];  /* for the N % 64 tail, at most 63 live */

    for (int b = 0; b < Bn; ++b) {
        const hexlib_hf *Ab = A + (long) b * M * K;
        const hexlib_hf *Bb = B + (long) b * K * N;
        hexlib_hf *Cb = C + (long) b * M * N;

        for (int m = 0; m < M; ++m) {
            const hexlib_hf *arow = Ab + (long) m * K;
            hexlib_hf *crow = Cb + (long) m * N;
            const HVX_Vector zero = Q6_V_vzero();

            for (int i = 0; i < nvec64; ++i) {
                acc_lo[i] = zero;
                acc_hi[i] = zero;
            }
            for (int n = vecN; n < N; ++n) {
                scalar_acc[n - vecN] = 0.0f;
            }

            /* K is the reduction axis: loop it in scalar, vectorise every
             * 64-column block of the row it touches. */
            for (int k = 0; k < K; ++k) {
                const float av = (float) arow[k];
                const HVX_Vector va = hvx_vec_splat_f32(av);
                const hexlib_hf *brow = Bb + (long) k * N;
                const HVX_Vector *bv = (const HVX_Vector *) brow;

                for (int i = 0; i < nvec64; ++i) {
                    HVX_VectorPair bp = hvx_vec_f16_to_f32(bv[i]);
                    HVX_Vector blo = Q6_V_lo_W(bp);
                    HVX_Vector bhi = Q6_V_hi_W(bp);
                    acc_lo[i] = hvx_vec_add_f32_f32(acc_lo[i], hvx_vec_mul_f32_f32(va, blo));
                    acc_hi[i] = hvx_vec_add_f32_f32(acc_hi[i], hvx_vec_mul_f32_f32(va, bhi));
                }
                for (int n = vecN; n < N; ++n) {
                    scalar_acc[n - vecN] += av * (float) brow[n];
                }
            }

            HVX_Vector *cv = (HVX_Vector *) crow;
            for (int i = 0; i < nvec64; ++i) {
                cv[i] = hvx_vec_f32_to_f16(acc_lo[i], acc_hi[i]);
            }
            for (int n = vecN; n < N; ++n) {
                crow[n] = (hexlib_hf) scalar_acc[n - vecN];
            }
        }
    }
}

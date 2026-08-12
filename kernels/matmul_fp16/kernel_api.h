/* kernels/matmul_fp16/kernel_api.h */
#ifndef HEXLIB_MATMUL_FP16_API_H
#define HEXLIB_MATMUL_FP16_API_H

typedef __fp16 hexlib_hf;

/* Batched matmul, fp16 in and out, NO bias and NO activation (that fused op is
 * `matmul_epilogue` -- hexlib/graph/opdefs/fused.py -- a different op and a
 * different kernel, kernels/matmul_epilogue_fp16/, out of scope here).
 *
 * SPEC, taken from:
 *   - hexlib/graph/opdefs/structural.py:55-61 -- the `matmul` OpDef itself:
 *       reference=lambda arrays, attrs: (arrays[0] @ arrays[1]).astype(arrays[0].dtype)
 *   - hexlib/graph/eager.py:20-29 (`NUMPY_DTYPE`) -- the ACCUMULATION PRECISION
 *     this reference actually runs at. Its own comment: "Every dtype is fed and
 *     computed as fp32 in the oracle" -- `NUMPY_DTYPE["fp16"] = float32`. So by
 *     the time `arrays[0] @ arrays[1]` runs, both operands are already float32
 *     arrays, `@` accumulates in (at least) float32, and `.astype(arrays[0].dtype)`
 *     is a no-op (arrays[0] is float32 already, not fp16). fp16 rounding happens
 *     exactly once, later, when `env.run` (eager.py:125) casts the op's output
 *     into the graph environment's fp16-tagged tensor -- never inside the matmul
 *     itself and never per partial sum.
 *
 * i.e. for each of the Bn independent batches (A is [M, K] row-major, B is
 * [K, N] row-major -- B already arrives pre-transposed to [k, n], per
 * structural.py's own module docstring, lines 3-5; C is [M, N] row-major):
 *
 *   out[m][n] = sum_{k=0}^{K-1} (float) A[m][k] * (float) B[k][n]   <- fp32 sum
 *   C[m][n]   = (fp16) out[m][n]                                    <- rounded ONCE
 *
 * THE QUIET WRONG ANSWER THIS RULES OUT: accumulating that K-reduction in fp16
 * instead of float32. At small K, on friendly data, that difference is smaller
 * than fp16's own legitimate 1-ULP rounding noise and is invisible to a loose
 * tolerance -- exactly how kernels/layernorm_fp16's unbiased-variance near-miss
 * was wrongly accepted once already (see ROADMAP.md / that kernel's own
 * comments). At this kernel's K (up to 256 in the real encoder; 128 in this
 * harness) it is not invisible -- see harness.c and nearmiss_fp16_accumulate.c
 * for the measured numbers that prove the harness discriminates it from
 * ordinary rounding noise, not just asserts that it does.
 *
 * SHAPES. The compiled plan needs exactly two, both batched matmuls with no
 * bias and no activation:
 *   fp16 (12, 256, 64)  @ fp16 (12, 64, 256)  -> fp16 (12, 256, 256)   [QK^T]
 *   fp16 (12, 256, 256) @ fp16 (12, 256, 64)  -> fp16 (12, 256, 64)    [AV]
 * i.e. B=12, (M,K,N) = (256,64,256) or (256,256,64). This kernel takes
 * (Bn, M, K, N) as parameters and covers both from one implementation, because
 * they are one op with two sizes, not two ops.
 *
 * MECHANISM. See kernel.c: this is an HVX-COMPUTE kernel (real vector
 * arithmetic -- multiply and add -- never just vector loads/stores). An HMX
 * (matrix-unit) version was attempted first; see the report for why it was
 * abandoned in favor of this one. HVX vectorises across the OUTPUT ROW (the
 * N axis) with the K reduction as a scalar outer loop: `C[m][:] += A[m][k] *
 * B[k][:]` accumulated in float32 vectors, one 64-column block at a time.
 * Both real encoder N values (256, 64) are multiples of 64 and take the fully
 * vectorised path; a scalar tail below still handles any N that is not, for
 * a future caller this kernel was not tuned for.
 *
 * ROUNDING. HVX's fp16<->fp32 widen/narrow (`hvx_vec_f16_to_f32` /
 * `hvx_vec_f32_to_f16`, include/hexlib/hvx/hvx-base.h) goes through the qf32
 * path and its narrow interleaves lanes via `Q6_Vh_vdeal_Vh` -- not
 * necessarily IEEE round-to-nearest-even, the same caveat this repo already
 * documents for `Q6_Vhf_equals_Wqf32`. So this kernel's result can differ
 * from the float32-accumulate-then-round-once reference by up to roughly one
 * fp16 ULP. That is why it is tolerance-compared (hexlib_close_f16), never
 * bit-exact.
 *
 * ALIGNMENT. A, B, and C must be 128-byte aligned; N should be a multiple of
 * 64 to take the vectorised path for the whole row (both real encoder shapes
 * satisfy this).
 */
#define MM_B 3
#define MM_M 40
#define MM_K 128
#define MM_N 192

void matmul_fp16(const hexlib_hf *A, const hexlib_hf *B, hexlib_hf *C,
                  int Bn, int M, int K, int N);

#endif

/* kernels/softmax_fp16/kernel_api.h */
#ifndef HEXLIB_SOFTMAX_FP16_API_H
#define HEXLIB_SOFTMAX_FP16_API_H

typedef __fp16 hexlib_hf;

/* Softmax over the last axis, row-wise, fp16 in and out.
 *
 * SPEC. Taken from hexlib/graph/opdefs/elementwise.py:141-146
 * (`_softmax_reference`), which is the op registry's own reference and the
 * eager/numpy implementation this kernel is held to:
 *
 *   axis = attrs["axis"]                       (== -1 for this kernel: last axis)
 *   shifted = x - max(x, axis, keepdims=True)   <- MAX SUBTRACTED BEFORE exp
 *   e = exp(shifted)
 *   out = e / sum(e, axis, keepdims=True)       <- SUMMED AFTER exp, divided last
 *
 * i.e. for each row r in [0, R):
 *   m        = max_c x[r][c]                          <- PER-ROW max, not global
 *   e[c]     = exp((float) x[r][c] - m)
 *   s        = sum_c e[c]
 *   y[r][c]  = (hexlib_hf) (e[c] / s)
 *
 * PRECISION. The reference upcasts to float64 (numpy's default promotion) --
 * see elementwise.py line 143 `x = arrays[0].astype(np.float64)`. This kernel,
 * like layernorm_fp16 and rmsnorm_fp16, computes in float32 instead: x and y are
 * fp16 storage, but the max, the exponential, and the sum are all float32, and
 * only the final quotient is rounded to fp16 once. float64 vs float32 is
 * invisible at fp16 output resolution for any well-conditioned row; it is NOT
 * invisible for a row engineered to make float32-vs-float16 summation differ,
 * which is what this kernel's near-miss and harness exist to demonstrate (see
 * harness.c and nearmiss_sum_fp16.c).
 *
 * EXP: NOT hvx_vec_exp2_f16. include/hexlib/hvx/hvx-exp.h's `hvx_vec_exp2_f16`
 * has a wrong E5 polynomial coefficient (0x5082 where upstream calls for
 * 0x090c) -- 262% error at fractional input 0.7, live in llama.cpp's own fp16
 * flash-attention softmax. This kernel uses `hvx_vec_exp_f32` instead (same
 * header), a natural-log-based degree-7 Taylor polynomial in fp32 with its own,
 * different and unaffected, coefficient table. Measured against real exp() in
 * Python (see kernel.c's header comment for the numbers): ~1e-6 relative over
 * the input range that matters after max-subtraction; the function's own
 * intentional clamp below -88 only affects terms that underflow to 0 in fp16
 * anyway, so it costs nothing here.
 *
 * SHAPE. The encoder's actual op is fp16 (12, 256, 256), axis=-1: 12 batches of
 * 256 rows of 256 elements, 3072 independent rows total, 12 ops. Rows are
 * independent, so like layernorm_fp16 (LN_R=4 vs. the encoder's R=256) this
 * kernel's own harness uses a SMALLER R than the real 3072 -- but the SAME
 * C=256, because C (the reduction width) is what determines the numerics and
 * the vector loop structure, not R. THE HARNESS'S R AND C ARE DELIBERATELY
 * UNEQUAL (SM_R != SM_C below), even though the real encoder's last two dims
 * ARE square (256x256): a wrong-axis near-miss on a square matrix produces a
 * same-shape, same-total-size output, so a shape check cannot distinguish it,
 * and even a value check could be fooled by an accidentally-symmetric test
 * matrix. R != C makes a row-softmax and a column-softmax structurally
 * different (different reduction group sizes) regardless of what the data
 * looks like. See nearmiss_wrong_axis.c and harness.c.
 *
 * ALIGNMENT. x and y must be 128-byte aligned. C must be a multiple of 64 (the
 * fp16 HVX vector width) for the vectorised path; the kernel still produces
 * correct results for a non-multiple C via a scalar tail, and falls back to a
 * fully scalar per-row computation if C exceeds SOFTMAX_SCRATCH_CAP (see
 * kernel.c) -- slow, but never silently wrong for a shape nobody vectorised.
 * Rows are independent.
 */
#define SOFTMAX_R 6
#define SOFTMAX_C 256

void softmax_fp16(const hexlib_hf *x, hexlib_hf *y, int R, int C);

#endif

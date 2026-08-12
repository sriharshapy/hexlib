/* kernels/matmul_epilogue_fp16/kernel_api.h */
#ifndef HEXLIB_MATMUL_EPILOGUE_FP16_API_H
#define HEXLIB_MATMUL_EPILOGUE_FP16_API_H

typedef __fp16 hexlib_hf;

/* matmul + bias + optional activation, with a q4_0 block-quantized weight.
 * 75 of the encoder's 259 real-work plan steps are this op (48x (256,768)x
 * (768,768), 12x (256,768)x(768,3072) act=gelu_tanh, 12x (256,3072)x(3072,768),
 * 1x (256,1536)x(1536,768), 1x (64,3072)x(3072,3072) act=gelu_erf, 1x
 * (64,3072)x(3072,1024)) -- more than a quarter of the whole encoder's work,
 * and 55.9 of its 58.6 MB of weights.
 *
 * ==========================================================================
 * SPEC SOURCE 1: the fused op's own reference, ORDER OF OPERATIONS.
 * ==========================================================================
 * hexlib/graph/opdefs/fused.py:44-53 (`_reference`):
 *
 *   out = (a @ b + bias).astype(a.dtype)      <- BIAS ADDED, THEN CAST
 *   if act == "none": return out
 *   return get(act).reference((out,), {})     <- ACTIVATION APPLIED AFTER
 *
 * i.e. for every element: y[m,n] = act(sum_k a[m,k]*w[k,n] + bias[n]).
 * BIAS IS ADDED BEFORE THE ACTIVATION, NEVER AFTER -- this is the one order
 * `fuse.py` (hexlib/graph/fuse.py:13,61-76) is even ALLOWED to produce: it only
 * folds a matmul -> add(bias) -> {gelu_tanh,gelu_erf} chain, bias-then-act by
 * construction. See nearmiss_bias_after_activation.c for what applying it in
 * the other order does to the output.
 *
 * ==========================================================================
 * SPEC SOURCE 2: the two activation formulas (DIFFERENT FUNCTIONS).
 * ==========================================================================
 * hexlib/graph/opdefs/elementwise.py:110-125. gelu_tanh and gelu_erf are two
 * different functions used in two different places in the model (line 4 of
 * that file's own header), not one op with a flag:
 *
 *   gelu_tanh(x) = 0.5*x*(1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
 *                  -- ACT2FN["gelu_pytorch_tanh"], the blocks' MLP
 *                     (modeling_qwen3_5.py:849, cited at elementwise.py:111)
 *   gelu_erf(x)  = 0.5*x*(1 + erf(x / sqrt(2)))
 *                  -- plain nn.GELU(), approximate='none', the merger
 *                     (modeling_qwen3_5.py:882, cited at elementwise.py:118-122)
 *
 * These two agree to within ~5e-4 absolute almost everywhere (see harness.c's
 * header for the measured max, ~4.7e-4 at x ~= -2.7) -- a genuinely quiet
 * near-miss if the two formulas are ever swapped for each other. See
 * nearmiss_gelu_swap.c and harness.c for how this kernel's tolerance is shown
 * to still catch it.
 *
 * ==========================================================================
 * SPEC SOURCE 3: the q4_0 block format.
 * ==========================================================================
 * hexlib/graph/ir.py:20-23 (`Q4_0_BLOCK = 32`, `Q4_0_BLOCK_BYTES = 18`):
 * "32 four-bit values (16 bytes) + one fp16 scale (2 bytes) = 18 bytes...
 * matches llama.cpp's block_q4_0." Dequant formula (adapted, not copied, from
 * llama.cpp's `dequantize_row_q4_0`, `ggml-quants.c`, per
 * docs/research/quantization.md section 1, "Q4_0 -- plain affine RTN"):
 *
 *   for a block of 32 values with scale d and 16 code bytes qs[0..15]:
 *     for j in [0, 16):
 *       value[j]    = d * ((qs[j] & 0x0F) - 8)   <- LOW nibble  -> index j
 *       value[j+16] = d * ((qs[j] >> 4)  - 8)   <- HIGH nibble -> index j+16
 *
 * NOT interleaved (2*j, 2*j+1). See nearmiss_swapped_nibble_order.c for what
 * reading it the other way does.
 *
 * ==========================================================================
 * THE LAYOUT DECISION THIS KERNEL COMMITS TO: PLAIN ROW-MAJOR q4_0, NOT
 * ggml-hexagon's 576-BYTE REPACKED TILE ORDER.
 * ==========================================================================
 * hexlib_dsp.h defines HEXLIB_LAYOUT_ROW_MAJOR (0) and HEXLIB_LAYOUT_
 * Q4_0_REPACKED (2) as distinct enum values precisely so "un-repacked weights
 * are a plan-time error rather than silent corruption" -- this kernel expects
 * HEXLIB_LAYOUT_ROW_MAJOR for its weight operand.
 *
 * The weight tensor's LOGICAL shape is (K, N) -- reduction dim first, output
 * dim second, numpy/C row-major -- exactly as `hexlib/exec/runner.py`'s own
 * q4_0 staging tests declare it (ENCODER_WEIGHT_SHAPES in
 * test_raw_q4_0_staging.py: (768,768), (768,3072), (3072,768), (1536,768),
 * (3072,3072), (3072,1024), all (K,N)). `ir.nbytes` requires shape[-1] % 32 ==
 * 0 (ir.py:47), i.e. the LAST axis -- N, the output/free dimension -- is the
 * one split into 32-element blocks, not K. Concretely: row k of the weight
 * occupies `(N/32)*18` contiguous bytes; block b of row k (bytes
 * `[b*18, b*18+18)` within that row) covers output columns `[b*32, b*32+32)`,
 * ALL AT THE SAME REDUCTION INDEX k, sharing one scale.
 *
 * WHY THIS AXIS, NOT llama.cpp's: llama.cpp's own Linear weight is stored
 * (out_features, in_features) and blocks along in_features (the reduction
 * axis) because its consumer is HMX/GEMM hardware that wants a scale
 * per reduction-strip. This kernel does the opposite deliberately: blocking
 * along N means that for a FIXED k, one 18-byte block dequantizes directly
 * into a 32-lane HVX vector representing 32 *output columns*, which is
 * multiplied by the single broadcast scalar a[m,k] and accumulated into a
 * 32-lane fp32 partial-output vector -- exactly the per-k multiply-accumulate
 * structure this (HVX-compute, not HMX) kernel is built around. See
 * "WHY HVX-COMPUTE, NOT HMX" below.
 *
 * PLAIN ROW-MAJOR, NOT REPACKED: this kernel reads the 18-byte blocks in their
 * natural (k, block-of-32-columns) order, exactly as `ir.nbytes` lays them
 * out. It does NOT expect llama.cpp/ggml-hexagon's 32x32-tile, 576-byte
 * repacked order (`docs/research/quantization.md` section 0, `repack_q4_0_
 * tiled`) -- that repacking exists to feed HMX's fixed 32x32 tile shape, and
 * this kernel does not drive HMX (see below). A caller MUST set the wire
 * layout id to HEXLIB_LAYOUT_ROW_MAJOR (0) for this weight, not
 * HEXLIB_LAYOUT_Q4_0_REPACKED (2) -- the latter would be a correctly-shaped
 * wrong answer, exactly the failure mode hexlib_dsp.h's enum exists to make
 * loud instead of silent.
 *
 * ==========================================================================
 * WHY HVX-COMPUTE, NOT HMX.
 * ==========================================================================
 * HMX needs weights pre-dequantized to fp16 tiles (q4_0 nibbles are not a
 * type HMX accepts directly -- docs/research/quantization.md section 0/2),
 * needs the activation load and weight load braced into ONE packet (silently
 * zeroing the accumulator otherwise), and needs an explicit outer reduction
 * loop for K > 32*32 = 1024 (K=3072 here needs 3 strips of <=32 dot-tiles,
 * accumulated) -- three separate, individually error-prone pieces of new
 * machinery. Per this task's own scope note: "an HVX-compute kernel that
 * dequantizes q4_0 blocks to fp16 and does a vectorised multiply-accumulate
 * is a COMPLETE AND VALUABLE result... HMX is the optimisation." This kernel
 * takes that path: HVX vector compute (widen-multiply-accumulate in qf32,
 * narrow once), no HMX. It makes all 75 plan steps dispatchable; a follow-up
 * kernel can add the HMX path later without changing this one's contract.
 *
 * ==========================================================================
 * ACCUMULATION PRECISION AND ROUNDING.
 * ==========================================================================
 * fp32 (Hexagon's qf32 pipeline) THE ENTIRE WAY THROUGH, WITH ONE UNAVOIDABLE
 * EXTRA ROUNDING: the only HVX primitive available to multiply the
 * dequantized weight by the activation scalar (`Q6_Wqf32_vmpy_VhfVhf`) takes
 * two fp16 VECTORS, so each dequantized weight value is stored as fp16 (one
 * rounding) BEFORE it is multiplied, not only at the final output. baseline.c
 * replicates this same intermediate fp16 rounding of the weight for exactly
 * this reason -- omitting it there was tried first and cost up to ~2.2e-3
 * absolute error at K=128 (a few ULP of weight-rounding noise per reduction
 * term, accumulated over many terms, is not negligible at the largest K this
 * harness tests), which would have been comparing this kernel against a
 * different, more-accurate-than-actual algorithm. Each dequantized fp16
 * weight value is then widened and multiplied by the fp16 activation scalar
 * into qf32, accumulated into a running fp32 sum over all of K, THEN the
 * fp32 bias is added, THEN the activation (also computed in fp32, via
 * `hvx_vec_exp_f32` -- never `hvx_vec_exp2_f16`, which has a wrong E5
 * coefficient, see kernel.c) is applied, and the fp16 STORE is the one and
 * only narrowing step. This matches kernels/softmax_fp16 and kernels/
 * layernorm_fp16's own established precedent ("float32 intermediate, narrow
 * once") and is provably more accurate than rounding to fp16 after the bias
 * add and again after the activation. The oracle itself
 * (hexlib/graph/eager.py:24-29) computes every dtype as float32/float64 and
 * is blind to this choice; harness.c's header comment gives the actual
 * measured numbers for what this choice costs against a scalar fp32
 * baseline, and why that is far below the size of a real bug.
 *
 * See kernel.c's header for the exact HVX lane-order mechanics (the
 * fp16<->qf32 widen/narrow permutes lanes; this kernel follows the same
 * shuffle-before/deal-after convention as kernels/layernorm_fp16 and
 * kernels/rope_2d_fp16, because it mixes computed (permuted-domain) values
 * with a plainly-loaded bias vector, exactly the situation those two kernels'
 * own comments warn about).
 *
 * ==========================================================================
 * SHAPES AND ALIGNMENT.
 * ==========================================================================
 * a: fp16 (M, K), row-major, 128-byte aligned.
 * w: q4_0 raw bytes, LOGICAL shape (K, N), row-major blocks per above. K and N
 *    are both required to be multiples of 32 (`ir.nbytes`'s own constraint;
 *    every encoder shape satisfies it for both dims).
 * bias: fp32 (N,), 128-byte aligned is not required (loaded unaligned).
 * out: fp16 (M, N), row-major, 128-byte aligned.
 * act: one of MM_ACT_NONE, MM_ACT_GELU_TANH, MM_ACT_GELU_ERF (below).
 *
 * THE CROUTON LAYOUT ("every two rows transposed") DOES NOT APPLY HERE. It is
 * called out in `ai_fp16_matmul_gelu/spec.json` (../HVX-clean/data/v6/tasks/)
 * as an edge case for THAT kernel's activation operand layout. This kernel's
 * activation operand `a` is required to be plain row-major fp16 (declared
 * above and enforced by the wire's `row_major` layout id on that input, per
 * hexlib/tests/test_raw_q4_0_staging.py's own `_spec()`); nothing here ever
 * reads `a` two rows at a time or reinterprets it as anything other than
 * (M, K) row-major. There is no crouton-shaped operand in this kernel's
 * contract to get wrong.
 */
#define MM_ACT_NONE      0
#define MM_ACT_GELU_TANH 1
#define MM_ACT_GELU_ERF  2

#define MM_Q4_0_BLOCK       32
#define MM_Q4_0_BLOCK_BYTES 18

void matmul_epilogue_fp16(const hexlib_hf *a, const unsigned char *w,
                          const float *bias, hexlib_hf *out,
                          int M, int K, int N, int act);

#endif

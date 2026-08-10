/* kernels/layernorm_fp16/kernel_api.h */
#ifndef HEXLIB_LAYERNORM_FP16_API_H
#define HEXLIB_LAYERNORM_FP16_API_H

typedef __fp16 hexlib_hf;

/* LayerNorm over the last axis, row-wise, fp16 data with fp32 affine params.
 *
 *   for each row r in [0, R):
 *     mean = (1/C) * sum_c x[r][c]
 *     var  = (1/C) * sum_c (x[r][c] - mean)^2          <- BIASED (divide by C)
 *     inv  = 1 / sqrt(var + eps)
 *     y[r][c] = (hexlib_hf) ((x[r][c] - mean) * inv * w[c] + b[c])
 *
 * 25 ops in the encoder at R=256, C=768, eps=1e-6.
 *
 * eps IS 1e-6 AND IT IS NOT IN config.json -- it is a literal in the model
 * source. The vision tower uses LayerNorm; the text tower uses RMSNorm with a
 * different epsilon. They are not interchangeable, and the encoder's LayerNorm
 * has a BIAS term that RMSNorm has no equivalent of.
 *
 * WHY THE VARIANCE IS BIASED: that is what LayerNorm specifies -- divide by C,
 * not C-1. A near-miss uses the unbiased form, which is the mistake anyone who
 * reaches for a statistics habit rather than the definition will make.
 *
 * w AND b ARE fp32, x AND y ARE fp16. The reduction, the reciprocal square root
 * and the affine are all computed in float; only the stored result is rounded to
 * fp16. That matches the reference implementation, which normalises in float
 * regardless of the activation dtype.
 *
 * ALIGNMENT: x, y, w, b must be 128-byte aligned. C must be a multiple of 64
 * (the fp16 vector width) -- 768 is. Rows are independent.
 */
#define LN_R   4
#define LN_C   768
#define LN_EPS 1e-6f

void layernorm_fp16(const hexlib_hf *x, const float *w, const float *b,
                    hexlib_hf *y, int R, int C, float eps);

#endif

/* kernels/rmsnorm_fp16/kernel_api.h */
#ifndef HEXLIB_RMSNORM_FP16_API_H
#define HEXLIB_RMSNORM_FP16_API_H

typedef __fp16 hexlib_hf;

/* RMSNorm with a per-column gain, row-wise.
 *
 *   for each row r in [0, R):
 *     ms   = (1/C) * sum_c x[r][c] * x[r][c]      accumulated in float
 *     inv  = 1 / sqrt(ms + eps)
 *     y[r][c] = (hexlib_hf) ((float) x[r][c] * inv * (float) w[c])
 *
 * x, w, y are __fp16. The reduction and the reciprocal square root are computed
 * in float; only the stored result is rounded to fp16. HVX float is the
 * non-IEEE qf16 path and float operations reorder, so results are compared with
 * hexlib_close_f16, never bit-exactly.
 *
 * Rows are independent. C is a multiple of 64 (one 64-lane fp16 HVX vector), so
 * there is no tail on the column axis.
 */
#define RMSNORM_R   8
#define RMSNORM_C   128
#define RMSNORM_EPS 1e-5f

void rmsnorm_fp16(const hexlib_hf *x, const hexlib_hf *w,
                  hexlib_hf *y, int R, int C, float eps);

#endif

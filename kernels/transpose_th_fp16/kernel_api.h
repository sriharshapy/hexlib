/* kernels/transpose_th_fp16/kernel_api.h */
#ifndef HEXLIB_TRANSPOSE_TH_FP16_API_H
#define HEXLIB_TRANSPOSE_TH_FP16_API_H

typedef __fp16 hexlib_hf;

/* [T, H, D] -> [H, T, D], fp16. The attention layout move, perm (1, 0, 2).
 *
 *   y[h][t][d] = x[t][h][d]
 *
 * 36 ops in the encoder, at T=256 H=12 D=64 -- the most numerous op that is not
 * a reshape.
 *
 * WHY THIS ONE IS EASY AND perm(0,2,1) IS NOT. D is the innermost axis of both
 * operands and it is untouched, so a whole run of D contiguous elements moves as
 * a unit. At D=64, fp16, that run is 64*2 = 128 bytes: EXACTLY one HVX vector,
 * and every run therefore starts at a 128-byte boundary if the base does. So the
 * whole op is a permutation of naturally-aligned whole vectors, with no
 * cross-lane shuffling anywhere. Transposing the innermost two axes has none of
 * that.
 *
 * WHY IT IS A KERNEL AT ALL, when reshape is free: reshape reinterprets the same
 * bytes in the same order, so it moves nothing. This reorders them. The two look
 * alike in an IR and are not alike at all -- there is a near-miss that treats
 * this as a copy.
 *
 * D*2 need not be a multiple of 128; a run that does not fill whole vectors is
 * copied scalar-wise. ALIGNMENT: x and y must be 128-byte aligned.
 */
#define TR_T 8
#define TR_H 3
#define TR_D 64

/* A second shape whose run is NOT a whole number of vectors, so the harness
 * exercises the scalar path too. 40*2 = 80 bytes. */
#define TR_D_ODD 40

void transpose_th_fp16(const hexlib_hf *x, hexlib_hf *y, int T, int H, int D);

#endif

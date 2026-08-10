/* kernels/scale_fp16/kernel_api.h */
#ifndef HEXLIB_SCALE_FP16_API_H
#define HEXLIB_SCALE_FP16_API_H

typedef __fp16 hexlib_hf;

/* Multiply a contiguous fp16 array by a scalar.
 *
 *   for i in [0, n):  y[i] = (hexlib_hf) ((float) x[i] * factor)
 *
 * WHERE THIS IS USED. Attention scales Q by 1/sqrt(head_dim) before the Q@K^T
 * contraction. At head_dim 64 the factor is 0.125, which is a power of two and
 * therefore EXACT in fp16: the multiply changes only the exponent field and no
 * mantissa bit is lost. So for the shape this kernel exists to serve the
 * operation is lossless, and the tolerance below is for the general case, not
 * for that one.
 *
 * The factor is passed as float, not __fp16: hexagon-clang rejects __fp16 as a
 * by-value parameter outright ("parameters cannot have __fp16 type"), and it
 * fires on the declaration, so a header with that signature cannot even be
 * included. The conversion to fp16 happens once inside the kernel.
 *
 * ALIGNMENT. x and y must be 128-byte aligned. Both callers guarantee it (the
 * harness with HEXLIB_ALIGN, the executor because a VTCM slot is aligned), and
 * requiring it buys the aligned load and store rather than the unaligned pair.
 * n need NOT be a multiple of the 64-lane fp16 vector; the tail is handled
 * scalar-wise.
 */
/* 4100 = 64*64 + 4, so there IS a tail. A multiple of 64 would let a kernel
 * that silently drops the tail pass, which is one of the two near-misses here
 * and a mistake that costs nothing to make. The encoder's own shape
 * (12*256*64 = 196608) has no tail, so a harness built only from the real
 * shape would never catch it -- the harness has to be harder than the caller. */
#define SCALE_N      4100
#define SCALE_FACTOR 0.125f

void scale_fp16(const hexlib_hf *x, hexlib_hf *y, int n, float factor);

#endif

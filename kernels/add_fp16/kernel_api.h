/* kernels/add_fp16/kernel_api.h */
#ifndef HEXLIB_ADD_FP16_API_H
#define HEXLIB_ADD_FP16_API_H

typedef __fp16 hexlib_hf;

/* Elementwise add of two contiguous fp16 arrays.
 *
 *   for i in [0, n):  y[i] = (hexlib_hf) ((float) a[i] + (float) b[i])
 *
 * WHERE THIS IS USED. Every residual connection in the encoder: 24 of the
 * graph's 25 adds are fp16 + fp16 at [256, 768]. The 25th adds the learned
 * position embedding, whose right operand is fp32; it is rounded to fp16 before
 * the call rather than given a second kernel, because a dedicated mixed-dtype
 * kernel would exist for exactly one op in the whole model.
 *
 * THE ARITHMETIC IS EXACT, whichever way it is implemented. The exact sum of two
 * fp16 values is always representable in fp32, so widening, adding and rounding
 * back rounds exactly once -- the same single rounding a native fp16 add would
 * do. That matters here because v75 has no fp16 add instruction (it arrives at
 * __HVX_ARCH__ 79), so the kernel is obliged to widen, and it is worth recording
 * that this costs speed and not accuracy.
 *
 * This holds for a SINGLE add and not for a sum of many terms, which is why the
 * reduction kernels accumulate in a wider format and this one does not need to.
 *
 * ALIGNMENT. a, b and y must be 128-byte aligned. n need not be a multiple of
 * the 64-lane fp16 vector; the tail is handled scalar-wise.
 */
/* 4100 = 64*64 + 4, so the tail path runs. The encoder's own shape
 * (256*768 = 196608) is a multiple of 64, so a harness sized from the caller
 * would never execute the tail -- and a missing tail is one of the near-misses. */
#define ADD_N 4100

void add_fp16(const hexlib_hf *a, const hexlib_hf *b, hexlib_hf *y, int n);

#endif

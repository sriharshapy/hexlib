/* kernels/cast_f32_f16/kernel_api.h */
#ifndef HEXLIB_CAST_F32_F16_API_H
#define HEXLIB_CAST_F32_F16_API_H

typedef __fp16 hexlib_hf;

/* Narrow fp32 to fp16.
 *
 *   for i in [0, n):  y[i] = (hexlib_hf) x[i]
 *
 * 1 op in the encoder, at [256, 1536]: the host/activation dtype boundary. The
 * host always hands in an fp32 image; activations inside the encoder are fp16.
 * The IR makes that boundary an explicit op so no other op can smuggle a dtype
 * change through its own output declaration.
 *
 * TWO fp32 VECTORS PRODUCE ONE fp16 VECTOR. A 128-byte vector holds 32 fp32
 * lanes or 64 fp16 lanes, so the loop consumes input two vectors at a time and
 * emits one. That is why `n` must be a multiple of 64 for the vector path, not
 * 32; a shorter remainder is handled scalar-wise.
 *
 * ROUNDING is whatever the hardware's qf32->fp16 narrowing does, which is NOT
 * guaranteed to be IEEE round-to-nearest-even -- see the tolerance below. The
 * scalar baseline uses the C cast, which IS round-to-nearest, so the two can
 * differ by one ULP and the comparison allows it.
 *
 * ALIGNMENT: x and y must be 128-byte aligned.
 */
/* 4160 = 64*65, a multiple of the 64-lane fp16 output vector, plus a separate
 * odd shape below for the scalar path. */
#define CAST_N     4160
#define CAST_N_ODD 4133

void cast_f32_f16(const float *x, hexlib_hf *y, int n);

#endif

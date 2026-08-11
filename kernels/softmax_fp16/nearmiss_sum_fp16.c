/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: accumulating the softmax denominator in fp16 (__fp16, rounding
 * to fp16 after EVERY addition) instead of float32. Everything else here is
 * IDENTICAL to baseline.c -- same max subtraction, same expf(), same division
 * at the end. Only the type of the running sum changes.
 *
 * WHY ANYONE WOULD WRITE IT. x and y are both fp16; e[c] = exp(shifted) is a
 * value the same "shape" as the data everywhere else in this kernel, so
 * accumulating it in the storage dtype looks consistent rather than careless --
 * especially next to a hardware target where an fp16 accumulator is one lane
 * width, and float32 is two. Nothing about the C source LOOKS unstable; the
 * loop is the same loop.
 *
 * WHY THIS IS THE INTERESTING NEAR-MISS. On "friendly" data (measured in
 * Python: 256 small values spread over roughly [-2, 2]) this bug differs from
 * a correct float32 accumulation by well under one ULP on most elements --
 * genuinely invisible to a max-error check, for the same reason the
 * unbiased-variance near-miss in kernels/layernorm_fp16/ was once wrongly
 * accepted: the bug is smaller than fp16's own legitimate rounding noise on
 * data that does not stress it. A tolerance loose enough to admit the real
 * kernel's noise on friendly data would ALSO admit this bug there.
 *
 * THAT IS WHY harness.c's row 1 IS NOT FRIENDLY. It is built specifically to
 * make 256 fp16-rounded additions lose real mass: one dominant term and 255
 * IDENTICAL followers each ~exp(-6.5) = 0.0015 -- individually close to the
 * running accumulator's own fp16 ULP, added one at a time, 255 times, so a
 * little rounds away on almost every step. Measured in Python running this
 * exact algorithm on that exact row: float32 sum = 1.383384, fp16-per-step sum
 * = 1.498047 -- an 8.3% difference in the denominator, landing as a 7.7%
 * relative / 0.0552 absolute error on the dominant output element, some 90-150x
 * the size of the real kernel's own single-fp16-rounding noise on the same
 * row (~4.9e-4). See harness.c's header comment for the full derivation and
 * why the chosen tolerance (1% relative, 1e-3 absolute) sits comfortably
 * between the two rather than having been loosened until something passed.
 */
#include "kernel_api.h"

#include <math.h>

void softmax_fp16(const hexlib_hf *x, hexlib_hf *y, int R, int C) {
    for (int r = 0; r < R; ++r) {
        const hexlib_hf *xr = x + (long) r * C;
        hexlib_hf *yr = y + (long) r * C;

        float m = (float) xr[0];
        for (int c = 1; c < C; ++c) {
            float v = (float) xr[c];
            if (v > m) m = v;
        }

        /* WRONG: the running sum is __fp16, so it rounds to fp16 after every
         * single addition instead of accumulating in float32. */
        hexlib_hf s16 = (hexlib_hf) 0.0f;
        for (int c = 0; c < C; ++c) {
            float e = expf((float) xr[c] - m);
            s16 = (hexlib_hf) ((float) s16 + e);
        }
        float s = (float) s16;

        for (int c = 0; c < C; ++c) {
            yr[c] = (hexlib_hf) (expf((float) xr[c] - m) / s);
        }
    }
}

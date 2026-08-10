/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: copying the array through unchanged, as though a transpose were a
 * reinterpretation of the same bytes.
 *
 * WHY ANYONE WOULD WRITE IT. This project establishes -- correctly -- that every
 * `reshape` in the encoder is FREE, because at these shapes a reshape reads the
 * same bytes in the same order and so moves nothing. `transpose` sits next to it
 * in the IR, takes the same kind of `perm`-looking attribute, and also does no
 * arithmetic. The step from "reshape is free" to "this layout op is free" is one
 * inference, and it is wrong: a transpose reorders the bytes.
 *
 * Both ops also produce an output with the same ELEMENT COUNT as their input, so
 * every shape check in the pipeline still passes. Nothing but the values catches
 * this.
 */
#include "kernel_api.h"

void transpose_th_fp16(const hexlib_hf *x, hexlib_hf *y, int T, int H, int D) {
    const long n = (long) T * H * D;
    /* WRONG: same bytes, same order. */
    for (long i = 0; i < n; ++i) {
        y[i] = x[i];
    }
}

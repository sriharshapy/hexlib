/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: exponentiating the raw score directly, without subtracting the
 * row max first. Mathematically softmax is shift-invariant -- subtracting ANY
 * per-row constant before dividing leaves the result unchanged -- so on paper
 * this "simplification" looks free. It is not free in float32: exp() overflows
 * long before the division would have cancelled it back out.
 *
 * WHY ANYONE WOULD WRITE IT. `e[c] = exp(x[c]); y[c] = e[c] / sum(e)` reads as
 * a more direct transcription of "softmax(x)_c = exp(x_c) / sum(exp(x))" than
 * the numerically-stable form with the max subtraction folded in -- the max
 * subtraction is a stability trick, not part of the mathematical definition,
 * and it is the kind of line a first draft omits.
 *
 * WHY IT IS LOUD, NOT SUBTLE (unlike nearmiss_sum_fp16.c). exp(90) alone
 * overflows float32 (ln(FLT_MAX) is ~88.72), so harness.c's row 0
 * (x[0][0] = 90.0f) turns into +inf, and inf / inf is NaN by IEEE 754 -- every
 * element of that row becomes NaN, and hexlib_close_f16 has no code path that
 * calls NaN close to anything. This near-miss is deliberately the "too easy"
 * one the task description warns about: it is included because it IS a real
 * mistake, not because it is a hard one to catch.
 */
#include "kernel_api.h"

#include <math.h>

void softmax_fp16(const hexlib_hf *x, hexlib_hf *y, int R, int C) {
    for (int r = 0; r < R; ++r) {
        const hexlib_hf *xr = x + (long) r * C;
        hexlib_hf *yr = y + (long) r * C;

        float s = 0.0f;
        for (int c = 0; c < C; ++c) {
            s += expf((float) xr[c]);   /* WRONG: no max subtraction */
        }
        for (int c = 0; c < C; ++c) {
            yr[c] = (hexlib_hf) (expf((float) xr[c]) / s);
        }
    }
}

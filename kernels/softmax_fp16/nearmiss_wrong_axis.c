/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: softmax reduced over axis 0 (down each column, across the R
 * rows) instead of axis -1 (across each row, over the C columns). Attention
 * softmax and "softmax the other way" are both a completely ordinary thing to
 * write; get the loop nesting backwards -- outer over columns, inner over
 * rows, normalising by the column instead of the row -- and every line still
 * type-checks and every array access is still in bounds.
 *
 * WHY THE SHAPE MATTERS HERE. The encoder's real op is fp16 (12, 256, 256),
 * axis=-1 -- the last two dims ARE square. On a square matrix, axis-0-softmax
 * and axis-(-1)-softmax produce the SAME shape and the SAME total element
 * count, so neither a shape assertion nor a naive "did the output resize
 * correctly" check can tell them apart; only the VALUES differ, and only if
 * the test data itself is not accidentally symmetric. kernel_api.h's own
 * SHAPE note is about exactly this: harness.c deliberately uses R=6 != C=256,
 * so a row-reduction and a column-reduction are summing over groups of very
 * different sizes (6 vs 256) no matter what the data looks like -- this
 * near-miss cannot pass by coincidence here the way it could on a square test
 * shape with unlucky (symmetric) data.
 *
 * WHY IT FAILS BY A LOT, NOT A LITTLE. Averaged over a column of only 6
 * elements, a typical output magnitude is around 1/6 ~ 0.167; averaged over a
 * row of 256, it is around 1/256 ~ 0.0039 -- a ~43x scale mismatch before
 * even accounting for the different values being combined. This is the
 * "too easy on its own" class the task description flags, included anyway
 * because it is a real, easy-to-write mistake, and because it is the one that
 * specifically requires the R != C harness shape to be caught reliably rather
 * than by luck.
 */
#include "kernel_api.h"

#include <math.h>

void softmax_fp16(const hexlib_hf *x, hexlib_hf *y, int R, int C) {
    /* WRONG AXIS: outer loop over columns, inner loop over rows -- this
     * normalises each COLUMN of R elements instead of each ROW of C
     * elements. */
    for (int c = 0; c < C; ++c) {
        float m = (float) x[0 * C + c];
        for (int r = 1; r < R; ++r) {
            float v = (float) x[r * C + c];
            if (v > m) m = v;
        }

        float s = 0.0f;
        for (int r = 0; r < R; ++r) {
            s += expf((float) x[r * C + c] - m);
        }

        for (int r = 0; r < R; ++r) {
            float e = expf((float) x[r * C + c] - m);
            y[r * C + c] = (hexlib_hf) (e / s);
        }
    }
}

/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: indexing the cos/sin table by HEAD instead of by TOKEN --
 * `costab + h*D` instead of `costab + t*D`.
 *
 * WHY ANYONE WOULD WRITE IT. x is [T, H, D] and the natural stride pattern
 * for "the other tensor associated with this loop" is to match whichever
 * index is closer at hand -- and this loop nests `h` inside `t`, so `h` is
 * the index most recently touched when the table lookup is written. cos/sin
 * are [T, D], with NO HEAD AXIS AT ALL (kernel_api.h), so `costab + h*D` is a
 * LEGAL, IN-BOUNDS read whenever h < T (true here, H=3 <= T=6) -- it does not
 * crash or read garbage, it silently reads the wrong row.
 *
 * WHY THE HARNESS CAN CATCH IT: only because T != H. This kernel reads row
 * `h` (always in [0, H)) instead of row `t` (in [0, T)). For every token
 * t >= H (here, t = 3, 4, 5 of 6), EVERY head reads one of rows {0, 1, 2}
 * instead of its own row {3, 4, 5} -- entirely wrong, for half the tokens, at
 * every head. Even for t < H it is wrong whenever h != t. A harness with
 * T == H (or a table whose rows all happened to agree) would let this pass by
 * coincidence; kernels/transpose_th_fp16's T=8,H=3 choice exists for exactly
 * this reason and this kernel's T=6,H=3 follows it.
 */
#include "kernel_api.h"

void rope_2d_fp16(const hexlib_hf *x, const float *costab, const float *sintab,
                  hexlib_hf *y, int T, int H, int D) {
    if (T <= 0 || H <= 0 || D <= 0) {
        return;
    }
    const int half = D / 2;

    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < H; ++h) {
            /* WRONG: indexed by h, but cos/sin have no head axis -- the
             * correct row is t. */
            const float *cr = costab + (long) h * D;
            const float *sr = sintab + (long) h * D;
            const hexlib_hf *xr = x + ((long) t * H + h) * D;
            hexlib_hf *yr = y + ((long) t * H + h) * D;
            for (int i = 0; i < half; ++i) {
                const float x0 = (float) xr[i];
                const float x1 = (float) xr[i + half];
                yr[i]        = (hexlib_hf) (x0 * cr[i]        - x1 * sr[i]);
                yr[i + half] = (hexlib_hf) (x1 * cr[i + half] + x0 * sr[i + half]);
            }
        }
    }
}

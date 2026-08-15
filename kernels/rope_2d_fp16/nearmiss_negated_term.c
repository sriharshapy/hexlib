/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: negating the wrong term. The rotation is
 *   y[i]      = x[i]     *cos[i]      - x[i+half]*sin[i]
 *   y[i+half] = x[i+half]*cos[i+half] + x[i]     *sin[i+half]
 * and this kernel drops the minus sign on the FIRST line, computing
 *   y[i]      = x[i]     *cos[i]      + x[i+half]*sin[i]      (WRONG)
 *   y[i+half] = x[i+half]*cos[i+half] + x[i]     *sin[i+half] (still right)
 *
 * WHY ANYONE WOULD WRITE IT. The two output halves look almost symmetric --
 * both are "cos of my own half plus sin of the other half's contribution" --
 * and the ONE sign that breaks that symmetry (rotate_half's `-x[...,half:]`
 * from structural.py:253) is easy to lose when writing the two lines side by
 * side, especially by someone who has just finished proving the SPLIT-HALF
 * pairing is right and is now transcribing the four products from memory
 * rather than re-reading the slice/negate/concat.
 *
 * WHY IT SURVIVES A SHAPE CHECK. A same-shaped, plausible-looking rotated
 * output; only a reference comparison with cos/sin that are not both zero at
 * once catches the missing sign.
 */
#include "kernel_api.h"

void rope_2d_fp16(const hexlib_hf *x, const float *costab, const float *sintab,
                  hexlib_hf *y, int T, int H, int D) {
    if (T <= 0 || H <= 0 || D <= 0) {
        return;
    }
    const int half = D / 2;

    for (int t = 0; t < T; ++t) {
        const float *cr = costab + (long) t * D;
        const float *sr = sintab + (long) t * D;
        for (int h = 0; h < H; ++h) {
            const hexlib_hf *xr = x + ((long) t * H + h) * D;
            hexlib_hf *yr = y + ((long) t * H + h) * D;
            for (int i = 0; i < half; ++i) {
                const float x0 = (float) xr[i];
                const float x1 = (float) xr[i + half];
                /* WRONG: should be x0*cr[i] - x1*sr[i]. */
                yr[i]        = (hexlib_hf) (x0 * cr[i]        + x1 * sr[i]);
                yr[i + half] = (hexlib_hf) (x1 * cr[i + half] + x0 * sr[i + half]);
            }
        }
    }
}

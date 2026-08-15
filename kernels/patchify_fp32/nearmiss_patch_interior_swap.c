/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: writing each patch's pixels with the two INTERIOR patch axes
 * swapped -- feature index `((c*T+t)*patch + pw)*patch + ph` instead of the
 * real `((c*T+t)*patch + ph)*patch + pw`. Every row and token lands in
 * exactly the right place (this kernel still runs the correct token
 * arithmetic); only the 16x16 (or PF2's 3x3) patch itself comes out
 * transposed inside its row.
 *
 * WHY ANYONE WOULD WRITE IT. The op moves through FOUR nested index
 * variables per patch (c, t, ph, pw) and only the last two, ph and pw, are
 * ever the same size (patch == patch) -- so a hand-written feature-index
 * expression that puts them in the wrong order is dimensionally invisible:
 * `(... * patch + ph) * patch + pw` and `(... * patch + pw) * patch + ph`
 * both type-check, both produce a value in [0, patch*patch), and neither
 * looks more "obviously right" than the other by inspection. This is the
 * task's own "walking the patch in row-major when the layout wants
 * patch-interior last" -- pw must be the fastest-varying (innermost) axis
 * per kernel_api.h's derivation from structural.py:206's axis order, and
 * this near-miss makes ph the innermost one instead.
 *
 * WHY THE HARNESS CATCHES IT: only the DIAGONAL of each patch (ph == pw)
 * lands in its correct feature slot; every off-diagonal element is written
 * to its transpose's slot instead. With distinct values at every input
 * position, an off-diagonal swap is never masked by two positions
 * coincidentally holding the same number.
 */
#include "kernel_api.h"

void patchify_fp32(const float *img, float *out,
                    int C, int T, int H, int W,
                    int patch, int merge, int grid_h, int grid_w) {
    if (C <= 0 || T <= 0 || H <= 0 || W <= 0 || patch <= 0 || merge <= 0
        || grid_h <= 0 || grid_w <= 0) {
        return;
    }

    const int Bw = grid_w / merge;
    const int out_cols = C * T * patch * patch;

    for (int gh = 0; gh < grid_h; ++gh) {
        const int bh = gh / merge;
        const int mh = gh % merge;

        for (int gw = 0; gw < grid_w; ++gw) {
            const int bw = gw / merge;
            const int mw = gw % merge;
            const int token = ((bh * Bw + bw) * merge + mh) * merge + mw;

            for (int c = 0; c < C; ++c) {
                for (int t = 0; t < T; ++t) {
                    for (int ph = 0; ph < patch; ++ph) {
                        const int h = gh * patch + ph;
                        const long src_row = (((long) c * T) + t) * H * W
                                             + (long) h * W;

                        for (int pw = 0; pw < patch; ++pw) {
                            const int w = gw * patch + pw;
                            /* WRONG: pw and ph swapped in the feature index. */
                            const long feat = (long) ((c * T + t) * patch + pw)
                                               * patch + ph;
                            out[(long) token * out_cols + feat] =
                                img[src_row + w];
                        }
                    }
                }
            }
        }
    }
}

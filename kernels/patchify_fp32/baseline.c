#include "kernel_api.h"

/* Scalar reference. Correct and obvious, never fast.
 *
 * Written directly against the grid, deliberately NOT reusing kernel.c's
 * row-staging structure: for every (gh, gw) patch in the grid, in plain grid
 * order, compute which output row it lands in (structural.py:204-206's merge
 * reshape/transpose) and copy that one patch's (C, T, ph, pw) pixels
 * (structural.py:202's H/W split) straight out of the image. No arithmetic
 * anywhere -- the comparison against the kernel is exact.
 *
 * See kernel_api.h for the full citation of hexlib/graph/opdefs/
 * structural.py's patchify OpDef, whose numpy reference this mirrors.
 */
void patchify_fp32_baseline(const float *img, float *out,
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
                        const long feat_row = (long) ((c * T + t) * patch + ph)
                                               * patch;

                        for (int pw = 0; pw < patch; ++pw) {
                            const int w = gw * patch + pw;
                            out[(long) token * out_cols + feat_row + pw] =
                                img[src_row + w];
                        }
                    }
                }
            }
        }
    }
}

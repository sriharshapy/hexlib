/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: emitting output rows in plain raster order (token = gh *
 * grid_w + gw) and never looking at `merge` at all.
 *
 * WHY ANYONE WOULD WRITE IT. `merge` looks, from the op's attrs alone, like
 * it could be pure metadata for a downstream consumer -- "the merger will
 * group these 2x2 later" -- rather than something THIS op has to act on.
 * grid_h and grid_w are the only two attrs that appear in the output SHAPE
 * (grid_h*grid_w rows), so it is easy to conclude the op is "just flatten
 * the grid" and never notice that the registry's own reference
 * (structural.py:172-207) reshapes the grid into merge blocks and
 * transposes them into (bh, bw, mh, mw) order BEFORE flattening -- i.e.
 * `merge` changes which row a given patch lands in, not just how a later op
 * groups rows that are already in raster order. kernel_api.h's own
 * docstring calls this out explicitly for exactly this reason.
 *
 * WHY THE HARNESS CATCHES IT: raster order and merge-block order agree only
 * for grid cells inside the very first merge block (bh == bw == 0), where
 * `gh * grid_w + gw` and the real block-order formula both evaluate to
 * small, coincidentally-matching numbers for a couple of entries -- but they
 * diverge everywhere else (e.g. at merge=2, grid_w=16: real gh=1, gw=0 lands
 * at token 2, raster order puts it at token 16), so almost every row of a
 * 16x16 or 4x4 grid ends up on the wrong output row. Every input element has
 * a distinct value, so a row landing in the wrong place is not masked by two
 * rows coincidentally holding the same numbers.
 */
#include "kernel_api.h"

#define PATCHIFY_ROWBUF_MAX 256

void patchify_fp32(const float *img, float *out,
                    int C, int T, int H, int W,
                    int patch, int merge, int grid_h, int grid_w) {
    (void) merge;   /* WRONG: never consulted. */

    if (C <= 0 || T <= 0 || H <= 0 || W <= 0 || patch <= 0
        || grid_h <= 0 || grid_w <= 0 || W > PATCHIFY_ROWBUF_MAX) {
        return;
    }

    const int out_cols = C * T * patch * patch;

    for (int c = 0; c < C; ++c) {
        for (int t = 0; t < T; ++t) {
            const float *chan = img + (((long) c * T) + t) * H * W;

            for (int h = 0; h < H; ++h) {
                const float *src_row = chan + (long) h * W;

                const int gh = h / patch;
                const int ph = h % patch;
                const int feat_row_base = ((c * T + t) * patch + ph) * patch;

                for (int gw = 0; gw < grid_w; ++gw) {
                    /* WRONG: plain raster order, ignoring the merge-block
                     * reordering the registry's reference performs. */
                    const int token = gh * grid_w + gw;

                    float *dst = out + (long) token * out_cols + feat_row_base;
                    const float *rp = src_row + (long) gw * patch;
                    for (int pw = 0; pw < patch; ++pw) {
                        dst[pw] = rp[pw];
                    }
                }
            }
        }
    }
}

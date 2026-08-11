/* [C, T, H, W] -> [grid_h*grid_w, C*T*patch*patch], fp32. See kernel_api.h
 * for the full index-arithmetic derivation (taken verbatim from
 * hexlib/graph/opdefs/structural.py's patchify OpDef, lines 145-207) and the
 * "WHY IT IS FAST" note this implementation follows.
 *
 * SHAPE OF THE MOVEMENT. grid_w * patch == W, so a full row of the image, for
 * one (c, t, h), is W contiguous fp32 with no arithmetic performed on it at
 * all -- it just needs to land in grid_w different destination rows, sliced
 * into patch-wide (16-float) pieces. The read side of that is a genuinely
 * whole-vector-aligned bulk move whenever W is a multiple of the 32-lane fp32
 * vector (true for the encoder's only shape, W=256=8*32); the write side
 * never is, because each destination row lives PF_PATCH*PF_PATCH*C*T floats
 * away from its neighbours and no run longer than one patch-row (16 floats,
 * half a vector) is ever contiguous in the destination either. So: HVX for
 * the row-wide read, scalar for the redistribution -- there is nothing to
 * gain from forcing the scatter into vector form (it would need real
 * cross-lane shuffles for no benefit, since the store addresses are neither
 * contiguous nor a fixed stride HVX can address directly).
 */
#include "kernel_api.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>

#define VEC_BYTES 128
#define VEC_FLOATS (VEC_BYTES / (int) sizeof(float)) /* 32 */

/* The encoder's only shape has W = grid_w * patch = 256; this is a hard cap
 * on the local row-staging buffer, not a general limit on the op itself. */
#define PATCHIFY_ROWBUF_MAX 256

void patchify_fp32(const float *img, float *out,
                    int C, int T, int H, int W,
                    int patch, int merge, int grid_h, int grid_w) {
    if (C <= 0 || T <= 0 || H <= 0 || W <= 0 || patch <= 0 || merge <= 0
        || grid_h <= 0 || grid_w <= 0 || W > PATCHIFY_ROWBUF_MAX) {
        return;
    }

    const int Bw = grid_w / merge;
    const int out_cols = C * T * patch * patch;

    const int nvec = W / VEC_FLOATS;      /* whole vectors per row */
    const int tail = W - nvec * VEC_FLOATS;

    float rowbuf[PATCHIFY_ROWBUF_MAX] __attribute__((aligned(VEC_BYTES)));

    for (int c = 0; c < C; ++c) {
        for (int t = 0; t < T; ++t) {
            const float *chan = img + (((long) c * T) + t) * H * W;

            for (int h = 0; h < H; ++h) {
                const float *src_row = chan + (long) h * W;

                /* Bulk-load the whole row through the vector unit: every
                 * element of it is needed by SOME destination patch, so this
                 * is not speculative over-fetch, just wider instructions for
                 * work the scalar loop below would do one float at a time. */
                const HVX_Vector *sv = (const HVX_Vector *) src_row;
                HVX_Vector *dv = (HVX_Vector *) rowbuf;
                for (int v = 0; v < nvec; ++v) {
                    dv[v] = sv[v];
                }
                for (int w = W - tail; w < W; ++w) {
                    rowbuf[w] = src_row[w];
                }

                const int gh = h / patch;
                const int ph = h % patch;
                const int bh = gh / merge;
                const int mh = gh % merge;
                const int feat_row_base = ((c * T + t) * patch + ph) * patch;

                for (int gw = 0; gw < grid_w; ++gw) {
                    const int bw = gw / merge;
                    const int mw = gw % merge;
                    const int token = ((bh * Bw + bw) * merge + mh) * merge + mw;

                    float *dst = out + (long) token * out_cols + feat_row_base;
                    const float *rp = rowbuf + (long) gw * patch;
                    for (int pw = 0; pw < patch; ++pw) {
                        dst[pw] = rp[pw];
                    }
                }
            }
        }
    }
}

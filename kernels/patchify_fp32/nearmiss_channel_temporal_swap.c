/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: computing the channel offset as if the image were packed
 * [T, C, H, W] instead of the real [C, T, H, W] -- i.e. treating T as the
 * OUTER stride (C*H*W... no, T*H*W) and C as the inner one, rather than the
 * other way round.
 *
 * WHY ANYONE WOULD WRITE IT. C and T sit right next to each other at the
 * front of the shape, both are small (3 and 2 for the encoder), and the
 * combined index expression `(c * T + t) * H * W` genuinely looks
 * symmetric with its wrong twin `(t * C + c) * H * W` -- swapping which of
 * the two multiplies the OTHER's extent is a one-character stride
 * confusion, not a logic error a reviewer would spot by eye. It is exactly
 * the class of mistake this project's own task description calls out:
 * "transposing the channel and temporal axes ... a stride confusion
 * produces a same-sized wrong answer."
 *
 * WHY THE HARNESS CATCHES IT: only because C != T. Both orderings produce a
 * [C*T, H, W]-shaped flattened channel axis of the same total size, so
 * every shape check still passes -- but for C=3, T=2 the two strides (T*H*W
 * vs C*H*W) differ, so almost every (c, t) pair reads from the wrong
 * channel-major slab. At C == T the two expressions would be identical and
 * this kernel would be silently CORRECT, which is why the harness's PF2_*
 * shape deliberately uses C=2, T=3 (as well as the encoder's own C=3, T=2)
 * rather than a square C==T shape that would let this slip through.
 */
#include "kernel_api.h"

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

    for (int c = 0; c < C; ++c) {
        for (int t = 0; t < T; ++t) {
            /* WRONG: (t * C + c), as though the image were packed
             * [T, C, H, W] rather than the real [C, T, H, W]. */
            const float *chan = img + (((long) t * C) + c) * H * W;

            for (int h = 0; h < H; ++h) {
                const float *src_row = chan + (long) h * W;

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
                    const float *rp = src_row + (long) gw * patch;
                    for (int pw = 0; pw < patch; ++pw) {
                        dst[pw] = rp[pw];
                    }
                }
            }
        }
    }
}

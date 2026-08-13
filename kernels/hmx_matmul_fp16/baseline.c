/* kernels/hmx_matmul_fp16/baseline.c
 *
 * The scalar reference, in the TILED layout the engine reads, so the comparison
 * is of arithmetic and not of a layout the baseline invented.
 *
 * LAYOUT, stated so it can be wrong loudly rather than quietly: both operands
 * are `n_dot_tiles` consecutive 32x32 row-major tiles. Activation tile t holds
 * a[m][32*t + j] at element (m, j); weight tile t holds b[32*t + j][n] at
 * element (j, n). Accumulation runs over all tiles and all 32 elements within
 * each, so K = 32 * n_dot_tiles.
 *
 * If the hardware disagrees -- llama.cpp's own weight repacker mentions a
 * "crouton" order with every two rows transposed -- the gate FAILS rather than
 * silently comparing two wrong things, which is the point of writing the
 * reference against a named layout instead of against the kernel.
 */
#include "kernel_api.h"

void hmx_matmul_fp16_baseline(const hexlib_hf *act, const hexlib_hf *wt,
                              const unsigned int *scales, hexlib_hf *out) {
    for (int m = 0; m < HMX_MM_M; ++m) {
        for (int n = 0; n < HMX_MM_N; ++n) {
            /* Word n is the fp16 pair for COLUMN n: low half scale, high half
             * bias. Decoded through a union rather than a cast so the fp16
             * bit pattern is read as fp16 and not reinterpreted. */
            union { unsigned short u; __fp16 h; } sc, bi;
            sc.u = (unsigned short) (scales[n] & 0xFFFFu);
            bi.u = (unsigned short) (scales[n] >> 16);
            float sum = 0.0f;
            for (int t = 0; t < HMX_MM_DOT_TILES; ++t) {
                const hexlib_hf *a = act + (long) t * HMX_TILE_ELMS;
                const hexlib_hf *b = wt  + (long) t * HMX_TILE_ELMS;
                for (int j = 0; j < 32; ++j) {
                    sum += (float) a[m * 32 + j] * (float) b[j * 32 + n];
                }
            }
            out[m * HMX_MM_N + n] = (hexlib_hf) (sum * (float) sc.h + (float) bi.h);
        }
    }
}

/* [T, H, D] -> [H, T, D] by moving whole vectors.
 *
 * WHY IT IS FAST. There is no arithmetic here at all, so the only question is how
 * many bytes move per instruction. D is untouched by the permutation, so D
 * contiguous elements move as a unit; when that unit is a whole number of
 * 128-byte vectors it moves as aligned vector loads and stores, 128 bytes per
 * instruction instead of 2. At the encoder's D=64 the unit is exactly ONE
 * vector, so the inner loop is a single load and a single store and there is
 * nothing left to optimise: the op is bounded by memory, and every access is
 * aligned and full width.
 *
 * The loop order is chosen so the WRITES are sequential within a head: for a
 * fixed h, t ascending walks y contiguously. The reads then stride by H*D. That
 * is the right way round -- a store that misses is more expensive than a load
 * that misses, because the load has a prefetcher's help and the store queue is
 * what stalls.
 */
#include "kernel_api.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>

#define VEC_BYTES 128

void transpose_th_fp16(const hexlib_hf *x, hexlib_hf *y, int T, int H, int D) {
    if (T <= 0 || H <= 0 || D <= 0) {
        return;
    }

    const int run_bytes = D * (int) sizeof(hexlib_hf);
    const int nvec = run_bytes / VEC_BYTES;      /* whole vectors per run */
    const int tail = D - nvec * (VEC_BYTES / (int) sizeof(hexlib_hf));

    for (int h = 0; h < H; ++h) {
        for (int t = 0; t < T; ++t) {
            const hexlib_hf *src = x + ((long) t * H + h) * D;
            hexlib_hf *dst = y + ((long) h * T + t) * D;

            const HVX_Vector *sv = (const HVX_Vector *) src;
            HVX_Vector *dv = (HVX_Vector *) dst;
            for (int v = 0; v < nvec; ++v) {
                dv[v] = sv[v];
            }
            for (int d = D - tail; d < D; ++d) {
                dst[d] = src[d];
            }
        }
    }
}

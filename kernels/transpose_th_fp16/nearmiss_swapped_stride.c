/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: using H where T belongs in the OUTPUT stride --
 * `(h * H + t)` instead of `(h * T + t)`.
 *
 * WHY ANYONE WOULD WRITE IT. The output is [H, T, D], so its row stride is T*D,
 * not H*D. But `h` is the index being multiplied, and pairing `h` with `H` reads
 * naturally and is what the fingers type. The input stride genuinely IS
 * `(t * H + h)`, with t paired against H, so the two lines look symmetric when
 * the wrong one is written and asymmetric when the right one is.
 *
 * WHY THE HARNESS CAN CATCH IT: only because T != H. At T == H the two
 * expressions are identical and this kernel is CORRECT. The encoder's real shape
 * is T=256, H=12, so they differ there -- but a harness built on a square shape
 * would pass this and ship it. TR_T is 8 and TR_H is 3 for exactly this reason.
 */
#include "kernel_api.h"

void transpose_th_fp16(const hexlib_hf *x, hexlib_hf *y, int T, int H, int D) {
    if (T <= 0 || H <= 0 || D <= 0) {
        return;
    }
    for (int h = 0; h < H; ++h) {
        for (int t = 0; t < T; ++t) {
            const hexlib_hf *src = x + ((long) t * H + h) * D;
            /* WRONG: h * H, but the output's row stride is T * D. */
            hexlib_hf *dst = y + ((long) h * H + t) * D;
            for (int d = 0; d < D; ++d) {
                dst[d] = src[d];
            }
        }
    }
}

/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: using D where T belongs in the OUTPUT stride -- the output row
 * for a fixed (b, d) is placed at `((b * D + d) * D + t)` instead of
 * `((b * D + d) * T + t)`.
 *
 * WHY ANYONE WOULD WRITE IT. The read side genuinely strides by D (successive
 * t's in x are D elements apart), and that D is sitting right there in the
 * expression one line above. The output's row stride is T -- the length of
 * the row being written, not the length of the row being read -- but D is the
 * variable that was just typed, and reusing it instead of reaching for T is
 * exactly the kind of copy-paste-adjacent slip that survives a quick read: the
 * two lines look symmetric when the wrong one is written and asymmetric when
 * the right one is.
 *
 * WHY THE HARNESS CAN CATCH IT: only because T != D. At T == D the two
 * expressions are identical and this kernel is CORRECT. The encoder's real
 * shape is T=256, D=64, so they differ there -- but a harness built on a
 * square shape would pass this and ship it. TR_T is 64 and TR_D is 7 for
 * exactly this reason.
 */
#include "kernel_api.h"

void transpose_hd_fp16(const hexlib_hf *x, hexlib_hf *y, int B, int T, int D) {
    if (B <= 0 || T <= 0 || D <= 0) {
        return;
    }
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int d = 0; d < D; ++d) {
                hexlib_hf v = x[((long) b * T + t) * D + d];
                /* WRONG: D used as the output row stride, but the output row
                 * (fixed b, d, varying t) has length T, not D. */
                y[((long) b * D + d) * D + t] = v;
            }
        }
    }
}

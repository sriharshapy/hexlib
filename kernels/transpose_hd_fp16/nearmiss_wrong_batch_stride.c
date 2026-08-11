/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: the transpose WITHIN each batch is correct -- t and d are
 * swapped exactly as they should be -- but the batch dimension is walked as
 * though y's flat layout were (T, D, B), batch LAST, instead of the actual
 * (B, D, T), batch FIRST.
 *
 * WHY ANYONE WOULD WRITE IT. Someone gets the hard part right -- the
 * inner two-axis swap that is the whole point of this kernel -- and then
 * places the batch index using the same "outermost index gets the biggest
 * stride" intuition that would be correct if batch actually varied slowest in
 * memory, without checking that y is still allocated (B, D, T) and not some
 * other order the intuition assumed. It is the mistake of getting the
 * documented part of the contract right and the undocumented part (which axis
 * is actually outermost in the buffer you were handed) wrong.
 *
 * WHY THE HARNESS CAN CATCH IT: because B, T and D are three distinct values,
 * this lands every batch's data at a completely different -- and wrongly
 * sized -- set of offsets than the real (B, D, T) layout, so almost nothing
 * ends up where transpose_hd_fp16_baseline puts it.
 */
#include "kernel_api.h"

void transpose_hd_fp16(const hexlib_hf *x, hexlib_hf *y, int B, int T, int D) {
    if (B <= 0 || T <= 0 || D <= 0) {
        return;
    }
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int d = 0; d < D; ++d) {
                /* WRONG: walks y as though its shape were (T, D, B) -- batch
                 * last -- instead of the real (B, D, T) -- batch first. */
                y[((long) t * D + d) * B + b] = x[((long) b * T + t) * D + d];
            }
        }
    }
}

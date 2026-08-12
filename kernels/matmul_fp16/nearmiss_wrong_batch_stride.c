/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: computing the per-batch pointer offset for A from `m` and `k`
 * (the loop variables already in scope) instead of from `b` -- so every batch
 * reads and writes batch 0's slice of A. B and C use the correct `b`-scaled
 * offset, which is what makes this "plausible": the bug is a single wrong
 * multiplier on one of three pointers, not a structural rewrite.
 *
 * WHY ANYONE WOULD WRITE IT. `M * K` is the per-batch element count for A,
 * and `(long) m * K` (the correct per-ROW offset within a batch) is a
 * visually similar expression to `(long) b * M * K` (the correct per-BATCH
 * offset) -- both are "some index times K, cast to long, added to a base
 * pointer". A copy-paste that grabs the row-offset idiom for the batch-offset
 * site compiles clean and is correct for b=0.
 *
 * WHY THE HARNESS CATCHES IT: it is caught only because A differs across
 * batches. harness.c's fill formula mixes `b` into every element of A (and of
 * B, though B's offset here is not the bug), so batch 1 and batch 2 read
 * batch 0's A values entirely -- wrong on nearly every output element in
 * those two batches, not just a rounding-sized difference.
 */
#include "kernel_api.h"

void matmul_fp16(const hexlib_hf *A, const hexlib_hf *B, hexlib_hf *C,
                  int Bn, int M, int K, int N) {
    for (int b = 0; b < Bn; ++b) {
        /* WRONG: always batch 0 of A, regardless of b. */
        const hexlib_hf *Ab = A;
        const hexlib_hf *Bb = B + (long) b * K * N;
        hexlib_hf *Cb = C + (long) b * M * N;
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    acc += (float) Ab[(long) m * K + k] * (float) Bb[(long) k * N + n];
                }
                Cb[(long) m * N + n] = (hexlib_hf) acc;
            }
        }
    }
}

#include "kernel_api.h"

/* Scalar reference. Correct and obvious, never fast.
 *
 * Matches hexlib/graph/opdefs/structural.py:55-61's reference lambda
 * (`arrays[0] @ arrays[1]`) at the precision hexlib/graph/eager.py:20-29 fixes
 * for it -- see kernel_api.h's SPEC comment for the derivation. The K-reduction
 * is accumulated in `float` (never `hexlib_hf`) and rounded to fp16 exactly
 * once, at the final store; that is the one property this file must not get
 * wrong, because it is the file every near-miss and the kernel itself are
 * checked against.
 */
void matmul_fp16_baseline(const hexlib_hf *A, const hexlib_hf *B, hexlib_hf *C,
                           int Bn, int M, int K, int N) {
    for (int b = 0; b < Bn; ++b) {
        const hexlib_hf *Ab = A + (long) b * M * K;
        const hexlib_hf *Bb = B + (long) b * K * N;
        hexlib_hf *Cb = C + (long) b * M * N;
        for (int m = 0; m < M; ++m) {
            const hexlib_hf *arow = Ab + (long) m * K;
            hexlib_hf *crow = Cb + (long) m * N;
            for (int n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    acc += (float) arow[k] * (float) Bb[(long) k * N + n];
                }
                crow[n] = (hexlib_hf) acc;
            }
        }
    }
}

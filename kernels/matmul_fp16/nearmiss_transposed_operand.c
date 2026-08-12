/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: reading B as if it were stored [N, K] row-major (the "weight
 * transposed" convention many BLAS-style APIs default to) instead of the
 * [K, N] row-major layout this op's own spec requires (kernel_api.h's SPEC
 * comment / hexlib/graph/opdefs/structural.py's "weights arrive pre-
 * transposed to [k, n]" docstring). Concretely: `B[n*K + k]` instead of the
 * correct `B[k*N + n]`.
 *
 * WHY ANYONE WOULD WRITE IT. "matmul with the second operand transposed" is
 * such a common BLAS convention (sgemm's TRANSB) that swapping the stride
 * pair is an easy slip, especially once QK^T is on your mind -- attention's
 * QK^T literally IS A @ K^T, so a kernel author moving between the two matmul
 * call sites can genuinely misremember which one this generic kernel expects.
 *
 * WHY THE HARNESS CATCHES IT: this indexing stays in-bounds regardless of the
 * shape (max index n*K+k = (N-1)*K+(K-1) < N*K always), so it cannot be
 * caught by a bounds check or a crash -- only by comparing values. The
 * harness's B data is NOT symmetric under this transpose (kernels/
 * transpose_th_fp16's own convention: B/M/K/N are all different numbers here,
 * and the fill formula mixes k and n asymmetrically), so this near-miss
 * reads essentially unrelated elements and produces a wrong value at nearly
 * every output position, not just the adversarial one.
 */
#include "kernel_api.h"

void matmul_fp16(const hexlib_hf *A, const hexlib_hf *B, hexlib_hf *C,
                  int Bn, int M, int K, int N) {
    for (int b = 0; b < Bn; ++b) {
        const hexlib_hf *Ab = A + (long) b * M * K;
        const hexlib_hf *Bb = B + (long) b * K * N;
        hexlib_hf *Cb = C + (long) b * M * N;
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    /* WRONG: should be Bb[k*N + n] -- B is [K, N], not
                     * [N, K]. */
                    acc += (float) Ab[(long) m * K + k] * (float) Bb[(long) n * K + k];
                }
                Cb[(long) m * N + n] = (hexlib_hf) acc;
            }
        }
    }
}

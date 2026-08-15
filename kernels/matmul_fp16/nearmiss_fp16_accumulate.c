/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: accumulating the K-reduction in fp16 (__fp16, rounding to fp16
 * after EVERY multiply-add) instead of float32. Everything else is IDENTICAL
 * to baseline.c -- same loop nest, same data, same final store. Only the type
 * of the running sum changes.
 *
 * WHY ANYONE WOULD WRITE IT. A and B are both fp16, the product of two fp16
 * values "looks like" it belongs in an fp16 accumulator, and on a target
 * where an fp16 register is one lane width and float32 is two, accumulating
 * in the storage dtype looks consistent rather than careless. Nothing about
 * the C source looks unstable; the loop is the same loop.
 *
 * WHY THE HARNESS CATCHES IT: see harness.c's header comment for the full
 * derivation. In short, its adversarial element (batch 1, row 5, column 7) is
 * one dominant product (2.0) plus 127 identical followers, each exactly one
 * fp16 ULP at the accumulator's own magnitude (2^-10 at ~1-2) -- individually
 * too small to survive round-to-even once added to an accumulator that has
 * already reached that magnitude. MEASURED (Python, numpy float32/float16,
 * this exact algorithm): float32 sum 2.1240234375, this near-miss's fp16-
 * accumulated sum 2.0 -- an absolute error of 0.1240234375, ~127x the correct
 * kernel's own single-rounding error (0.0009765625) and ~5.8% relative,
 * comfortably outside the harness's tolerance (rel=1e-2, abs=1e-3) on both
 * measures. On the harness's OTHER (non-adversarial) elements this bug is far
 * smaller and can be invisible -- that is exactly why the harness does not
 * rely on generic data alone to catch it.
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
                /* WRONG: the running sum is __fp16, so it rounds to fp16
                 * after every single multiply-add instead of accumulating in
                 * float32. */
                hexlib_hf acc16 = (hexlib_hf) 0.0f;
                for (int k = 0; k < K; ++k) {
                    float p = (float) Ab[(long) m * K + k] * (float) Bb[(long) k * N + n];
                    acc16 = (hexlib_hf) ((float) acc16 + p);
                }
                Cb[(long) m * N + n] = acc16;
            }
        }
    }
}

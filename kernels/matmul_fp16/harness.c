/* kernels/matmul_fp16/harness.c
 *
 * Builds inputs, runs the baseline for reference, times ONLY the kernel call,
 * compares with tolerance, and prints the two lines the driver parses.
 *
 * SHAPE: (Bn,M,K,N) = (MM_B,MM_M,MM_K,MM_N) = (3, 40, 128, 192). All four
 * DELIBERATELY DIFFERENT numbers -- kernels/transpose_th_fp16's own
 * convention (see its harness.c header comment): with any two of B/M/K/N
 * equal, a stride-confusion or transposed-operand bug can produce a
 * same-shape, same-size result that a shape check (or an unlucky data set)
 * cannot see. All four distinct here means nearmiss_wrong_batch_stride.c and
 * nearmiss_transposed_operand.c cannot pass by accident. N is a multiple of
 * 64 (the fp16 HVX vector width) so this harness's own timed run exercises
 * kernel.c's fully vectorised column-block path, not its scalar tail -- the
 * near-misses below are therefore rejected by the SAME code path the real
 * encoder shapes (N = 256 or 64, both multiples of 64) use.
 *
 * GENERIC DATA. A[b][m][k] and B[b][k][n] are deterministic, small, and
 * exact multiples of 0.25 (exactly representable in fp16), built from a
 * formula that mixes b, m/k, and k/n so no two batches and no row/column look
 * alike -- this is what makes nearmiss_wrong_batch_stride.c (always reads
 * batch 0) and nearmiss_transposed_operand.c (reads B with (k,n) swapped)
 * fail on ordinary data, without needing special-casing for them.
 *
 * THE ADVERSARIAL ELEMENT is C[1][5][7] -- batch 1, row 5, column 7 -- and it
 * exists for exactly one purpose: to make accumulating the K-reduction in
 * fp16 instead of float32 (nearmiss_fp16_accumulate.c) a LARGE, unmistakable
 * error rather than a rounding-noise near-miss. Read this carefully, because
 * a per-element tolerance loose enough to admit the real kernel's own
 * legitimate noise can be looser than a real bug -- that already happened
 * once in this repo (layernorm_fp16's unbiased-variance near-miss was wrongly
 * accepted on its first run because the bug's size was smaller than fp16's
 * own ULP noise; see kernels/softmax_fp16's harness.c for the same lesson
 * applied to a fp16-accumulated sum).
 *
 * A[1][5][k] is set to 2.0 for k=0 and 2^-10 = 0.0009765625 (one fp16 ULP at
 * magnitude ~1-2) for k=1..127 (127 identical followers); B[1][k][7] is set
 * to 1.0 for every k. So C[1][5][7] = 2.0*1.0 + 127 * (2^-10 * 1.0).
 *
 * MEASURED IN PYTHON (numpy float32 / float16, same algorithm
 * nearmiss_fp16_accumulate.c implements, K = MM_K = 128 terms: one 2.0 term
 * plus 127 followers of 2^-10):
 *   float32 accumulate                         = 2.1240234375
 *   correctly rounded ONCE to fp16 (this spec)  = 2.125       (diff from the
 *                                                  float32 sum: 0.0009765625,
 *                                                  exactly one fp16 ULP -- the
 *                                                  single unavoidable rounding
 *                                                  every correct implementation
 *                                                  pays exactly once)
 *   fp16-accumulated (the near-miss's bug)      = 2.0         (diff from the
 *                                                  float32 sum: 0.1240234375 --
 *                                                  127x the correct kernel's
 *                                                  own single-rounding error,
 *                                                  ~5.8% relative)
 * Every one of the 127 identical 2^-10 increments rounds away once the fp16
 * accumulator reaches magnitude ~2 (fp16 ULP there is 2^-9 = 0.001953125, so a
 * 2^-10 increment is exactly half a ULP and never survives round-to-even) --
 * the same "many increments each near the accumulator's own ULP, added one at
 * a time" shape that made kernels/softmax_fp16's row 1 discriminating.
 *
 * TOLERANCE (see hexlib_close_f16 calls below): rel=1e-2, abs=1e-3. The
 * correct kernel's worst case on this element (one legitimate fp16 rounding,
 * 0.0009765625 absolute) sits just under the absolute bound and far under the
 * relative one (1e-2 * 2.125 = 0.02125); the near-miss's error (0.1240234375)
 * clears BOTH by more than an order of magnitude (~124x the absolute bound,
 * ~5.8x the relative one). This tolerance was derived from those two measured
 * numbers, not loosened until something passed.
 *
 * Rows/columns/batches not involved in the adversarial element use the
 * generic formula everywhere, including at batch 1 row 5 and column 7 outside
 * k -- there is nothing special about this harness beyond the one element
 * needed to discriminate the one quiet bug a friendly random data set would
 * hide.
 */
#include "hexlib/hexlib_harness.h"
#include "kernel_api.h"

void matmul_fp16_baseline(const hexlib_hf *, const hexlib_hf *, hexlib_hf *,
                           int, int, int, int);

#define MM_ADV_B 1
#define MM_ADV_M 5
#define MM_ADV_N 7

static hexlib_hf A[MM_B * MM_M * MM_K]   HEXLIB_ALIGN;
static hexlib_hf Bm[MM_B * MM_K * MM_N]  HEXLIB_ALIGN;
static hexlib_hf C[MM_B * MM_M * MM_N]   HEXLIB_ALIGN;
static hexlib_hf REF[MM_B * MM_M * MM_N] HEXLIB_ALIGN;

static void fill(void) {
    for (int b = 0; b < MM_B; ++b) {
        for (int m = 0; m < MM_M; ++m) {
            for (int k = 0; k < MM_K; ++k) {
                int q = ((b * 131 + m * 17 + k * 7) % 13) - 6;   /* -6..6 */
                A[(long) b * MM_M * MM_K + (long) m * MM_K + k] =
                    (hexlib_hf) ((float) q * 0.25f);
            }
        }
        for (int k = 0; k < MM_K; ++k) {
            for (int n = 0; n < MM_N; ++n) {
                int q = ((b * 89 + k * 13 + n * 5) % 15) - 7;    /* -7..7 */
                Bm[(long) b * MM_K * MM_N + (long) k * MM_N + n] =
                    (hexlib_hf) ((float) q * 0.25f);
            }
        }
    }

    /* The adversarial row/column -- see the header comment for the numbers. */
    for (int k = 0; k < MM_K; ++k) {
        float av = (k == 0) ? 2.0f : (1.0f / 1024.0f);   /* 2^-10, exact in fp16 */
        A[(long) MM_ADV_B * MM_M * MM_K + (long) MM_ADV_M * MM_K + k] = (hexlib_hf) av;
        Bm[(long) MM_ADV_B * MM_K * MM_N + (long) k * MM_N + MM_ADV_N] = (hexlib_hf) 1.0f;
    }

    for (int i = 0; i < MM_B * MM_M * MM_N; ++i) {
        C[i] = (hexlib_hf) 12345.0f;   /* poison: a no-op kernel cannot pass */
    }
}

int main(void) {
    fill();

    matmul_fp16_baseline(A, Bm, REF, MM_B, MM_M, MM_K, MM_N);

    unsigned long long kcyc = 0;
    HEXLIB_TIME_KERNEL(kcyc, matmul_fp16(A, Bm, C, MM_B, MM_M, MM_K, MM_N));

    int n_wrong = 0;
    double max_err = 0.0;
    for (int i = 0; i < MM_B * MM_M * MM_N; ++i) {
        if (!hexlib_close_f16((float) C[i], (float) REF[i], 1e-2f, 1e-3f)) {
            ++n_wrong;
        }
        double d = (double) (float) C[i] - (double) (float) REF[i];
        if (d < 0.0) d = -d;
        if (d > max_err) max_err = d;
    }

    hexlib_report(n_wrong == 0, n_wrong, max_err, kcyc);
    return 0;
}

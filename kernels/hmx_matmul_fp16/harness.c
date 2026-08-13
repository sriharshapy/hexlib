/* kernels/hmx_matmul_fp16/harness.c
 *
 * THREE THINGS THIS HARNESS DOES DELIBERATELY:
 *
 *  1. THE INPUT IS NOT UNIFORM. A tile filled with one repeated value gives the
 *     same answer under ANY permutation of its elements, so a kernel with a
 *     completely wrong tile layout would pass. Every element here is distinct
 *     modulo a small period and the two operands use DIFFERENT periods, so a
 *     transposed tile, a swapped operand pair or a wrong dot-tile stride all
 *     move the answer.
 *
 *  2. THE BIAS IS NON-ZERO AND NOT CONSTANT. `bias = mxmem2()` is set once
 *     before the multiply and it is easy to write a kernel that ignores it and
 *     still matches on a zero bias.
 *
 *  3. THE OUTPUT IS POISONED before the call. A kernel that writes nothing --
 *     which is exactly what the split-packet failure looks like on the readout
 *     path -- fails on the poison rather than passing on a zeroed buffer.
 *
 * TOLERANCE. HMX accumulates internally and narrows once at the store, while
 * the baseline accumulates in fp32 and narrows per element. Over K=64 those
 * differ by more than one fp16 ULP, so the comparison uses the same
 * hexlib_close_f16 (rel 0.02, abs 1e-3) every fp16 kernel here uses. The check
 * that a wrong LAYOUT cannot hide inside that tolerance is point 1, not the
 * tolerance.
 */
#include "hexlib/hexlib_harness.h"
#include "kernel_api.h"

/* TILE-SIZED ALIGNMENT, NOT HEXLIB_ALIGN'S 128 BYTES. `mxmem` addresses a
 * 2048-byte tile and the first version of this harness used the project's
 * standard 128-byte alignment, which faulted with badva0=04114a68 -- an address
 * whose low 11 bits are not zero. */
#define HMX_TILE_ALIGN __attribute__((aligned(2048)))
static hexlib_hf ACT [HMX_MM_DOT_TILES * HMX_TILE_ELMS] HMX_TILE_ALIGN;
static hexlib_hf WT  [HMX_MM_DOT_TILES * HMX_TILE_ELMS] HMX_TILE_ALIGN;
/* 256 bytes, per-column (scale, bias) fp16 pairs -- NOT a 32x32 tile. */
static unsigned int SCALES[HMX_SCALES_WORDS] __attribute__((aligned(256)));
static hexlib_hf OUT [HMX_MM_M * HMX_MM_N]              HMX_TILE_ALIGN;
static hexlib_hf REF [HMX_MM_M * HMX_MM_N]              HEXLIB_ALIGN;

int main(void) {
    /* Small magnitudes: K=64 accumulations of products must stay inside fp16's
     * range, and this kernel is establishing a SEQUENCE, not overflow behaviour. */
    for (int i = 0; i < HMX_MM_DOT_TILES * HMX_TILE_ELMS; ++i) {
        ACT[i] = (hexlib_hf) (((float) ((i % 11) - 5)) * 0.125f);
        WT[i]  = (hexlib_hf) (((float) ((i % 7)  - 3)) * 0.250f);
    }
    /* Scale 1.0 (0x3c00) with a per-column bias that VARIES, so a kernel that
     * ignores the scale operand cannot match. Columns beyond 32 are padding. */
    for (int i = 0; i < HMX_SCALES_WORDS; ++i) {
        SCALES[i] = 0u;
    }
    for (int n = 0; n < HMX_MM_N; ++n) {
        union { unsigned short u; __fp16 h; } bi;
        bi.h = (__fp16) (((float) ((n % 5) - 2)) * 0.5f);
        SCALES[n] = 0x3c00u | ((unsigned int) bi.u << 16);
    }
    for (int i = 0; i < HMX_MM_M * HMX_MM_N; ++i) {
        OUT[i] = (hexlib_hf) 12345.0f;
    }

    hmx_matmul_fp16_baseline(ACT, WT, SCALES, REF);

    unsigned long long kcyc = 0;
    HEXLIB_TIME_KERNEL(kcyc, hmx_matmul_fp16(ACT, WT, SCALES, OUT));

    int n_wrong = 0;
    double max_err = 0.0;
    for (int i = 0; i < HMX_MM_M * HMX_MM_N; ++i) {
        if (!hexlib_close_f16((float) OUT[i], (float) REF[i], 0.02f, 1e-3f)) {
            ++n_wrong;
        }
        double d = (double) (float) OUT[i] - (double) (float) REF[i];
        if (d < 0.0) d = -d;
        if (d > max_err) max_err = d;
    }

    hexlib_report(n_wrong == 0, n_wrong, max_err, kcyc);
    return 0;
}

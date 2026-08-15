/* kernels/hmx_matmul_fp16/nearmiss_split_packet.c
 *
 * THE FAILURE THIS PROJECT ALREADY MADE ONCE, PRESERVED.
 *
 * Identical to kernel.c except the activation and weight loads are issued as
 * TWO packets instead of one. `docs/hardware/hmx-int8.md` records four rounds of
 * probing that concluded from exactly this that the tile engine "could not be
 * made to accumulate", and derived a precise law (`out = bias_high >> 7`) for
 * behaviour that was purely an artifact of the split.
 *
 * It is the most dangerous kind of wrong: it compiles, it runs, it writes an
 * output of the right shape, and the readout path works. Only the values are
 * wrong, and they are wrong in a way that looks like a hardware limitation
 * rather than a coding error.
 */
#include "kernel_api.h"

void hmx_matmul_fp16(const hexlib_hf *act, const hexlib_hf *wt,
                     const unsigned int *scales, hexlib_hf *out) {
    const unsigned int range = (unsigned int) (HMX_TILE_BYTES * HMX_MM_DOT_TILES - 1);

    asm volatile("bias = mxmem2(%0)\n" :: "r"(scales));
    asm volatile("mxclracc.hf\n");
    /* THE DEFECT: two packets where the working kernel has one set of braces. */
    asm volatile("activation.hf = mxmem(%1, %0):deep\n" :: "r"(range), "r"(act));
    asm volatile("weight.hf = mxmem(%1, %0)\n" :: "r"(range), "r"(wt));
    asm volatile("mxmem(%0, %1):after.hf = acc\n" :: "r"(out), "r"(0) : "memory");
}

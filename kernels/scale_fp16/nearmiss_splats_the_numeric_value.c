/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: splatting the factor by CASTING it to int rather than
 * reinterpreting its fp16 bit pattern.
 *
 * Q6_Vh_vsplat_R takes an integer register and copies its low 16 bits into
 * every lane. It does not know or care what those bits mean. So the value handed
 * to it must already BE the fp16 encoding. `(int) factor` instead converts the
 * number: 0.125 truncates to 0, every lane receives 0x0000, and the kernel
 * multiplies the whole array by zero.
 *
 * This is the mistake kernel.c's union exists to prevent, and it is worth a
 * near-miss because the wrong version looks simpler and compiles without a
 * warning. It fails loudly here -- an all-zero output -- but the same confusion
 * with a factor like 2.5 would truncate to 2 and produce an answer that is
 * merely wrong rather than obviously wrong.
 */
#include "kernel_api.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>

#define LANES_FP16 64

void scale_fp16(const hexlib_hf *x, hexlib_hf *y, int n, float factor) {
    if (n <= 0) {
        return;
    }
    /* WRONG: converts the numeric value instead of reinterpreting the bits. */
    const HVX_Vector vfactor = Q6_Vh_vsplat_R((int) factor);

    const int nvec = n / LANES_FP16;
    const HVX_Vector *xv = (const HVX_Vector *) x;
    HVX_Vector *yv = (HVX_Vector *) y;

    for (int i = 0; i < nvec; ++i) {
        yv[i] = Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(xv[i], vfactor));
    }
    for (int i = nvec * LANES_FP16; i < n; ++i) {
        y[i] = (hexlib_hf) ((float) x[i] * factor);
    }
}

/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: assuming n is a multiple of the 64-lane fp16 vector and handling
 * no tail. This is the single cheapest bug to write in any HVX kernel -- the
 * vector loop is the interesting part, the tail is boilerplate, and it is easy
 * to convince yourself the caller's shape is always a multiple.
 *
 * WHY IT WOULD SURVIVE A CARELESS HARNESS. The encoder's actual shape is
 * 12*256*64 = 196608, which IS a multiple of 64. A harness built from the real
 * caller's shape would never execute the tail path and this kernel would pass
 * the gate, ship, and then silently corrupt the last few elements the first
 * time anything called it with a different length. SCALE_N is 4100 = 64*64 + 4
 * precisely so the tail exists.
 *
 * The harness poisons the output buffer before the call, so the four untouched
 * elements keep the poison value rather than holding a plausible number.
 */
#include "kernel_api.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>

#define LANES_FP16 64

static inline HVX_Vector splat_fp16(float f) {
    union {
        hexlib_hf h;
        unsigned short u;
    } bits;
    bits.h = (hexlib_hf) f;
    return Q6_Vh_vsplat_R((int) bits.u);
}

void scale_fp16(const hexlib_hf *x, hexlib_hf *y, int n, float factor) {
    if (n <= 0) {
        return;
    }
    const HVX_Vector vfactor = splat_fp16(factor);
    const int nvec = n / LANES_FP16;
    const HVX_Vector *xv = (const HVX_Vector *) x;
    HVX_Vector *yv = (HVX_Vector *) y;

    for (int i = 0; i < nvec; ++i) {
        yv[i] = Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(xv[i], vfactor));
    }
    /* WRONG: no tail loop. Elements nvec*64 .. n-1 are never written. */
}

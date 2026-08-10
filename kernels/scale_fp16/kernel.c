/* Scale a contiguous fp16 array by a scalar, 64 lanes at a time.
 *
 * WHY IT IS FAST. The whole operation is one pass with no reduction and no
 * cross-lane dependency, so the only thing that matters is how many elements
 * each instruction touches and whether the loads and stores are aligned. A
 * 128-byte HVX vector holds 64 fp16 lanes, so one multiply replaces 64 scalar
 * multiplies, and the aligned load/store pair costs one instruction each
 * instead of the two an unaligned access needs. The scalar factor is converted
 * to fp16 and splatted ONCE, outside the loop -- doing it inside would add a
 * scalar-to-vector transition per iteration for a value that never changes.
 *
 * WHY THE MULTIPLY GOES THROUGH qf32. There is no single-vector fp16 multiply
 * that keeps its result in fp16; the available primitive widens a pair of fp16
 * vectors into a qf32 VectorPair. So the sequence is: multiply into qf32
 * (Q6_Wqf32_vmpy_VhfVhf), then narrow back to IEEE fp16
 * (Q6_Vhf_equals_Wqf32). This is not a detour -- it is what gives the
 * arithmetic more precision than fp16 in the middle, matching the scalar
 * baseline, which also multiplies in float and rounds once on store. qf32 is
 * Qualcomm's internal float format, not IEEE, and is treated as opaque here:
 * nothing inspects its bits.
 *
 * THE TAIL IS SCALAR ON PURPOSE. n is not required to be a multiple of 64. A
 * masked vector store would work, but the tail is at most 63 elements out of
 * however many there are, so the scalar loop costs nothing measurable and is
 * obviously correct. The encoder's own shape (12*256*64 = 196608) has no tail
 * at all.
 */
#include "kernel_api.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>

#define LANES_FP16 64

/* One fp16 value in all 64 lanes.
 *
 * Q6_Vh_vsplat_R takes an INTEGER register and copies its low 16 bits to every
 * lane, so the fp16 must be reinterpreted as its bit pattern first. The union
 * is that reinterpretation; a cast would convert the numeric value instead and
 * splat something else entirely.
 */
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
        /* Widen-multiply into qf32, then narrow back to IEEE fp16. */
        yv[i] = Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(xv[i], vfactor));
    }

    /* Tail: whatever did not fill a whole vector. */
    for (int i = nvec * LANES_FP16; i < n; ++i) {
        y[i] = (hexlib_hf) ((float) x[i] * factor);
    }
}

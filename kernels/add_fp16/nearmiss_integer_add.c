/* A plausible WRONG implementation the harness must reject.
 *
 * THE MISTAKE: Q6_Vh_vadd_VhVh instead of Q6_Vhf_vadd_VhfVhf -- a 16-bit
 * INTEGER add where a 16-bit FLOAT add was meant.
 *
 * WHY ANYONE WOULD WRITE IT. The two intrinsics differ by two characters. Both
 * take and return the same HVX_Vector type, because HVX_Vector is just 128 bytes
 * and carries no element type at all, so the compiler cannot object. `h` is the
 * natural-looking tag for a 16-bit lane; that it means "halfword integer" while
 * "halfword float" is `hf` is a convention you have to know. There is no
 * warning, no cast, and nothing at the call site to suggest anything is wrong.
 *
 * WHAT IT ACTUALLY COMPUTES: the sum of the two IEEE fp16 bit patterns
 * interpreted as signed integers. For same-sign operands with equal exponents
 * this can even land near the right answer, which is what makes it worth a
 * near-miss rather than being caught by any input at all. The harness uses both
 * signs throughout, and fp16 is sign-magnitude rather than two's complement, so
 * a negative operand makes this produce a result with the wrong sign entirely.
 */
#include "kernel_api.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>

#define LANES_FP16 64

void add_fp16(const hexlib_hf *a, const hexlib_hf *b, hexlib_hf *y, int n) {
    if (n <= 0) {
        return;
    }
    const int nvec = n / LANES_FP16;
    const HVX_Vector *av = (const HVX_Vector *) a;
    const HVX_Vector *bv = (const HVX_Vector *) b;
    HVX_Vector *yv = (HVX_Vector *) y;

    for (int i = 0; i < nvec; ++i) {
        /* WRONG: integer add of the fp16 bit patterns. */
        yv[i] = Q6_Vh_vadd_VhVh(av[i], bv[i]);
    }
    for (int i = nvec * LANES_FP16; i < n; ++i) {
        y[i] = (hexlib_hf) ((float) a[i] + (float) b[i]);
    }
}

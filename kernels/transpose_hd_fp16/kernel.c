/* [B, T, D] -> [B, D, T] by gathering strided reads scalar-wise and committing
 * them to the contiguous output row with HVX vector stores.
 *
 * WHY THE WRITE SIDE, NOT THE READ SIDE. See kernel_api.h for the full
 * argument. In short: for a fixed (b, d), the output row y[b][d][:] is T
 * contiguous elements, while the corresponding input column x[b][:][d] is T
 * elements each D*2 bytes apart -- there is no run shared by both sides to
 * hand a vector load. Exactly one side can be made contiguous by loop order,
 * and it is cheaper to spend the vector unit on the side that is more
 * expensive to get wrong (stores stall the store queue; loads have a
 * prefetcher's help).
 *
 * HD_VLEN elements (one HVX vector's worth of fp16, 128 bytes / 2 = 64) are
 * gathered into a small aligned stack buffer with scalar loads, then written
 * out in one vector store via HVX_UVector -- the compiler-recognized
 * "unaligned vector" pointer type from the SDK's hexagon_types.h, documented
 * in this repo's own include/hexlib/hvx/hvx-base.h (`hvx_vmemu`). It is needed
 * because T*2 bytes need not put every output row on a 128-byte boundary, so
 * an aligned vector store is not always safe here.
 *
 * This is a genuine but PARTIAL acceleration: the store side is vectorized,
 * the load side stays scalar. A full cross-lane HVX transpose (vdelta/vshuff)
 * would vectorize both sides; that path was attempted and set aside -- see the
 * kernel-transpose-hd report for what was tried and why.
 */
#include "kernel_api.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>

/* Elements per HVX vector at fp16: 128 bytes / 2 bytes each = 64. */
#define HD_VLEN (128 / (int) sizeof(hexlib_hf))

void transpose_hd_fp16(const hexlib_hf *x, hexlib_hf *y, int B, int T, int D) {
    if (B <= 0 || T <= 0 || D <= 0) {
        return;
    }

    /* Aligned so it can be read back as a single HVX_Vector below. */
    hexlib_hf buf[HD_VLEN] __attribute__((aligned(128)));

    for (int b = 0; b < B; ++b) {
        const hexlib_hf *xb = x + (long) b * T * D;
        hexlib_hf *yb = y + (long) b * D * T;

        for (int d = 0; d < D; ++d) {
            const hexlib_hf *src_col = xb + d;      /* stride D between t's */
            hexlib_hf *dst_row = yb + (long) d * T; /* contiguous over t */

            int t = 0;
            for (; t + HD_VLEN <= T; t += HD_VLEN) {
                for (int i = 0; i < HD_VLEN; ++i) {
                    buf[i] = src_col[(long) (t + i) * D];
                }
                const HVX_Vector *bv = (const HVX_Vector *) buf;
                HVX_UVector *dv = (HVX_UVector *) (dst_row + t);
                *dv = *bv;
            }
            /* Scalar tail: fewer than one whole vector's worth of t remains. */
            for (; t < T; ++t) {
                dst_row[t] = src_col[(long) t * D];
            }
        }
    }
}

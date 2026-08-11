/* kernels/transpose_hd_fp16/kernel_api.h */
#ifndef HEXLIB_TRANSPOSE_HD_FP16_API_H
#define HEXLIB_TRANSPOSE_HD_FP16_API_H

typedef __fp16 hexlib_hf;

/* [B, T, D] -> [B, D, T], fp16. perm (0, 2, 1) -- the QK^T operand layout move.
 *
 *   y[b][d][t] = x[b][t][d]
 *
 * 12 ops in the encoder, at B=12 T=256 D=64.
 *
 * WHY THIS ONE IS HARD (see kernels/transpose_th_fp16/kernel_api.h for the easy
 * sibling, perm (1,0,2), and read its header first -- this comment assumes it).
 * There, D is untouched and innermost on BOTH operands, so a whole run of D
 * contiguous elements moves as one unit. Here the permutation swaps the
 * innermost two axes, so D is innermost on the INPUT and T is innermost on the
 * OUTPUT: no run longer than one element is contiguous on both sides at once.
 * A naive scalar loop is either a D*2-byte-strided load or a D*2-byte-strided
 * store (D=64, fp16 -> 128 bytes = exactly one HVX vector at the encoder
 * shape), and whichever side you make contiguous, the other is fully strided.
 * There is no run of shared contiguity to hand to a vector load/store the way
 * transpose_th_fp16 does.
 *
 * THE CHOICE MADE HERE: make the WRITE side contiguous and batch it into HVX
 * vector stores; leave the READ side scalar. Two independent reasons:
 *   1. For a fixed (b, d), y[b][d][:] is T contiguous elements -- the output's
 *      whole innermost run -- while x[b][:][d] is T elements each D*2 bytes
 *      apart, with no contiguity on the read side to exploit at any stride.
 *      Only one side of this op can ever be made contiguous by choice of loop
 *      order; here that side is the write.
 *   2. transpose_th_fp16's own header argues that a store that misses is more
 *      expensive than a load that misses, because the load has a prefetcher's
 *      help and the store queue is what stalls. Batching the more expensive
 *      side into whole vectors is the side worth batching.
 * So the inner loop gathers up to one HVX vector's worth of elements from the
 * strided read (scalar, unavoidable -- there is no contiguous run to load) into
 * a small aligned local buffer, then commits it with a SINGLE HVX vector store
 * to the contiguous output row (unaligned-safe, via HVX_UVector, since T*2
 * bytes need not put every row on a 128-byte boundary). At the encoder's
 * T=256, D=64 this turns 256 scalar stores per (b, d) row into 4 vector stores;
 * the scalar reads are unchanged in count, but the effort that matters -- the
 * store side -- is genuinely vectorized.
 *
 * NOT A FULL HVX TRANSPOSE. A true cross-lane 64x64 vector transpose (a
 * vdelta/vshuff network) would vectorize the read side too, turning this into
 * a real 64x speedup on both sides rather than one. That was attempted and set
 * aside for this version -- see the accompanying report for what was tried and
 * why it was not landed. This version's HVX use is real and ELF-provable (a
 * genuine vector load and a genuine vector store per chunk) but PARTIAL: it
 * accelerates the store side only, not the load side.
 *
 * ALIGNMENT: none required. x and y may start at any address; the gather
 * buffer is aligned internally by the kernel, and the output store always uses
 * HVX's unaligned vector-store form.
 *
 * A second shape whose run is NOT a whole number of vectors, so the harness
 * exercises the scalar-only tail path too. TR_T_ODD * 2 = 80 bytes < 128, so no
 * chunk of it ever reaches a whole vector.
 */
#define TR_B 5
#define TR_T 64
#define TR_D 7

#define TR_T_ODD 40

void transpose_hd_fp16(const hexlib_hf *x, hexlib_hf *y, int B, int T, int D);

#endif

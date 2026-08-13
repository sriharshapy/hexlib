/* kernels/hmx_matmul_fp16/kernel_api.h */
#ifndef HEXLIB_HMX_MATMUL_FP16_API_H
#define HEXLIB_HMX_MATMUL_FP16_API_H

typedef __fp16 hexlib_hf;

/* THE FIRST HMX KERNEL IN THIS PROJECT. Deliberately the smallest shape that
 * exercises the whole instruction sequence: one 32x32 fp16 output tile, a
 * K depth of two 32-element dot tiles.
 *
 * WHY SO SMALL. Everything downstream -- weight repacking, deep-K accumulation
 * past 32 dot tiles, folding into matmul_epilogue -- rests on the tile
 * sequence being right, and that sequence has ALREADY been got wrong once in
 * this project's own history. `docs/hardware/hmx-int8.md` records four probe
 * rounds that concluded the tile engine "could not be made to accumulate" and
 * derived a precise, wrong law for it. The cause was issuing the activation and
 * weight loads as two separate packets. So this kernel establishes the sequence
 * at a shape where a scalar reference is trivially checkable, before anything is
 * built on top of it.
 *
 * THE SEQUENCE, from ../llama.cpp/ggml/src/ggml-hexagon/htp/hmx-utils.h:207-220
 * (MIT; see ATTRIBUTION.md) and its caller `core_dot_chunk_fp16_short`
 * (hmx-mm-kernels-tiled.h:604-629):
 *
 *     bias = mxmem2(scales)                   <- ONCE, before the loops
 *     mxclracc.hf                             <- per OUTPUT tile
 *     { activation.hf = mxmem(act, range):deep
 *       weight.hf     = mxmem(wt,  range)  }  <- ONE PACKET. see below.
 *     mxmem(out, 0):after.hf = acc
 *
 * with `range = 2048 * n_dot_tiles - 1`.
 *
 * THE ONE-PACKET RULE IS THE WHOLE POINT. The braces are not formatting. HMX
 * forms the multiply from an activation/weight pair issued together in a single
 * instruction packet; split across two packets the accumulator's contribution is
 * not degraded, it is CLEARED -- which reads as "the readout works but nothing
 * accumulates", exactly the false conclusion the int8 investigation reached.
 * `nearmiss_split_packet.c` preserves that failure.
 *
 * THE "BIAS" OPERAND IS NOT A BIAS TILE, AND GETTING THIS WRONG CRASHED THE
 * FIRST VERSION OF THIS KERNEL. `HMX_SET_BIAS` takes a 256-BYTE area, not a
 * 32x32 tile: `hmx_init_column_scales` (hmx-utils.h:19-23) writes one HVX
 * vector of per-COLUMN packed words then one zero vector. Each 32-bit word is
 * an fp16 PAIR -- low half the multiplicative scale, high half the additive
 * bias -- which is what upstream's `Q6_V_vsplat_R(0x3c00)` means when its
 * comment says "scale: 1.0, bias: 0.0 in FP16" (0x3c00 is fp16 1.0 in the low
 * half, 0 in the high). 32 words covers 32 columns = 128 bytes; the second
 * vector is padding.
 *
 * `docs/hardware/hmx-int8.md` recorded "the bias tile is a scale, not an
 * additive bias" from int8 probing. That was half of it: it is BOTH, one pair
 * per column.
 *
 * ==========================================================================
 * WHY THIS KERNEL CANNOT BE GATED BY `hexlib test`, ESTABLISHED BY READING
 * llama.cpp RATHER THAN BY MORE PROBING.
 * ==========================================================================
 * HMX needs FOUR things before a single `mxmem` will execute, and a standalone
 * simulator ELF -- which is what every other kernel in this repository is
 * gated as -- has none of them. All four are in
 * ../llama.cpp/ggml/src/ggml-hexagon/htp (MIT; see ATTRIBUTION.md):
 *
 *   1. POWER. `main.c:473-492`, guarded `#if __HVX_ARCH__ >= 75`, and SEPARATE
 *      from the HVX power request beside it:
 *          request.type = HAP_power_set_HMX_v2;
 *          request.hmx_v2.set_power = TRUE;  .power_up = TRUE;
 *          request.hmx_v2.set_clock = TRUE;
 *          request.hmx_v2.target_corner = HAP_DCVS_EXP_VCORNER_MAX;  (min, max too)
 *          request.hmx_v2.perf_mode = HAP_CLK_PERF_HIGH;
 *          HAP_power_set(ctx, &request);
 *
 *   2. ACQUISITION, in the SAME compute-res attr as VTCM, not a separate one
 *      (`vtcm_alloc`, main.c:259-291):
 *          HAP_compute_res_attr_set_hmx_param(&attr, 1);
 *          rctx = HAP_compute_res_acquire(&attr, 1000000);
 *
 *   3. AN EXPLICIT LOCK around every use (`hmx-queue.c:17-30`):
 *          HAP_compute_res_hmx_lock(rctx);   ... mxmem ...   _hmx_unlock(rctx);
 *
 *   4. A DEDICATED THREAD owning that lock -- upstream queues all HMX work to
 *      `hmx_queue_thread`, created only `if (n_hmx)` (main.c:386-394). Whether
 *      the lock is thread-scoped, and therefore whether hexlib's
 *      single-threaded skel can hold it inline, IS NOT SETTLED and must not be
 *      assumed either way.
 *
 * WHAT THAT MEANS FOR hexlib. The pieces map almost one-to-one onto code that
 * already exists here, which is why this is a short list and not a redesign:
 *
 *   - `session.c` passes n_hmx = 0 unconditionally today. llama.cpp's
 *     `htp_iface_start(..., n_hvx, n_hmx, max_vmem)` is the same signature
 *     hexlib ported, and `n_hmx` was always meant for exactly this.
 *   - `skel_vtcm.c:102` ALREADY has `if (ctx->n_hmx > 0)
 *     HAP_compute_res_attr_set_hmx_param(&attr, 1);` -- requirement 2, written
 *     and never once executed.
 *   - Requirements 1 and 3 do not exist in hexlib at all.
 *
 * So an HMX kernel belongs on the QuRT-hosted BATCH path (skel + VTCM + a real
 * compute-res context), not on the standalone-ELF gate path. The kernel source
 * below is probably fine; it was being run somewhere it can never work. The
 * fault it produces there is exception 0x18 with badva on the stack.
 *
 * TILE GEOMETRY. A 32x32 fp16 tile is 1024 elements and 2048 bytes
 * (HTP_MM_HMX_TILE_N_ELMS = 1024, matmul-ops.h:19). Operands are SYMMETRIC in
 * fp16 mode -- both 2048 bytes -- unlike the int8 path, where the activation
 * tile is masked at 2047 and the weight at 1023.
 */
#define HMX_MM_M          32          /* output rows                          */
#define HMX_MM_N          32          /* output cols                          */
#define HMX_MM_DOT_TILES   2          /* K = 32 * this                        */
#define HMX_MM_K          (32 * HMX_MM_DOT_TILES)
#define HMX_TILE_ELMS   1024
#define HMX_TILE_BYTES  2048
/* One HVX vector of per-column (scale, bias) words + one vector of padding. */
#define HMX_SCALES_BYTES 256
#define HMX_SCALES_WORDS (HMX_SCALES_BYTES / 4)

void hmx_matmul_fp16(const hexlib_hf *act, const hexlib_hf *wt,
                     const unsigned int *scales, hexlib_hf *out);

void hmx_matmul_fp16_baseline(const hexlib_hf *act, const hexlib_hf *wt,
                              const unsigned int *scales, hexlib_hf *out);

#endif

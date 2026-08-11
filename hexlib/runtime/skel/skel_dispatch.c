/* hexlib/runtime/skel/skel_dispatch.c -- validate a batch, then walk it.
 *
 * THE RESPONSE HEADER GOES DOWN FIRST, with a non-OK status. Every early return
 * therefore leaves a readable failure, and there is no path on which the host
 * reads a zero-filled buffer and has to guess. Status OK is 1, so an unwritten
 * buffer cannot read as success.
 *
 * PCYCLE BRACKETS THE KERNEL CALL AND NOTHING ELSE -- not tensor resolution, not
 * mapping, not the response write. Harness overhead is roughly constant, so
 * including it manufactures ratios out of nothing; this is the same counter and
 * the same placement `hexlib.sim` reports, which is what makes sim-vs-silicon
 * comparison mean anything.
 *
 * EVERY OFFSET IS VALIDATED IN 64-BIT ARITHMETIC BEFORE A BYTE IS READ. `len` and
 * `rsp_cap` are uint32_t, and so are the wire offsets and counts, so a host-
 * supplied count multiplied by a struct size can wrap a 32-bit accumulator back
 * into range and turn a bounds check into a lie -- in particular there is no
 * upper bound on `n_ops` on the wire, so an unwidened `off_ops + n_ops *
 * sizeof(op_desc)` could overflow back under `len` for a large enough n_ops.
 * Every offset+size computation below widens to uint64_t first for exactly that
 * reason: a host bug must not become an out-of-bounds read or write on the DSP.
 */
#include "skel_internal.h"

#include <string.h>

#include "HAP_farf.h"
#include "HAP_perf.h"

/* THE SDK'S OWN READ, NOT A HAND-ROLLED ONE -- AND THE DIFFERENCE IS NOT
 * COSMETIC. This was `__asm__ __volatile__("%0 = c15:14")`, issued directly.
 * That instruction only advances if SYSCFG.PCYCLEEN is set, and A USER-MODE
 * UNSIGNED PD CANNOT SET THAT BIT: this project's own
 * include/hexlib/hexlib_harness.h sets it explicitly (`hexlib_enable_pcycle`,
 * "bit 5 = PCYCLEEN") precisely because the standalone-ELF runtime it belongs
 * to runs where that is permitted. The skel does not. So the hand-rolled read
 * had no handling of the one precondition it depends on, in the one PD where
 * that precondition may not hold -- and the SIMULATOR CANNOT TELL US, because
 * there the bit is effectively always on and this path measures a plausible
 * four-figure number either way.
 *
 * HAP_perf_get_pcycles() ($HEXAGON_SDK_ROOT/incs/HAP_perf.h) issues the
 * IDENTICAL `C15:14` read -- so this is not a change of mechanism and the
 * measured number is expected to be unchanged (it was: 1287 cycles on the
 * simulator before and after). What changes is whose claim it is. If
 * Qualcomm's own documented perf API returns 0 in an unsigned PD, that is a
 * platform fact about the PD, discoverable from the SDK and reportable as
 * such; if our own inline asm returned 0 it would be indistinguishable from
 * our bug. Reading 0 on silicon remains POSSIBLE -- nothing here prevents it,
 * and no simulator run can rule it out -- which is exactly why main.c prints
 * cycles_total, cli.py now requires it to be > 0, and the on-device test
 * asserts it. See docs §6.1 and this file's PCYCLE note above.
 *
 * KEPT AS A NAMED WRAPPER rather than calling HAP_perf_get_pcycles() at the
 * two sites: the bracketing test (test_skel_dispatch_source.py) locates the
 * before/after pair by this name and checks that ONLY `k->fn(&a)` sits between
 * them, and one name is also the one place to state the above. */
static inline uint64_t hexlib_read_pcycle(void) {
    return (uint64_t) HAP_perf_get_pcycles();
}

/* Shared with skel.c (see skel_internal.h): both callers write the same header
 * shape. `arch` is never a caller-supplied value -- it is always what THIS
 * BINARY was built for, via __HEXAGON_ARCH__, so a skel built for the wrong
 * part shows up as a visible disagreement rather than a silently accepted
 * parameter. */
void hexlib_write_rsp_hdr(uint8_t *rsp, uint32_t status, uint32_t n_ops,
                          uint64_t cycles) {
    struct hexlib_batch_rsp_hdr h;
    memset(&h, 0, sizeof(h));
    h.magic        = HEXLIB_BATCH_MAGIC;
    h.version      = HEXLIB_BATCH_VERSION;
    h.status       = status;
    h.n_ops        = n_ops;
    h.cycles_total = cycles;
    h.arch         = __HEXAGON_ARCH__;
    memcpy(rsp, &h, sizeof(h));
}

int hexlib_dispatch_batch(struct hexlib_ctx *ctx, const uint8_t *batch, uint32_t len,
                          uint8_t *rsp, uint32_t rsp_cap, uint32_t *rsp_len) {
    *rsp_len = 0;
    if (rsp_cap < sizeof(struct hexlib_batch_rsp_hdr)) {
        /* Cannot write anything readable into a buffer this small. There is
         * nothing left to say except at the RPC-return-code level, which is
         * the caller's (skel.c's) job. */
        return HEXLIB_DSP_ERR_TRUNCATED;
    }
    /* FAILURE IS THE DEFAULT. Every return below either overwrites this with a
     * specific status (OK included) or leaves this one in place. */
    hexlib_write_rsp_hdr(rsp, HEXLIB_DSP_ERR_INTERNAL, 0, 0);
    *rsp_len = sizeof(struct hexlib_batch_rsp_hdr);

    if (len < sizeof(struct hexlib_batch_hdr)) {
        hexlib_write_rsp_hdr(rsp, HEXLIB_DSP_ERR_TRUNCATED, 0, 0);
        return HEXLIB_DSP_ERR_TRUNCATED;
    }
    struct hexlib_batch_hdr hdr;
    memcpy(&hdr, batch, sizeof(hdr));

    if (hdr.magic != HEXLIB_BATCH_MAGIC) {
        hexlib_write_rsp_hdr(rsp, HEXLIB_DSP_ERR_BAD_MAGIC, 0, 0);
        return HEXLIB_DSP_ERR_BAD_MAGIC;
    }
    if (hdr.version != HEXLIB_BATCH_VERSION) {
        hexlib_write_rsp_hdr(rsp, HEXLIB_DSP_ERR_BAD_VERSION, 0, 0);
        return HEXLIB_DSP_ERR_BAD_VERSION;
    }
    if (hdr.total_size != len) {
        FARF(ERROR, "hexlib: header says %u bytes, got %u", hdr.total_size, len);
        hexlib_write_rsp_hdr(rsp, HEXLIB_DSP_ERR_TRUNCATED, 0, 0);
        return HEXLIB_DSP_ERR_TRUNCATED;
    }
    if (hdr.n_bufs > HEXLIB_MAX_BUFS || hdr.n_tensors > HEXLIB_MAX_TENSORS) {
        hexlib_write_rsp_hdr(rsp, HEXLIB_DSP_ERR_INVAL_PARAMS, 0, 0);
        return HEXLIB_DSP_ERR_INVAL_PARAMS;
    }
    /* Every section must lie inside the blob, and the response must have room
     * for every op's result -- checked before a byte of either is read. Widened
     * to uint64_t (see file header) so a huge host-supplied n_ops cannot wrap
     * this very check back into passing. */
    if ((uint64_t) hdr.off_bufs + (uint64_t) hdr.n_bufs * sizeof(struct hexlib_buf_desc) > (uint64_t) len ||
        (uint64_t) hdr.off_tensors + (uint64_t) hdr.n_tensors * sizeof(struct hexlib_tensor) > (uint64_t) len ||
        (uint64_t) hdr.off_ops + (uint64_t) hdr.n_ops * sizeof(struct hexlib_op_desc) > (uint64_t) len) {
        hexlib_write_rsp_hdr(rsp, HEXLIB_DSP_ERR_TRUNCATED, 0, 0);
        return HEXLIB_DSP_ERR_TRUNCATED;
    }
    if ((uint64_t) sizeof(struct hexlib_batch_rsp_hdr) +
        (uint64_t) hdr.n_ops * sizeof(struct hexlib_op_result) > (uint64_t) rsp_cap) {
        hexlib_write_rsp_hdr(rsp, HEXLIB_DSP_ERR_TRUNCATED, 0, 0);
        return HEXLIB_DSP_ERR_TRUNCATED;
    }

    /* Working copies: base and data are filled in HERE, never taken from the
     * host's bytes. The input blob is const for exactly that reason.
     *
     * STATIC, NOT STACK OR HEAP: there is no allocator on the DSP side, and 512
     * tensors (~22 KB) is more than is safe to put on a QuRT thread's stack.
     * The trade is that hexlib_dispatch_batch is NOT reentrant -- two concurrent
     * invokes on the same skel instance would corrupt each other's working
     * copy. FastRPC already serializes calls to a single handle, and skel.c
     * hands out exactly one handle (g_ctx), so this holds today; it would need
     * revisiting if that ever changed (e.g. multiple sessions on one skel). */
    static struct hexlib_buf_desc bufs[HEXLIB_MAX_BUFS];
    static struct hexlib_tensor   tens[HEXLIB_MAX_TENSORS];
    memcpy(bufs, batch + hdr.off_bufs, hdr.n_bufs * sizeof(bufs[0]));
    memcpy(tens, batch + hdr.off_tensors, hdr.n_tensors * sizeof(tens[0]));

    int rc = hexlib_bufs_map(ctx, bufs, hdr.n_bufs);
    if (rc != HEXLIB_DSP_OK) {
        hexlib_write_rsp_hdr(rsp, (uint32_t) rc, 0, 0);
        return rc;
    }
    rc = hexlib_tensors_resolve(ctx, bufs, hdr.n_bufs, tens, hdr.n_tensors);
    if (rc != HEXLIB_DSP_OK) {
        hexlib_write_rsp_hdr(rsp, (uint32_t) rc, 0, 0);
        return rc;
    }
    rc = hexlib_vtcm_acquire(ctx);
    if (rc != HEXLIB_DSP_OK) {
        hexlib_write_rsp_hdr(rsp, (uint32_t) rc, 0, 0);
        return rc;
    }

    struct hexlib_op_result *results =
        (struct hexlib_op_result *) (rsp + sizeof(struct hexlib_batch_rsp_hdr));
    uint64_t total = 0;
    uint32_t done = 0;
    int batch_status = HEXLIB_DSP_OK;

    for (uint32_t i = 0; i < hdr.n_ops; i++) {
        struct hexlib_op_desc op;
        memcpy(&op, batch + hdr.off_ops + (uint64_t) i * sizeof(op), sizeof(op));

        /* Never left at whatever was in the response buffer before: filled in
         * before the kernel lookup even runs, same "failure is the default"
         * discipline as the batch-level header above, just per-op. */
        results[i].kind   = op.kind;
        results[i].status = HEXLIB_DSP_ERR_INTERNAL;
        results[i].cycles = 0;
        done = i + 1;

        const struct hexlib_kernel_entry *k = 0;
        for (uint32_t j = 0; j < hexlib_kernel_table_len; j++) {
            if (hexlib_kernel_table[j].kind == op.kind) {
                k = &hexlib_kernel_table[j];
                break;
            }
        }
        if (!k) {
            FARF(ERROR, "hexlib: no kernel for kind %u", op.kind);
            results[i].status = HEXLIB_DSP_ERR_NO_KERNEL;
            batch_status = HEXLIB_DSP_ERR_NO_KERNEL;
            break;
        }

        hexlib_args a;
        memset(&a, 0, sizeof(a));
        uint32_t nb = 0;
        int op_invalid = 0;
        for (uint32_t s = 0; s < HEXLIB_MAX_SRC && !op_invalid; s++) {
            if (op.src[s] == 0xFFFF) continue;
            if (op.src[s] >= hdr.n_tensors || nb >= HEXLIB_MAX_BUFS) {
                op_invalid = 1;
                break;
            }
            struct hexlib_tensor *t = &tens[op.src[s]];
            a.buf[nb]    = (void *) (uintptr_t) t->data;
            a.dtype[nb]  = t->dtype;
            a.layout[nb] = t->layout;
            for (int e = 0; e < 4; e++) a.ne[nb][e] = t->ne[e];
            nb++;
        }
        for (uint32_t o = 0; o < HEXLIB_MAX_DST && !op_invalid; o++) {
            if (op.dst[o] == 0xFFFF) continue;
            if (op.dst[o] >= hdr.n_tensors || nb >= HEXLIB_MAX_BUFS) {
                op_invalid = 1;
                break;
            }
            struct hexlib_tensor *t = &tens[op.dst[o]];
            a.buf[nb]    = (void *) (uintptr_t) t->data;
            a.dtype[nb]  = t->dtype;
            a.layout[nb] = t->layout;
            for (int e = 0; e < 4; e++) a.ne[nb][e] = t->ne[e];
            nb++;
        }
        if (op_invalid) {
            results[i].status = HEXLIB_DSP_ERR_INVAL_PARAMS;
            batch_status = HEXLIB_DSP_ERR_INVAL_PARAMS;
            break;
        }
        a.n_buf     = nb;
        a.vtcm      = ctx->vtcm_base;
        a.vtcm_size = ctx->vtcm_size;
        a.params    = op.params;
        a.n_threads = 1;

        uint64_t t0 = hexlib_read_pcycle();
        int krc = k->fn(&a);
        uint64_t t1 = hexlib_read_pcycle();

        results[i].status = (uint32_t) krc;
        results[i].cycles = t1 - t0;
        total += (t1 - t0);

        if (krc != HEXLIB_DSP_OK) {
            batch_status = krc;
            break;
        }
        /* A competing session asked for VTCM back. Stop cleanly at an op
         * boundary and ACTUALLY GIVE IT BACK: the release callback in
         * skel_vtcm.c only records the request (it must not release memory a
         * batch in flight is still using), so the OS-level release happens
         * here, once we are between ops and genuinely done with it -- not
         * merely stopping while still holding the reservation the competing
         * session is waiting on. */
        if (ctx->vtcm_needs_release) {
            FARF(HIGH, "hexlib: VTCM reclaim requested after op %u of %u",
                 i + 1, hdr.n_ops);
            hexlib_vtcm_release(ctx);
            batch_status = HEXLIB_DSP_ERR_VTCM_RECLAIMED;
            break;
        }
    }

    hexlib_write_rsp_hdr(rsp, (uint32_t) batch_status, done, total);
    *rsp_len = sizeof(struct hexlib_batch_rsp_hdr) +
               done * sizeof(struct hexlib_op_result);
    return batch_status;
}

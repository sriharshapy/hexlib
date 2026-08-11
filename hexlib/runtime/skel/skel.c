/* hexlib/runtime/skel/skel.c -- the FastRPC entry points qaic's skel calls.
 *
 * Session lifecycle adapted from llama.cpp ggml-hexagon htp/main.c (MIT); see
 * ATTRIBUTION.md. `start` deliberately takes no dsp_queue_id: dispatch is a
 * synchronous invoke, because dspqueue has no simulator path.
 *
 * THESE PROTOTYPES ARE THE REAL ONES QAIC GENERATES from
 * runtime/idl/hexlib_iface.idl -- verified by actually running qaic (not
 * derived by hand) into a scratch directory and reading hexlib_iface.h back.
 * Two things that are easy to get wrong by guessing:
 *   - `open`/`close` (qaic's implicit pair for a `: remote_handle64` interface)
 *     return plain `int`, not `AEEResult`. They are the same underlying type
 *     (AEEResult is `typedef int AEEResult`), but the header spells it `int`.
 *   - `invoke` has NO "resultLenOut" parameter. `rout sequence<octet> result`
 *     marshals only a capacity in `resultLen`; there is no channel back to the
 *     host for "how many bytes are actually meaningful" -- the transport
 *     doesn't carry one. The response is therefore self-describing:
 *     `hexlib_batch_rsp_hdr.n_ops` is what the host reads to know how many
 *     `hexlib_op_result` entries follow, not any out-parameter here.
 */
#include "hexlib_iface.h"
#include "skel_internal.h"

#include <string.h>

#include "HAP_farf.h"

static struct hexlib_ctx g_ctx;

int hexlib_iface_open(const char *uri, remote_handle64 *handle) {
    (void) uri;
    memset(&g_ctx, 0, sizeof(g_ctx));
    for (uint32_t i = 0; i < HEXLIB_MAX_MMAPS; i++) {
        g_ctx.mmap[i].fd = -1;
    }
    *handle = (remote_handle64) &g_ctx;
    return AEE_SUCCESS;
}

int hexlib_iface_close(remote_handle64 handle) {
    struct hexlib_ctx *ctx = (struct hexlib_ctx *) handle;
    hexlib_vtcm_release(ctx);
    hexlib_vtcm_free(ctx);
    ctx->started = 0;
    return AEE_SUCCESS;
}

AEEResult hexlib_iface_start(remote_handle64 handle, uint32 sess_id, uint32 n_hvx,
                             uint32 n_hmx, uint64 max_vmem) {
    struct hexlib_ctx *ctx = (struct hexlib_ctx *) handle;
    ctx->sess_id  = sess_id;
    ctx->n_hvx    = n_hvx;
    ctx->n_hmx    = n_hmx;
    ctx->max_vmem = max_vmem;

    int rc = hexlib_vtcm_alloc(ctx);
    if (rc != HEXLIB_DSP_OK) {
        /* CARRY THE SPECIFIC STATUS ACROSS THE RPC BOUNDARY. This returned a
         * bare AEE_EFAILED, which flattened every VTCM failure into the one
         * outcome the host already reports for a dozen unrelated causes, so on
         * a device the operator could not tell VTCM contention from a signing
         * failure, a URI error, or a missing skel -- the same undiagnosable
         * session-open dead end as the arch-decode bug, from a different cause.
         *
         * The FARF above is not enough on its own: it lands in the DSP log,
         * which an operator running a device-farm job may not be able to
         * retrieve. The return value always comes back.
         *
         * HEXLIB_AEE_FROM_STATUS keeps this a nonzero failure for every caller
         * that only tests success, while making the reason recoverable for one
         * that looks. If FastRPC ever normalises the value we lose only the
         * detail, never the failure. */
        ctx->start_status = rc;
        FARF(ERROR, "hexlib: start failed, VTCM rc %d", rc);
        return HEXLIB_AEE_FROM_STATUS(rc);
    }
    ctx->start_status = HEXLIB_DSP_OK;
    ctx->started = 1;
    FARF(HIGH, "hexlib: session %u started, VTCM %u bytes",
         sess_id, (uint32_t) ctx->vtcm_size);
    return AEE_SUCCESS;
}

AEEResult hexlib_iface_stop(remote_handle64 handle) {
    struct hexlib_ctx *ctx = (struct hexlib_ctx *) handle;
    hexlib_vtcm_release(ctx);
    hexlib_vtcm_free(ctx);
    ctx->started = 0;
    return AEE_SUCCESS;
}

AEEResult hexlib_iface_mmap(remote_handle64 handle, uint32 fd, uint32 size) {
    struct hexlib_ctx *ctx = (struct hexlib_ctx *) handle;
    int rc = hexlib_bufs_register(ctx, fd, size);
    return rc == HEXLIB_DSP_OK ? AEE_SUCCESS : AEE_EFAILED;
}

AEEResult hexlib_iface_munmap(remote_handle64 handle, uint32 fd) {
    struct hexlib_ctx *ctx = (struct hexlib_ctx *) handle;
    int rc = hexlib_bufs_unregister(ctx, fd);
    return rc == HEXLIB_DSP_OK ? AEE_SUCCESS : AEE_EFAILED;
}

AEEResult hexlib_iface_hwinfo(remote_handle64 handle, uint32 *arch,
                              uint32 *n_threads, uint32 *n_hvx, uint32 *n_hmx,
                              uint64 *vtcm_size) {
    struct hexlib_ctx *ctx = (struct hexlib_ctx *) handle;
    /* __HEXAGON_ARCH__ is what THIS BINARY was built for; the host cross-checks
     * it against what the driver reports the part to be, so a skel built for
     * the wrong arch is a visible disagreement rather than a mystery. Verified
     * (not assumed) to expand to 75 when compiled -mv75 on the 19.0.04
     * toolchain -- see task-6-report.md for how. */
    *arch      = __HEXAGON_ARCH__;
    *n_threads = 1;
    *n_hvx     = ctx->n_hvx;
    *n_hmx     = ctx->n_hmx;
    /* The ACQUIRED size, never the part's total: vtcm_size test guards this. */
    *vtcm_size = (uint64) ctx->vtcm_size;
    return AEE_SUCCESS;
}

AEEResult hexlib_iface_invoke(remote_handle64 handle, const unsigned char *batch,
                              int batchLen, unsigned char *result, int resultLen) {
    struct hexlib_ctx *ctx = (struct hexlib_ctx *) handle;

    /* Nothing readable can be written into a buffer smaller than the response
     * header itself. The only channel left to say so is the RPC return code. */
    if (resultLen < 0 || (uint32_t) resultLen < sizeof(struct hexlib_batch_rsp_hdr)) {
        FARF(ERROR, "hexlib: invoke result buffer too small (%d)", resultLen);
        return AEE_EFAILED;
    }

    /* BOTH LENGTHS ARE SIGNED ON THE WIRE (qaic spells `sequence<octet>` as a
     * pointer plus an `int`), so both need this and only `resultLen` had it.
     * A negative `batchLen` cast to uint32_t becomes an enormous length, and
     * hexlib_dispatch_batch's first size test is `len < sizeof(struct
     * hexlib_batch_hdr)` -- which a huge value PASSES. It then memcpy()s the
     * full 40-byte header out of `batch` (skel_dispatch.c) BEFORE
     * `hdr.total_size != len` can reject anything: an out-of-bounds read of a
     * buffer the host may have made much shorter than that. Rejecting it here,
     * before the cast, is the only place the sign is still visible.
     *
     * Written into the response and returned AEE_SUCCESS, not returned as an
     * RPC error: the result buffer was just proven big enough for a header
     * (above), so the host can be told exactly what was wrong instead of
     * having a marshalled response discarded. Same reasoning as the
     * invoke-before-start refusal below. */
    if (batchLen < 0) {
        FARF(ERROR, "hexlib: invoke batch length is negative (%d)", batchLen);
        hexlib_write_rsp_hdr(result, HEXLIB_DSP_ERR_INVAL_PARAMS, 0, 0);
        return AEE_SUCCESS;
    }

    if (!ctx->started) {
        /* No op runs -- not even the truncation path inside
         * hexlib_dispatch_batch. The response gets the REAL status
         * (HEXLIB_DSP_ERR_NOT_STARTED), not a generic one reached by feeding
         * hexlib_dispatch_batch a batch length of zero: a host that reads
         * HEXLIB_DSP_ERR_NOT_STARTED off the wire knows exactly what to fix
         * (call start() first), rather than seeing HEXLIB_DSP_ERR_TRUNCATED
         * and wondering whether its own encoder is broken. */
        FARF(ERROR, "hexlib: invoke before start (%d)", HEXLIB_DSP_ERR_NOT_STARTED);
        hexlib_write_rsp_hdr(result, HEXLIB_DSP_ERR_NOT_STARTED, 0, 0);
        return AEE_SUCCESS;
    }

    uint32_t rsp_len = 0;
    int rc = hexlib_dispatch_batch(ctx, batch, (uint32_t) batchLen, result,
                                   (uint32_t) resultLen, &rsp_len);
    /* A non-OK batch still returns AEE_SUCCESS with a populated response: the
     * host reads hexlib_batch_rsp_hdr.status off the wire, and an RPC-level
     * error would discard the response qaic already marshaled back to it.
     * `rsp_len` has no wire home either -- see the file header -- so it is
     * only useful to a caller of hexlib_dispatch_batch directly (e.g. a test),
     * not to this FastRPC entry point. */
    (void) rc;
    (void) rsp_len;
    return AEE_SUCCESS;
}

/* hexlib/runtime/skel/skel_vtcm.c -- acquire VTCM once per session.
 *
 * THE SIZE COMES FROM THE RUNTIME, NEVER A HARDCODED BYTE COUNT OR A FIXED
 * SILICON ADDRESS. VTCM is acquired at session start and what we get is what we
 * may use; the M1 allocator's budget must be this number.
 *
 * AND IT CAN BE TAKEN AWAY. llama.cpp registers a release callback and
 * deliberately drops its own priority so that it RECEIVES a reclaim request from
 * competing sessions -- a QNN-HTP or another GGML-HTP session can take VTCM
 * mid-run. So VTCM is not a static budget the compiler owns. Making the
 * allocator resilient to that is M2's problem; noticing it and failing loudly is
 * this file's.
 *
 * Adapted from llama.cpp ggml-hexagon htp/main.c vtcm_acquire/vtcm_alloc (MIT);
 * see ATTRIBUTION.md. Upstream aborts the process on failure; we return a
 * status instead.
 */
#include "skel_internal.h"

#include "HAP_compute_res.h"
#include "HAP_farf.h"
#include "qurt_thread.h"

static int release_callback(unsigned int rctx, void *state) {
    struct hexlib_ctx *ctx = (struct hexlib_ctx *) state;
    (void) rctx;
    /* Do not release here -- the batch in flight is still using it. Record it,
     * and let the dispatcher finish the current op and report. */
    ctx->vtcm_needs_release = 1;
    return 0;
}

int hexlib_vtcm_alloc(struct hexlib_ctx *ctx) {
    /* BOTH SIZES, AND THE SECOND ONE IS THE POINT. The signature is
     * (application_id, total_block_size, total_block_layout, avail_block_size,
     * avail_block_layout) -- HAP_compute_res.h:1087-1106. `total` is the whole
     * partition assigned to this application type (8388608 on v75); `avail` is
     * the SDK's own words "largest contiguous memory chunk available". An
     * earlier version of this function passed 0 for avail and asked for total
     * as an absolute requirement, which is the bug below. */
    unsigned int vtcm_total = 0;
    unsigned int vtcm_avail = 0;
    if (HAP_compute_res_query_VTCM(0, &vtcm_total, 0, &vtcm_avail, 0) != 0 ||
        vtcm_total == 0) {
        FARF(ERROR, "hexlib: HAP_compute_res_query_VTCM failed");
        return HEXLIB_DSP_ERR_INTERNAL;
    }
    /* Nothing at all is a real failure and a distinct one: the partition
     * exists but every byte of it is held by someone else. */
    if (vtcm_avail == 0) {
        FARF(ERROR, "hexlib: VTCM fully contended -- total %u, available 0",
             vtcm_total);
        return HEXLIB_DSP_ERR_VTCM_TOO_SMALL;
    }

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr, 0);
    HAP_compute_res_attr_set_cache_mode(&attr, 1);
    /* min_page_size = 0: best-fit page layout (fewest page mappings). The SDK
     * only accepts specific page-size values here (4 KB..16 MB); the queried
     * size is not guaranteed to be one of them, so passing it as the page size
     * (as an earlier draft of this file did) risks the manager rejecting a
     * legitimate request.
     *
     * min_vtcm_size = vtcm_avail, AND THIS IS A BUG FIX, NOT A TUNING CHOICE.
     * It was 0, and HAP_compute_res.h:544-546 defines 0 as "the size is an
     * absolute requirement" -- so this asked for the part's ENTIRE VTCM and
     * refused anything less. On a shared CDSP that means one other client
     * holding a single 4 KB page makes HAP_compute_res_acquire below return 0
     * after burning its full one-second timeout, hexlib_iface_start fails, and
     * every mode exits at session open. The SIMULATOR CANNOT SHOW THIS,
     * because nothing else there holds VTCM -- which is exactly why stage 1
     * was green with this live.
     *
     * The floor is the SDK's own `avail`, not a constant, which keeps this
     * file's governing rule intact (the size comes from the runtime, never a
     * hardcoded byte count): ask for the whole partition, accept down to what
     * the manager just said is actually free. Asking for `avail` directly
     * instead would cap us at a value that can go stale between query and
     * acquire, and would give up headroom that may have been freed in between.
     *
     * WHAT THIS DELIBERATELY DOES NOT DO: check the result against the plan's
     * high water. The DSP does not know the plan's high water -- see §8 of the
     * design doc, which used to claim this check existed. So a session can now
     * start with less VTCM than a given plan needs, and the honest division of
     * labour is that `hwinfo` reports the acquired size, the host records it,
     * and M2 compares. The exposure today is nil in practice: `hexlib_args`
     * carries vtcm/vtcm_size to every kernel, but no kernel on this branch
     * uses either. Revisit the moment one does. */
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_total, 0, vtcm_avail);
    HAP_compute_res_attr_set_release_callback(&attr, release_callback, (void *) ctx);
    /* CONDITIONAL ON THE SESSION ACTUALLY ASKING FOR HMX. `ctx->n_hmx` is set
     * by hexlib_iface_start() (skel.c) before this function ever runs; no
     * kernel on this branch requests HMX, so hexlib_open (session.c) always
     * passes n_hmx = 0. Requesting HMX unconditionally here, regardless of
     * that, risked the CDSP refusing the whole compute-res reservation for an
     * HMX-availability reason that HAP_compute_res_acquire's single status
     * code cannot distinguish from a VTCM-size failure -- the operator would
     * see a VTCM error for what was actually an HMX one. REVISIT THIS when an
     * HMX kernel first lands: this parameter is not requested at all today. */
    if (ctx->n_hmx > 0) {
        HAP_compute_res_attr_set_hmx_param(&attr, 1);
    }

    uint32_t rctx = HAP_compute_res_acquire(&attr, 1000000);
    if (!rctx) {
        /* Both numbers, so a device log distinguishes "the partition is busy"
         * from "the manager refused a request it should have satisfied". */
        FARF(ERROR, "hexlib: HAP_compute_res_acquire failed -- wanted %u, "
                    "floor %u, total %u", vtcm_total, vtcm_avail, vtcm_total);
        return HEXLIB_DSP_ERR_VTCM_TOO_SMALL;
    }

    void *ptr = 0;
    unsigned int got = 0;
    if (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &ptr, &got) != 0 || !ptr) {
        HAP_compute_res_release(rctx);
        FARF(ERROR, "hexlib: could not get VTCM pointer");
        return HEXLIB_DSP_ERR_VTCM_TOO_SMALL;
    }

    ctx->vtcm_base          = (uint8_t *) ptr;
    ctx->vtcm_size          = got;
    ctx->vtcm_rctx          = rctx;
    ctx->vtcm_valid         = 0;
    ctx->vtcm_needs_release = 0;

    /* THREE NUMBERS, ON PURPOSE. `got` alone cannot tell the operator whether a
     * short session is contention or a manager quirk; got-vs-total-vs-floor
     * can, and this is the only place any of it is observable on a device. */
    FARF(HIGH, "hexlib: VTCM %u bytes at %p (total %u, available %u)",
         got, ptr, vtcm_total, vtcm_avail);
    if (got < vtcm_total) {
        FARF(HIGH, "hexlib: VTCM is CONTENDED -- got %u of %u bytes. The session "
                   "is usable; whether it is large enough for a given plan is "
                   "not checked here (see design doc SS8)", got, vtcm_total);
    }
    return HEXLIB_DSP_OK;
}

int hexlib_vtcm_acquire(struct hexlib_ctx *ctx) {
    if (ctx->vtcm_valid) {
        return HEXLIB_DSP_OK;
    }
    if (HAP_compute_res_acquire_cached(ctx->vtcm_rctx, 1000000u) != 0) {
        FARF(ERROR, "hexlib: failed to acquire cached VTCM");
        return HEXLIB_DSP_ERR_VTCM_TOO_SMALL;
    }
    ctx->vtcm_needs_release = 0;
    ctx->vtcm_valid         = 1;
    /* Drop priority to the QuRT default. In QuRT, 1 is the highest thread
     * priority and 254 the lowest of the user-assignable range (0 and 255 are
     * reserved for the kernel; see qurt_thread.h), so this is the lowest
     * priority we can hold -- a competing session at any elevated priority
     * will reach us through the release callback instead of silently winning
     * arbitration. There is no compute-res-specific "default priority"
     * constant in the SDK; QURT_THREAD_ATTR_PRIORITY_DEFAULT is the nearest
     * verified named constant (HAP_compute_res_update_priority's own doc
     * states its priority argument is "in terms of QuRT thread priority"), so
     * it is used here instead of a guessed magic number. */
    HAP_compute_res_update_priority(ctx->vtcm_rctx, QURT_THREAD_ATTR_PRIORITY_DEFAULT);
    return HEXLIB_DSP_OK;
}

void hexlib_vtcm_release(struct hexlib_ctx *ctx) {
    if (ctx->vtcm_valid) {
        ctx->vtcm_valid         = 0;
        ctx->vtcm_needs_release = 0;
        HAP_compute_res_release_cached(ctx->vtcm_rctx);
    }
}

void hexlib_vtcm_free(struct hexlib_ctx *ctx) {
    if (ctx->vtcm_rctx) {
        HAP_compute_res_release(ctx->vtcm_rctx);
        ctx->vtcm_rctx = 0;
        ctx->vtcm_base = 0;
        ctx->vtcm_size = 0;
    }
}

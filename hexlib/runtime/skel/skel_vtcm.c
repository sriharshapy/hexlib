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
    unsigned int vtcm_size = 0;
    if (HAP_compute_res_query_VTCM(0, &vtcm_size, 0, 0, 0) != 0 || vtcm_size == 0) {
        FARF(ERROR, "hexlib: HAP_compute_res_query_VTCM failed");
        return HEXLIB_DSP_ERR_INTERNAL;
    }

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr, 0);
    HAP_compute_res_attr_set_cache_mode(&attr, 1);
    /* min_page_size = 0: best-fit page layout (fewest page mappings). The SDK
     * only accepts specific page-size values here (4 KB..16 MB); the queried
     * vtcm_size is not guaranteed to be one of them, so passing vtcm_size
     * itself (as an earlier draft of this file did) risks the manager
     * rejecting a legitimate request.
     * min_vtcm_size = 0: the queried size is an absolute requirement -- if it
     * is not available we fail rather than silently accepting less. */
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_size, 0, 0);
    HAP_compute_res_attr_set_release_callback(&attr, release_callback, (void *) ctx);
    HAP_compute_res_attr_set_hmx_param(&attr, 1);

    uint32_t rctx = HAP_compute_res_acquire(&attr, 1000000);
    if (!rctx) {
        FARF(ERROR, "hexlib: HAP_compute_res_acquire failed for %u bytes", vtcm_size);
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

    FARF(HIGH, "hexlib: VTCM %u bytes at %p", got, ptr);
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

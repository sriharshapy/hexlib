/* hexlib/runtime/skel/skel_bufs.c -- fd to mapped address, and nothing else.
 *
 * THE INVARIANT THIS FILE EXISTS TO HOLD: the host writes fds and offsets, this
 * side writes addresses. `hexlib_buf_desc.base` arrives as zero and is
 * OVERWRITTEN before it is read, from a table only hexlib_bufs_register
 * populates. So a host address cannot reach a kernel even if one were somehow
 * placed on the wire.
 *
 * WHY THAT MATTERS MORE THAN IT LOOKS. Under the simulator qexe the host side and
 * this side are one process in one address space. A skel that used the host's
 * pointer would work perfectly on the simulator and fail instantly on silicon --
 * a gate certifying nothing. The unmapped-fd test is what catches it: on the
 * simulator, an implementation leaning on the shared address space returns the
 * RIGHT ANSWER to a request whose buffer was never mapped, so the test fails
 * exactly when the invariant is broken.
 *
 * Adapted from llama.cpp ggml-hexagon htp/main.c reuse_buf/mmap_buf/prep_tensor
 * (MIT); see ATTRIBUTION.md. Upstream's silent `base == 0` fallthrough and its
 * hard process abort on a failed mapping are both replaced by error returns.
 */
#include "skel_internal.h"

#include <string.h>

#include "HAP_farf.h"
#include "HAP_mem.h"

/* Defined at the bottom of this file, after every caller. The lookup-by-fd
 * comparison it contains is what `test_base_is_cleared_before_any_lookup`
 * checks the position of relative to `hexlib_bufs_map`'s clearing of `base` --
 * keeping the definition below the callers keeps that ordering honest instead
 * of coincidental. */
static struct hexlib_mmap *find_by_fd(struct hexlib_ctx *ctx, uint32_t fd);

int hexlib_bufs_register(struct hexlib_ctx *ctx, uint32_t fd, uint32_t size) {
    if (find_by_fd(ctx, fd)) {
        return HEXLIB_DSP_OK;  /* already mapped; idempotent */
    }
    for (uint32_t i = 0; i < HEXLIB_MAX_MMAPS; i++) {
        struct hexlib_mmap *m = &ctx->mmap[i];
        if (m->size) {
            continue;
        }
        /* HAP_mmap's `len` is `int`; HAP_mmap2's is `size_t`. Cast explicitly per
         * branch rather than relying on the implicit uint32_t->int narrowing the
         * older API's signature otherwise forces on us. See HAP_mem.h. */
#if __HVX_ARCH__ > 73
        void *va = HAP_mmap2(0, (size_t) size, HAP_PROT_READ | HAP_PROT_WRITE, 0, (int) fd, 0);
#else
        void *va = HAP_mmap(0, (int) size, HAP_PROT_READ | HAP_PROT_WRITE, 0, (int) fd, 0);
#endif
        if (va == (void *) -1 || va == 0) {
            FARF(ERROR, "hexlib: mmap failed fd %u size %u", fd, size);
            return HEXLIB_DSP_ERR_MMAP_FAILED;
        }
        m->base = (uint64_t) va;
        m->size = size;
        m->fd   = (int32_t) fd;
        FARF(HIGH, "hexlib: mmap fd %u base %p size %u", fd, va, size);
        return HEXLIB_DSP_OK;
    }
    /* Upstream returns silently here and lets the caller compute 0 + offset.
     * That is a write to a low address, not a diagnosable failure. */
    FARF(ERROR, "hexlib: no free mmap slot for fd %u (max %u)", fd, HEXLIB_MAX_MMAPS);
    return HEXLIB_DSP_ERR_NO_MMAP_SLOT;
}

int hexlib_bufs_unregister(struct hexlib_ctx *ctx, uint32_t fd) {
    struct hexlib_mmap *m = find_by_fd(ctx, fd);
    if (!m) {
        return HEXLIB_DSP_ERR_UNMAPPED;
    }
#if __HVX_ARCH__ > 73
    HAP_munmap2((void *) m->base, (size_t) m->size);
#else
    HAP_munmap((void *) m->base, (int) m->size);
#endif
    m->base = 0;
    m->size = 0;
    m->fd   = -1;
    return HEXLIB_DSP_OK;
}

int hexlib_bufs_map(struct hexlib_ctx *ctx, struct hexlib_buf_desc *bufs, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        struct hexlib_buf_desc *b = bufs + i;

        /* CLEAR FIRST. Whatever the host put here is destroyed before it can be
         * read. This single line is the invariant. */
        b->base = 0;

        struct hexlib_mmap *m = find_by_fd(ctx, b->fd);
        if (!m) {
            /* NOT mapped on demand. The host must have called mmap(). A buffer
             * appearing for the first time inside invoke() is a host bug, and
             * mapping it here would hide it -- and on the simulator, would let
             * the shared address space paper over the invariant entirely. */
            FARF(ERROR, "hexlib: buffer %u fd %u was never mapped", i, b->fd);
            return HEXLIB_DSP_ERR_UNMAPPED;
        }
        if (b->size > m->size) {
            FARF(ERROR, "hexlib: buffer %u claims %u bytes, mapping has %u",
                 i, (uint32_t) b->size, (uint32_t) m->size);
            return HEXLIB_DSP_ERR_INVAL_PARAMS;
        }
        b->base = m->base;
    }
    return HEXLIB_DSP_OK;
}

int hexlib_tensors_resolve(struct hexlib_ctx *ctx, struct hexlib_buf_desc *bufs,
                           uint32_t n_bufs, struct hexlib_tensor *tens,
                           uint32_t n_tens) {
    (void) ctx;
    for (uint32_t i = 0; i < n_tens; i++) {
        struct hexlib_tensor *t = tens + i;
        if (t->bi >= n_bufs) {
            FARF(ERROR, "hexlib: tensor %u names buffer %u of %u", i, t->bi, n_bufs);
            return HEXLIB_DSP_ERR_INVAL_PARAMS;
        }
        struct hexlib_buf_desc *b = bufs + t->bi;
        if (!b->base) {
            return HEXLIB_DSP_ERR_UNMAPPED;
        }
        /* Bounds-checked HERE as well as on the host. The host is the thing
         * being served, not the thing being trusted. */
        if ((uint64_t) t->offset + (uint64_t) t->nbytes > b->size) {
            FARF(ERROR, "hexlib: tensor %u offset %u + %u exceeds buffer %u size %u",
                 i, t->offset, t->nbytes, t->bi, (uint32_t) b->size);
            return HEXLIB_DSP_ERR_TRUNCATED;
        }
        t->data = (uint32_t) (b->base + t->offset);
    }
    return HEXLIB_DSP_OK;
}

/* THE LOOKUP THE WHOLE FILE EXISTS TO GATE. Matches by fd, never by whatever
 * `base` the host sent -- callers above have already cleared it before
 * reaching here. Occupied slots have a nonzero size; fd alone is not enough,
 * since an unregistered slot's fd field is reset to -1, not left stale. */
static struct hexlib_mmap *find_by_fd(struct hexlib_ctx *ctx, uint32_t fd) {
    for (uint32_t i = 0; i < HEXLIB_MAX_MMAPS; i++) {
        struct hexlib_mmap *m = &ctx->mmap[i];
        if (m->size && m->fd == (int32_t) fd) {
            return m;
        }
    }
    return 0;
}

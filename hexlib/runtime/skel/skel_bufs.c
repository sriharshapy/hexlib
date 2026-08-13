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

#include "HAP_farf.h"
#include "HAP_mem.h"

/* QuRT ONLY, AND THE GUARD IS NARROW ON PURPOSE. `qurt_memory.h` exists in the
 * Hexagon SDK's RTOS tree and nowhere else, and `hexlib/tests/
 * test_genentry_entry_probe.py` compiles this file with the HOST compiler to
 * drive the dispatcher behaviourally. Both the device skel and the QuRT-hosted
 * simulator .so define `__hexagon__`, so the real calls are taken everywhere
 * they can possibly matter; the host fallback exists solely so that probe can
 * link, and it is NOT a portability layer.
 *
 * The danger of a fallback like this is that it silently disables the very
 * thing it stands in for. That is bound to the artifact rather than trusted:
 * `test_device_cache_maintenance.py` asserts `qurt_mem_cache_clean` is an
 * UNDEFINED symbol in the linked libhexlib_iface_skel.so, which is false the
 * moment this stub is what got compiled. */
#if defined(__hexagon__)
#include "qurt_memory.h"
#else
typedef unsigned long qurt_addr_t;
typedef unsigned long qurt_size_t;
typedef int qurt_mem_cache_op_t;
typedef int qurt_mem_cache_type_t;
#define QURT_MEM_CACHE_FLUSH       0
#define QURT_MEM_CACHE_INVALIDATE  1
#define QURT_MEM_DCACHE            0
static int qurt_mem_cache_clean(qurt_addr_t a, qurt_size_t n,
                                qurt_mem_cache_op_t op, qurt_mem_cache_type_t t) {
    (void) a; (void) n; (void) op; (void) t;
    return 0;   /* host probe only -- see the comment above */
}
#endif

/* THE LOOKUP THE WHOLE FILE EXISTS TO GATE. Matches by fd, never by whatever
 * `base` the host sent. Occupied slots have a nonzero size; fd alone is not
 * enough, since an unregistered slot's fd field is reset to -1, not left
 * stale. */
static struct hexlib_mmap *find_by_fd(struct hexlib_ctx *ctx, uint32_t fd) {
    for (uint32_t i = 0; i < HEXLIB_MAX_MMAPS; i++) {
        struct hexlib_mmap *m = &ctx->mmap[i];
        if (m->size && m->fd == (int32_t) fd) {
            return m;
        }
    }
    return 0;
}

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
    uint64_t base = m->base;
    uint64_t size = m->size;
    /* Free the slot regardless of the unmap outcome below: whatever happens at
     * the OS level, this fd must stop being something hexlib_bufs_map() can
     * hand back on a future lookup. */
    m->base = 0;
    m->size = 0;
    m->fd   = -1;
#if __HVX_ARCH__ > 73
    int rc = HAP_munmap2((void *) base, (size_t) size);
#else
    int rc = HAP_munmap((void *) base, (int) size);
#endif
    if (rc != 0) {
        /* No dedicated "unmap failed" status exists on the wire (see
         * hexlib_dsp.h, not modified by this file); MMAP_FAILED is the closest
         * available fit for "a HAP_mem mapping call did not do what we asked".
         * Checked rather than ignored: a failed unmap does not threaten the
         * pointer invariant (the slot above is already cleared either way),
         * but silently discarding an OS-level failure here is exactly the
         * kind of thing this file exists to stop doing. */
        FARF(ERROR, "hexlib: munmap failed for fd %u base %p size %u rc %d",
             fd, (void *) base, (uint32_t) size, rc);
        return HEXLIB_DSP_ERR_MMAP_FAILED;
    }
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

/* ---------------------------------------------------------------------------
 * CACHE MAINTENANCE, WHICH FASTRPC DOES NOT DO FOR THESE BUFFERS.
 *
 * FastRPC keeps the two caches coherent for anything passed AS AN INVOKE
 * ARGUMENT -- it knows the direction of each `rin`/`rout` parameter and cleans
 * or invalidates accordingly. Our data buffers are not arguments. They are
 * mapped once, out of band, through `fastrpc_mmap` and named on the wire only
 * by fd, precisely so no address ever crosses between the two processors. That
 * design decision stands. Its consequence is that FastRPC has no idea these
 * pages were touched, so NOTHING happens unless we do it here.
 *
 * WHAT THAT COST, MEASURED ON SM8650 BEFORE THIS EXISTED. All three of these
 * are the same bug wearing different clothes:
 *
 *   * `hexlib_run --self-test`: 3859 of 4100 fp16 values not bit-exact, where
 *     the simulator gives exactly 0 error. Not garbage -- a PARTIAL write. The
 *     241 correct values were the lines that happened to get evicted.
 *   * `--coherency-check`: the sentinel came back unchanged, exit 6. The write
 *     never reached DDR at all.
 *   * The whole 49-op encoder through `--batch`: 179,124 bytes of the arena
 *     changed, but `merger.out` -- written by the LAST op, 1024 bytes -- came
 *     back all zero. That one is decisive because it is ORDERED. Early writes
 *     landed because subsequent work evicted them; the final write was still
 *     sitting in the cache when the invoke returned.
 *
 * BOTH DIRECTIONS ARE NEEDED AND THEY ARE NOT THE SAME OPERATION.
 * `invalidate` before the batch: the host has just written weights and inputs
 * into these pages, and any line this DSP still holds from a PREVIOUS invoke is
 * stale. `flush` after: our writes must reach memory before the host reads
 * them. Doing only the flush works for exactly one invoke per session and then
 * silently reads old data, which is a worse bug than the one being fixed
 * because it needs two runs to show up.
 *
 * WHOLE BUFFERS, NOT WRITTEN RANGES. The op descriptors say which tensors an op
 * writes, so a narrower flush is computable -- and it would be a per-op loop
 * over sub-ranges, each rounded out to a cache line, with the line-straddling
 * case at every boundary. `qurt_memory.h` warns that the operation takes the
 * whole line and therefore "the contents of the adjoining buffer can be flushed
 * and invalidated if it falls in any of the cache line", so partial ranges make
 * neighbours another correctness question. Once per batch over whole buffers
 * has no such case. If this ever shows up in a profile, narrow it THEN, with a
 * measurement to point at.
 */
static int cache_op_all(struct hexlib_ctx *ctx, qurt_mem_cache_op_t op,
                        const char *what) {
    int worst = HEXLIB_DSP_OK;
    for (uint32_t i = 0; i < HEXLIB_MAX_MMAPS; i++) {
        struct hexlib_mmap *m = &ctx->mmap[i];
        if (!m->size) {
            continue;
        }
        int rc = qurt_mem_cache_clean((qurt_addr_t) m->base,
                                      (qurt_size_t) m->size,
                                      op, QURT_MEM_DCACHE);
        if (rc != 0) {
            /* LOUD, AND IT CHANGES THE BATCH STATUS. A cache operation that
             * quietly failed would put us back in exactly the state this code
             * exists to leave: a run that reports OK and returns data the host
             * cannot see. */
            FARF(ERROR, "hexlib: %s failed for fd %d base %p size %u rc %d",
                 what, (int) m->fd, (void *) (uintptr_t) m->base,
                 (uint32_t) m->size, rc);
            worst = HEXLIB_DSP_ERR_CACHE;
        }
    }
    return worst;
}

int hexlib_bufs_invalidate(struct hexlib_ctx *ctx) {
    return cache_op_all(ctx, QURT_MEM_CACHE_INVALIDATE, "cache invalidate");
}

int hexlib_bufs_flush(struct hexlib_ctx *ctx) {
    return cache_op_all(ctx, QURT_MEM_CACHE_FLUSH, "cache flush");
}

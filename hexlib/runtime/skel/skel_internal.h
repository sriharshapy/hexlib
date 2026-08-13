/* hexlib/runtime/skel/skel_internal.h -- skel-private state. */
#ifndef HEXLIB_SKEL_INTERNAL_H
#define HEXLIB_SKEL_INTERNAL_H

#include "hexlib_dsp.h"

#define HEXLIB_MAX_MMAPS 32

struct hexlib_mmap {
    uint64_t base;
    uint64_t size;
    int32_t  fd;
};

struct hexlib_ctx {
    struct hexlib_mmap mmap[HEXLIB_MAX_MMAPS];
    uint64_t max_vmem;

    /* The last hexlib_iface_start() status. Recorded because start()'s
     * AEEResult may be normalised by the RPC layer, and because a later call
     * that finds !started can then say WHY rather than only that it must not
     * proceed. HEXLIB_DSP_OK once a session is up. */
    int start_status;

    uint8_t *vtcm_base;
    size_t   vtcm_size;
    uint32_t vtcm_rctx;
    int      vtcm_valid;

    /* VOLATILE BECAUSE IT IS WRITTEN BY A DIFFERENT THREAD. `release_callback`
     * in skel_vtcm.c sets this to 1 from HAP_compute_res's own QuRT thread,
     * and hexlib_dispatch_batch's per-op loop reads it. It was a plain `int`.
     * That worked only by accident: the opaque `k->fn(&a)` call in the loop
     * body is a call through a function pointer the compiler cannot see into,
     * so it must assume the callee may have written any escaped object and
     * reloads this from memory each iteration. An inlined or
     * constant-propagated kernel removes that barrier, and the compiler is
     * then entitled to hoist the load out of the loop entirely -- at which
     * point a reclaim request arriving mid-batch is never noticed, the
     * competing session waits on VTCM this one will not give back, and
     * nothing about the source looks different.
     *
     * `volatile`, not an atomic: this is a one-way 0 -> 1 flag with a single
     * writer and a single reader, and the only requirement is that the reader
     * actually re-reads memory. There is no read-modify-write to make atomic
     * and no other object whose ordering relative to this one matters -- the
     * release itself happens on the dispatch thread, after the flag is
     * observed (skel_dispatch.c), never in the callback. */
    volatile int vtcm_needs_release;

    uint32_t sess_id;
    uint32_t n_hvx;
    uint32_t n_hmx;
    int      started;
};

int hexlib_bufs_register(struct hexlib_ctx *ctx, uint32_t fd, uint32_t size);
int hexlib_bufs_unregister(struct hexlib_ctx *ctx, uint32_t fd);
int hexlib_bufs_map(struct hexlib_ctx *ctx, struct hexlib_buf_desc *bufs, uint32_t n);
/* Cache maintenance over every mapped buffer. See skel_bufs.c's block comment
 * for why FastRPC does not do this for us and what its absence measured. */
int hexlib_bufs_invalidate(struct hexlib_ctx *ctx);
int hexlib_bufs_flush(struct hexlib_ctx *ctx);
int hexlib_tensors_resolve(struct hexlib_ctx *ctx, struct hexlib_buf_desc *bufs,
                           uint32_t n_bufs, struct hexlib_tensor *tens, uint32_t n_tens);

int hexlib_vtcm_alloc(struct hexlib_ctx *ctx);
void hexlib_vtcm_free(struct hexlib_ctx *ctx);
int hexlib_vtcm_acquire(struct hexlib_ctx *ctx);
void hexlib_vtcm_release(struct hexlib_ctx *ctx);

int hexlib_dispatch_batch(struct hexlib_ctx *ctx, const uint8_t *batch, uint32_t len,
                          uint8_t *rsp, uint32_t rsp_cap, uint32_t *rsp_len);

/* Defined in skel_dispatch.c, shared with skel.c: both hexlib_dispatch_batch's
 * own failure returns AND hexlib_iface_invoke's invoke-before-start refusal
 * must write the exact same response header shape, with the arch this binary
 * was built for -- never a value either caller passes in. */
void hexlib_write_rsp_hdr(uint8_t *rsp, uint32_t status, uint32_t n_ops, uint64_t cycles);

#endif /* HEXLIB_SKEL_INTERNAL_H */

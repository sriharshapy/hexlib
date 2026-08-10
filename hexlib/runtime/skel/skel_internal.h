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

    uint8_t *vtcm_base;
    size_t   vtcm_size;
    uint32_t vtcm_rctx;
    int      vtcm_valid;
    int      vtcm_needs_release;

    uint32_t sess_id;
    uint32_t n_hvx;
    uint32_t n_hmx;
    int      started;
};

int hexlib_bufs_register(struct hexlib_ctx *ctx, uint32_t fd, uint32_t size);
int hexlib_bufs_unregister(struct hexlib_ctx *ctx, uint32_t fd);
int hexlib_bufs_map(struct hexlib_ctx *ctx, struct hexlib_buf_desc *bufs, uint32_t n);
int hexlib_tensors_resolve(struct hexlib_ctx *ctx, struct hexlib_buf_desc *bufs,
                           uint32_t n_bufs, struct hexlib_tensor *tens, uint32_t n_tens);

int hexlib_vtcm_alloc(struct hexlib_ctx *ctx);
void hexlib_vtcm_free(struct hexlib_ctx *ctx);
int hexlib_vtcm_acquire(struct hexlib_ctx *ctx);
void hexlib_vtcm_release(struct hexlib_ctx *ctx);

int hexlib_dispatch_batch(struct hexlib_ctx *ctx, const uint8_t *batch, uint32_t len,
                          uint8_t *rsp, uint32_t rsp_cap, uint32_t *rsp_len);

#endif /* HEXLIB_SKEL_INTERNAL_H */

/* hexlib/runtime/host/buffers.c -- rpcmem allocation, fastrpc_mmap, and the
 * DSP-side registration handshake.
 *
 * THE HOST NEVER PUTS AN ADDRESS ON THE WIRE. `hexlib_buf_desc.base` (see
 * hexlib_dsp.h) is DSP-side scratch: the skel resolves its own address for a
 * registered fd via HAP_mmap (skel_bufs.c) when a batch actually references
 * it. There is no field on the wire for a host address at all, so
 * hexlib_buf_to_desc writes `base = 0` explicitly, at the one place
 * a hexlib_buf_desc is ever filled in from this side -- the same invariant
 * hexlib.runtime.wire.py enforces on the Python side and hexlib_dsp.h states
 * for the DSP side.
 */
#include "hexlib_host.h"

#include <limits.h>         /* INT_MAX -- see hexlib_alloc's size guard. */
#include <remote.h>
#include <rpcmem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hexlib_dsp.h"     /* struct hexlib_buf_desc */
#include "hexlib_iface.h"   /* hexlib_iface_mmap / hexlib_iface_munmap */

int hexlib_alloc(hexlib_ctx *ctx, hexlib_buf **out, size_t size) {
    *out = NULL;

    /* SIZE IS NARROWED TWICE BELOW, AND NEITHER CAST CAN REPORT A LOSS.
     * `rpcmem_alloc` takes an `int` and `hexlib_iface_mmap` takes a `uint32`
     * (that is qaic's own generated signature, from the IDL -- not something
     * this file chose), so a `size_t` of 2 GiB or more truncates on the way to
     * one or both: `(int) size` can even go NEGATIVE, and the two casts can
     * disagree, which would register a mapping of one length for a buffer
     * allocated at another.
     *
     * UNREACHABLE TODAY, AND GUARDED ANYWAY. Every call site passes a plan-
     * computed tensor size; the largest thing the encoder moves is far below
     * 2 GiB, and if a truncated pair ever did get through, the DSP side fails
     * CLOSED rather than reading out of bounds (skel_bufs.c's `b->size >
     * m->size` check answers HEXLIB_DSP_ERR_INVAL_PARAMS). So this is a guard
     * and a comment, deliberately NOT a widening of the IDL or of
     * hexlib_alloc's own signature -- changing the wire for a case that
     * cannot arise would be the larger risk.
     *
     * INT_MAX, not UINT32_MAX: the narrower of the two casts is the binding
     * one, and this must fail before either happens rather than after one of
     * them has already silently succeeded.
     *
     * NOTE THE DUPLICATE: main.c's `alloc_maybe_unmapped` is a deliberate
     * local copy of this allocation sequence (for --unmapped) and carries the
     * identical pair of narrowings. Its only call site passes a fixed 8200
     * bytes, so it is not reachable there either. */
    if (size == 0 || size > (size_t) INT_MAX) {
        fprintf(stderr,
                "hexlib: hexlib_alloc: %zu bytes is out of range -- rpcmem_alloc "
                "takes an int and hexlib_iface_mmap takes a uint32, so a size "
                "at or above 2 GiB would be truncated by one or both with no "
                "way to report it\n", size);
        return -1;
    }

    void *ptr = hexlib_rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS,
                                    (int) size);
    if (ptr == NULL) {
        fprintf(stderr, "hexlib: rpcmem_alloc(%zu bytes) failed\n", size);
        return -1;
    }

    int fd = hexlib_rpcmem_to_fd(ptr);
    if (fd < 0) {
        fprintf(stderr, "hexlib: rpcmem_to_fd failed\n");
        hexlib_rpcmem_free(ptr);
        return -1;
    }

    /* Maps the buffer into the CDSP's address space on the CPU-driver side.
     * `addr` is the CPU virtual address purely so the driver can track which
     * local mapping this fd corresponds to for cache maintenance -- it is
     * NEVER the address the DSP will use, and never crosses the wire. */
    int rc = hexlib_fastrpc_mmap(ctx->domain, fd, ptr, 0, size, FASTRPC_MAP_FD);
    if (rc != 0) {
        fprintf(stderr,
                "hexlib: fastrpc_mmap(fd=%d, size=%zu) failed (rc %d)\n",
                fd, size, rc);
        hexlib_rpcmem_free(ptr);
        return -1;
    }

    /* Registers the SAME fd with the DSP-side skel (hexlib_iface_mmap ->
     * hexlib_bufs_register -> HAP_mmap in skel_bufs.c). This is a second,
     * independent map of one fd: the fastrpc_mmap above lets the CPU driver
     * account for the buffer, this one is what the skel's buffer table looks
     * up by fd at invoke time. Neither one hands the other side an address. */
    int arc = hexlib_iface_mmap(ctx->handle, (uint32_t) fd, (uint32_t) size);
    if (arc != AEE_SUCCESS) {
        fprintf(stderr, "hexlib: hexlib_iface_mmap(fd=%d) failed (rc %d)\n", fd, arc);
        hexlib_fastrpc_munmap(ctx->domain, fd, ptr, size);
        hexlib_rpcmem_free(ptr);
        return -1;
    }

    hexlib_buf *buf = (hexlib_buf *) calloc(1, sizeof(*buf));
    if (buf == NULL) {
        hexlib_iface_munmap(ctx->handle, (uint32_t) fd);
        hexlib_fastrpc_munmap(ctx->domain, fd, ptr, size);
        hexlib_rpcmem_free(ptr);
        return -1;
    }
    buf->ptr  = ptr;
    buf->fd   = fd;
    buf->size = size;
    *out = buf;
    return 0;
}

void hexlib_free(hexlib_ctx *ctx, hexlib_buf *buf) {
    if (buf == NULL) {
        return;
    }
    hexlib_iface_munmap(ctx->handle, (uint32_t) buf->fd);
    hexlib_fastrpc_munmap(ctx->domain, buf->fd, buf->ptr, buf->size);
    hexlib_rpcmem_free(buf->ptr);
    free(buf);
}

/* THE HOST NEVER PUTS AN ADDRESS ON THE WIRE -- see the file header. `d->base`
 * is set to 0 unconditionally, first, before any other field: there is no
 * value derived from `buf->ptr` (the CPU-side virtual address) that could
 * ever legally end up here. */
void hexlib_buf_to_desc(const hexlib_buf *buf, struct hexlib_buf_desc *d) {
    memset(d, 0, sizeof(*d));
    d->base  = 0;                    /* DSP-side scratch. Never a host address. */
    d->size  = (uint64_t) buf->size;
    d->fd    = (uint32_t) buf->fd;
    d->flags = 0;
}

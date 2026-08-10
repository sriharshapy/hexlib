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

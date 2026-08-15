/* hexlib/runtime/host/hexlib_host.h -- the CPU-side (aarch64, Android) API.
 *
 * This is the ONLY path that ever exercises qaic's real argument marshalling
 * (see main.c): on the simulator (Task 7/8) the qaic stub was deliberately
 * never linked, because skel.c's function names collide with it in one
 * address space. On a device there are two separate binaries -- this one
 * links the qaic-generated STUB (hexlib_iface_stub.c) and calls
 * hexlib_iface_open/_start/_mmap/_munmap/_hwinfo/_invoke/_stop/_close exactly
 * like any other function; the marshalling into remote_handle64_invoke()
 * happens inside those generated wrappers, invisibly to this file.
 *
 * THE SDK IS NEVER VENDORED. <remote.h> and the rest of the Hexagon SDK are
 * found via HEXAGON_SDK_ROOT at build time (Task 10), never copied into this
 * repository.
 */
#ifndef HEXLIB_HOST_H
#define HEXLIB_HOST_H

#include <stddef.h>
#include <stdint.h>

#include <remote.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * driver.c -- dlopen'd libcdsprpc.so, resolved by symbol name.
 *
 * NONE of these are linked at build time: libcdsprpc.so exists only on a
 * device with the FastRPC driver installed, and even then only for the
 * domains that device supports. hexlib_drv_init() must be called (and must
 * return 0) before any of the function pointers below are valid.
 * ========================================================================*/

typedef void *(*hexlib_rpcmem_alloc_fn)(int heapid, uint32_t flags, int size);
typedef void *(*hexlib_rpcmem_alloc2_fn)(int heapid, uint32_t flags, size_t size);
typedef void  (*hexlib_rpcmem_free_fn)(void *po);
typedef int   (*hexlib_rpcmem_to_fd_fn)(void *po);
typedef int   (*hexlib_fastrpc_mmap_fn)(int domain, int fd, void *addr, int offset,
                                        size_t length, enum fastrpc_map_flags flags);
typedef int   (*hexlib_fastrpc_munmap_fn)(int domain, int fd, void *addr, size_t length);
typedef int   (*hexlib_remote_handle64_open_fn)(const char *name, remote_handle64 *ph);
typedef int   (*hexlib_remote_handle64_invoke_fn)(remote_handle64 h, uint32_t dwScalars,
                                                  remote_arg *pra);
typedef int   (*hexlib_remote_handle64_close_fn)(remote_handle64 h);
typedef int   (*hexlib_remote_handle_control_fn)(uint32_t req, void *data, uint32_t datalen);
typedef int   (*hexlib_remote_session_control_fn)(uint32_t req, void *data, uint32_t datalen);

extern hexlib_rpcmem_alloc_fn             hexlib_rpcmem_alloc;
extern hexlib_rpcmem_alloc2_fn            hexlib_rpcmem_alloc2;   /* may stay NULL */
extern hexlib_rpcmem_free_fn              hexlib_rpcmem_free;
extern hexlib_rpcmem_to_fd_fn             hexlib_rpcmem_to_fd;
extern hexlib_fastrpc_mmap_fn             hexlib_fastrpc_mmap;
extern hexlib_fastrpc_munmap_fn           hexlib_fastrpc_munmap;
extern hexlib_remote_handle64_open_fn     hexlib_remote_handle64_open;
extern hexlib_remote_handle64_invoke_fn   hexlib_remote_handle64_invoke;
extern hexlib_remote_handle64_close_fn    hexlib_remote_handle64_close;
extern hexlib_remote_handle_control_fn    hexlib_remote_handle_control;
extern hexlib_remote_session_control_fn   hexlib_remote_session_control;

/* Loads libcdsprpc.so and resolves every symbol above by name. Returns 0 on
 * success. Idempotent -- a second call is a no-op that also returns 0. A
 * missing REQUIRED symbol fails the whole call (named in stderr, via
 * dlerror()) rather than leaving some pointers NULL for a later call site to
 * crash on. */
int hexlib_drv_init(void);

/* ==========================================================================
 * session.c -- CDSP only, unsigned PD, arch cross-checked against the skel.
 * ========================================================================*/

typedef struct hexlib_ctx {
    int             domain;      /* Always CDSP_DOMAIN_ID; see hexlib_open(). */
    remote_handle64 handle;
    uint32_t        arch;
    uint32_t        n_threads;
    uint32_t        n_hvx;
    uint32_t        n_hmx;
    uint64_t        vtcm_size;   /* ACQUIRED size, not the part's total. */
    int             started;
} hexlib_ctx;

/* One DSPRPC_GET_DSP_INFO query per field, each attribute named from
 * <remote.h>'s own `enum remote_dsp_attributes` -- see session.c. */
struct hexlib_caps {
    uint32_t domain_support;
    uint32_t unsigned_pd_support;
    uint32_t hvx_support_128b;
    uint32_t vtcm_page;
    uint32_t vtcm_count;
    uint32_t arch_ver;
    uint32_t hmx_support_depth;  /* 0 is NOT evidence HMX is absent. */
};

/* Requires hexlib_drv_init() to have already succeeded. */
int hexlib_query_caps(int domain, struct hexlib_caps *out);

/* Opens a session on `domain`, which MUST be CDSP_DOMAIN_ID -- ADSP is a v73
 * part with a different UNSIGNED_PD_SUPPORT and must never be substituted
 * silently. Requests an unsigned PD, opens the qaic handle, starts the
 * session, reads hwinfo back, and cross-checks the arch the driver reports
 * against the arch the skel reports; disagreement is refused, not logged and
 * ignored. On success, *out is a session ready for hexlib_alloc/hexlib_invoke. */
int hexlib_open(hexlib_ctx **out, int domain);
int hexlib_close(hexlib_ctx *ctx);

int hexlib_hwinfo(hexlib_ctx *ctx, uint32_t *arch, uint32_t *n_threads,
                  uint32_t *n_hvx, uint32_t *n_hmx, uint64_t *vtcm_size);

/* Runs one batch. `batch`/`batch_len` is a wire-format blob (hexlib_dsp.h /
 * hexlib.runtime.wire.py); `rsp`/`rsp_cap` is the caller's response buffer.
 * `*rsp_len` is set to the number of bytes that are actually meaningful,
 * computed from hexlib_batch_rsp_hdr.n_ops -- qaic's `rout sequence<octet>`
 * carries no out-length of its own (see hexlib_iface.h / ATTRIBUTION.md), so
 * there is nowhere else this number could come from. Returns 0 only if the
 * RPC itself succeeded; the caller must still check the response's own magic
 * and status (see main.c) before trusting the bytes as a result. */
int hexlib_invoke(hexlib_ctx *ctx, const void *batch, size_t batch_len,
                  void *rsp, size_t rsp_cap, size_t *rsp_len);

/* ==========================================================================
 * buffers.c -- rpcmem + fastrpc_mmap. The host never puts an address on the
 * wire; see hexlib_buf_to_desc() below and hexlib_dsp.h's own comment.
 * ========================================================================*/

typedef struct hexlib_buf {
    void   *ptr;    /* CPU-side virtual address -- for THIS process only. */
    int     fd;
    size_t  size;
} hexlib_buf;

int  hexlib_alloc(hexlib_ctx *ctx, hexlib_buf **out, size_t size);
void hexlib_free(hexlib_ctx *ctx, hexlib_buf *buf);

/* Forward-declared, not included: hexlib_buf_desc lives in
 * runtime/skel/hexlib_dsp.h, a DSP-side header this one does not otherwise
 * need. Only a pointer to it crosses this interface. */
struct hexlib_buf_desc;

/* Fills `d->base = 0` always -- the DSP resolves its own address for `fd`
 * (skel_bufs.c); there is no field on the wire for a host address at all. */
void hexlib_buf_to_desc(const hexlib_buf *buf, struct hexlib_buf_desc *d);

#ifdef __cplusplus
}
#endif

#endif /* HEXLIB_HOST_H */

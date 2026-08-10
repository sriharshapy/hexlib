/* hexlib/runtime/host/driver.c -- dlopen the FastRPC driver, by symbol name.
 *
 * Adapted from llama.cpp ggml-hexagon's htp-drv.cpp `htpdrv_init()` (MIT); see
 * ATTRIBUTION.md. Rewritten in C -- upstream is C++ and carries dspqueue
 * plumbing hexlib does not use (see runtime/idl/hexlib_iface.idl for why:
 * dspqueue has no simulator path). What is kept is the shape that mattered:
 * `libcdsprpc.so` is dlopen'd, never linked, and every symbol this file needs
 * is resolved by NAME through dlsym and checked before use.
 *
 * WHY DLOPEN AND NOT A LINK-TIME DEPENDENCY. `libcdsprpc.so` exists only on a
 * device that actually has the FastRPC driver installed for the compute DSP.
 * Linking it directly turns "this device doesn't have it" into an
 * unresolved-symbol failure at process load, before main() ever runs, with no
 * message a user can act on. dlopen makes that failure a readable string
 * instead -- this is the exact property that made the same author's
 * capability probe work on the first real-silicon attempt: a missing driver
 * said so, rather than refusing to start.
 *
 * A MISSING SYMBOL IS A NAMED ERROR, NEVER A NULL CALL. Calling through a NULL
 * function pointer segfaults with no indication of which symbol was missing.
 * HEXLIB_DLSYM logs the exact symbol name and dlerror()'s text, then fails
 * hexlib_drv_init() outright for anything required -- it does not leave a
 * NULL pointer behind for some later call site to trip over.
 */
#include "hexlib_host.h"

#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>

hexlib_rpcmem_alloc_fn             hexlib_rpcmem_alloc             = NULL;
hexlib_rpcmem_alloc2_fn            hexlib_rpcmem_alloc2            = NULL;
hexlib_rpcmem_free_fn              hexlib_rpcmem_free              = NULL;
hexlib_rpcmem_to_fd_fn             hexlib_rpcmem_to_fd             = NULL;
hexlib_fastrpc_mmap_fn             hexlib_fastrpc_mmap             = NULL;
hexlib_fastrpc_munmap_fn           hexlib_fastrpc_munmap           = NULL;
hexlib_remote_handle64_open_fn     hexlib_remote_handle64_open     = NULL;
hexlib_remote_handle64_invoke_fn   hexlib_remote_handle64_invoke   = NULL;
hexlib_remote_handle64_close_fn    hexlib_remote_handle64_close    = NULL;
hexlib_remote_handle_control_fn    hexlib_remote_handle_control    = NULL;
hexlib_remote_session_control_fn   hexlib_remote_session_control   = NULL;

static void *g_driver_handle = NULL;
static int   g_initialized   = 0;

/* Resolve `symbol` into `pfn` (whose type is already declared, so __typeof__
 * gives back the right function-pointer type for the cast dlsym's void*
 * return needs). `required` false means "log and keep going" -- used only for
 * rpcmem_alloc2, which some driver builds lack (see htp-drv.cpp's own
 * treatment of it); every other symbol here is required, and its absence
 * fails hexlib_drv_init() rather than being silently tolerated. */
#define HEXLIB_DLSYM(pfn, symbol, required)                                  \
    do {                                                                     \
        (pfn) = (__typeof__(pfn)) dlsym(handle, #symbol);                    \
        if ((pfn) == NULL) {                                                 \
            if (required) {                                                  \
                fprintf(stderr, "hexlib: dlsym(%s) failed: %s\n", #symbol,    \
                        dlerror());                                          \
                return -1;                                                   \
            }                                                                \
            fprintf(stderr,                                                  \
                    "hexlib: %s not present on this driver (optional), "     \
                    "continuing without it\n", #symbol);                     \
        }                                                                    \
    } while (0)

int hexlib_drv_init(void) {
    if (g_initialized) {
        return 0;
    }

    /* Two candidate paths, tried in order: the normal dynamic-linker-visible
     * name, then the vendor partition path some builds only expose the
     * driver under. The first that dlopen()s successfully wins. */
    static const char *const candidates[] = {
        "libcdsprpc.so",
        "/vendor/lib64/libcdsprpc.so",
    };

    void *handle = NULL;
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        handle = dlopen(candidates[i], RTLD_NOW);
        if (handle != NULL) {
            break;
        }
        fprintf(stderr, "hexlib: dlopen(%s) failed: %s\n", candidates[i], dlerror());
    }
    if (handle == NULL) {
        fprintf(stderr,
                "hexlib: could not load the FastRPC driver from any candidate "
                "path -- is this a compute-DSP-capable device?\n");
        return -1;
    }

    HEXLIB_DLSYM(hexlib_rpcmem_alloc,           rpcmem_alloc,           1);
    HEXLIB_DLSYM(hexlib_rpcmem_alloc2,          rpcmem_alloc2,          0);
    HEXLIB_DLSYM(hexlib_rpcmem_free,            rpcmem_free,            1);
    HEXLIB_DLSYM(hexlib_rpcmem_to_fd,           rpcmem_to_fd,           1);
    HEXLIB_DLSYM(hexlib_fastrpc_mmap,           fastrpc_mmap,           1);
    HEXLIB_DLSYM(hexlib_fastrpc_munmap,         fastrpc_munmap,         1);
    HEXLIB_DLSYM(hexlib_remote_handle64_open,   remote_handle64_open,   1);
    HEXLIB_DLSYM(hexlib_remote_handle64_invoke, remote_handle64_invoke, 1);
    HEXLIB_DLSYM(hexlib_remote_handle64_close,  remote_handle64_close,  1);
    HEXLIB_DLSYM(hexlib_remote_handle_control,  remote_handle_control,  1);
    HEXLIB_DLSYM(hexlib_remote_session_control, remote_session_control, 1);

    g_driver_handle = handle;
    g_initialized   = 1;
    return 0;
}

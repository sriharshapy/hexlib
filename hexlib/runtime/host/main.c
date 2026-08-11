/* hexlib/runtime/host/main.c -- hexlib_run: the CPU-side FastRPC client.
 *
 * ON A DEVICE, THE QAIC STUB IS LINKED -- THE OPPOSITE OF THE SIMULATOR
 * ARRANGEMENT. Through Task 8, hexlib_iface_open/_start/_mmap/_invoke/_stop/
 * _close were called as plain C functions bound directly to skel.c's
 * definitions in one Hexagon ELF (see runtime/build.py's build_sim_qexe):
 * the qaic-generated stub was deliberately never linked there, because it
 * defines those exact same names and both live in one address space. Here
 * the skel is a separate Hexagon .so the FastRPC framework loads on the
 * CDSP, and this aarch64 binary links hexlib_iface_stub.c instead -- so
 * calling hexlib_iface_invoke() from run_self_test() below is the FIRST
 * thing in this project ever to exercise qaic's real argument marshalling
 * into a remote_arg[] and a genuine remote_handle64_invoke() call. Every
 * simulator run before this task tested hexlib's own code (batch parsing,
 * dispatch, kernels) with the marshalling layer completely bypassed; this is
 * the one binary that finally puts it in the loop.
 *
 * ABSENCE OF A RESPONSE IS A FAILURE, NEVER A SUCCESS. HEXLIB_DSP_OK is 1,
 * never 0 (hexlib_dsp.h), specifically so a zero-filled buffer that nothing
 * ever wrote cannot read as success. Every path below that reads a response
 * checks its magic FIRST, before its status: absent, truncated, or
 * wrong-magic all fail with a distinct exit code and, on the --batch path,
 * write no output file at all. This project has already shipped a device-farm
 * job that ran no tests and reported passing off an empty result; the same
 * shape of bug here would be a "successful" run with a garbage or all-zero
 * output file.
 */
#include "hexlib_host.h"
#include "hexlib_dsp.h"
#include "hexlib_iface.h"   /* hexlib_iface_mmap/_munmap -- needed ONLY for
                             * --unmapped's deliberately-skipped registration
                             * call; see alloc_maybe_unmapped() below. Every
                             * ordinary allocation still goes through
                             * hexlib_alloc() (buffers.c), which already
                             * pulls this header in the same way. */

#include <remote.h>
#include <rpcmem.h>         /* RPCMEM_HEAP_ID_SYSTEM / RPCMEM_DEFAULT_FLAGS,
                             * for the same reason as above. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The wire "kind" for the scale op. Must match
 * hexlib.runtime.genentry.KIND_ID["scale"] == 9 -- there is no shared C
 * header for these ids (genentry.py emits the DSP-side dispatch table
 * straight from that Python dict; nothing generates a host-side mirror of
 * it), so this one constant is pinned here, by name and by comment, rather
 * than left to drift silently. A wrong value here is not silent, though: it
 * would make hexlib_dispatch_batch() return HEXLIB_DSP_ERR_NO_KERNEL, which
 * --self-test below reports as a failure, never a pass. */
#define HEXLIB_KIND_SCALE 9u

#define SELF_TEST_N       4100      /* 64*64 + 4: exercises the scalar tail. */
#define SELF_TEST_FACTOR  0.125f    /* A power of two: exact in fp16. */

/* --coherency-check's two constants -- see run_coherency_check()'s own
 * header comment for why each one is what it is. */
#define COHERENCY_SENTINEL 1.0f     /* Any nonzero, finite fp16 value works;
                                     * the expected result is bit-exact zero,
                                     * so this can never be confused with it. */
#define COHERENCY_FACTOR   0.0f     /* x * 0.0 is bit-exact zero in fp16 for
                                     * any finite, non-NaN x -- no numerically
                                     * ambiguous case, so a wrong result here
                                     * cannot be blamed on kernel arithmetic. */

enum {
    HEXLIB_EXIT_OK             = 0,
    HEXLIB_EXIT_USAGE          = 1,
    HEXLIB_EXIT_SESSION_FAILED = 2,
    HEXLIB_EXIT_NO_RESPONSE    = 3,   /* absent / truncated / wrong-magic */
    HEXLIB_EXIT_OP_FAILED      = 4,
    HEXLIB_EXIT_MISMATCH       = 5,
    HEXLIB_EXIT_COHERENCY_MISS = 6,   /* --coherency-check: status OK, op OK,
                                       * but the sentinel survived -- either a
                                       * dispatch bug or a real coherency miss;
                                       * see run_coherency_check()'s printed
                                       * cycles_total to tell which. */
};

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s --caps\n"
            "       %s --self-test [--unmapped | --coherency-check]\n"
            "       %s --batch <file> --in <file> --out <file>\n",
            argv0, argv0, argv0);
}

/* Checks the ONE thing that makes a response trustworthy at all: the magic.
 * A NULL/too-short/wrong-magic buffer is refused before its status field is
 * even read -- there is no status to trust in a response that was never
 * written, or that belongs to some other protocol entirely. */
static int response_is_valid(const uint8_t *rsp, size_t rsp_len, uint32_t *status_out) {
    if (rsp == NULL || rsp_len < sizeof(struct hexlib_batch_rsp_hdr)) {
        return 0;
    }
    struct hexlib_batch_rsp_hdr hdr;
    memcpy(&hdr, rsp, sizeof(hdr));
    if (hdr.magic != HEXLIB_BATCH_MAGIC) {
        return 0;
    }
    *status_out = hdr.status;
    return 1;
}

static void print_caps(void) {
    if (hexlib_drv_init() != 0) {
        fprintf(stderr, "hexlib: --caps: could not load the FastRPC driver\n");
        return;
    }
    struct hexlib_caps caps;
    if (hexlib_query_caps(CDSP_DOMAIN_ID, &caps) != 0) {
        fprintf(stderr, "hexlib: --caps: capability query failed\n");
        return;
    }
    printf("domain              = CDSP (%d)\n", CDSP_DOMAIN_ID);
    printf("domain_support      = %u\n", caps.domain_support);
    printf("unsigned_pd_support = %u\n", caps.unsigned_pd_support);
    printf("hvx_support_128b    = %u\n", caps.hvx_support_128b);
    printf("vtcm_page           = %u\n", caps.vtcm_page);
    printf("vtcm_count          = %u\n", caps.vtcm_count);
    printf("vtcm_total_bytes    = %llu\n",
           (unsigned long long) caps.vtcm_page * (unsigned long long) caps.vtcm_count);
    printf("arch_ver            = %u (0x%04x)\n", caps.arch_ver, caps.arch_ver);
    /* HMX_SUPPORT_DEPTH reads 0 on the measured target. That is NOT evidence
     * HMX is absent -- see the task's own measured-device-facts record -- so
     * this prints the raw number and says so, rather than translating it
     * into a yes/no HMX verdict this query cannot actually support. */
    printf("hmx_support_depth   = %u (0 is not evidence HMX is absent -- "
           "settle by direct test, not by this query)\n",
           caps.hmx_support_depth);
}

/* Build a one-op scale_fp16 batch: two buffers (x, y), one tensor per
 * buffer, one `scale` op. Field-for-field, this is hexlib_batch_hdr /
 * hexlib_buf_desc / hexlib_tensor / hexlib_op_desc from hexlib_dsp.h, which
 * on a little-endian aarch64 host has the IDENTICAL in-memory layout as
 * hexlib.runtime.wire.py's struct-packed format (verified: every field in
 * every one of those four C structs is naturally aligned already, so there
 * is no padding a Python `struct.pack("<...")` format string would not also
 * produce). So this function fills the real C structs and memcpy()s them
 * into the blob -- no hand-rolled byte packing, and nothing here can drift
 * from hexlib_dsp.h the way independently-maintained packing code could. */
static uint8_t *build_scale_batch(int fd_x, int fd_y, size_t nbytes, float factor,
                                  size_t *out_len) {
    size_t total = sizeof(struct hexlib_batch_hdr)
                 + 2 * sizeof(struct hexlib_buf_desc)
                 + 2 * sizeof(struct hexlib_tensor)
                 + 1 * sizeof(struct hexlib_op_desc);
    uint8_t *blob = (uint8_t *) calloc(1, total);
    if (blob == NULL) {
        return NULL;
    }

    uint32_t off_bufs    = (uint32_t) sizeof(struct hexlib_batch_hdr);
    uint32_t off_tensors = off_bufs + 2 * (uint32_t) sizeof(struct hexlib_buf_desc);
    uint32_t off_ops     = off_tensors + 2 * (uint32_t) sizeof(struct hexlib_tensor);

    struct hexlib_batch_hdr hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic       = HEXLIB_BATCH_MAGIC;
    hdr.version     = HEXLIB_BATCH_VERSION;
    hdr.total_size  = (uint32_t) total;
    hdr.n_bufs      = 2;
    hdr.n_tensors   = 2;
    hdr.n_ops       = 1;
    hdr.off_bufs    = off_bufs;
    hdr.off_tensors = off_tensors;
    hdr.off_ops     = off_ops;
    hdr.flags       = 0;
    memcpy(blob, &hdr, sizeof(hdr));

    struct hexlib_buf_desc bufs[2];
    memset(bufs, 0, sizeof(bufs));
    bufs[0].base = 0;   /* Host never writes an address -- see buffers.c. */
    bufs[0].size = (uint64_t) nbytes;
    bufs[0].fd   = (uint32_t) fd_x;
    bufs[0].flags = 0;
    bufs[1].base = 0;
    bufs[1].size = (uint64_t) nbytes;
    bufs[1].fd   = (uint32_t) fd_y;
    bufs[1].flags = 0;
    memcpy(blob + off_bufs, bufs, sizeof(bufs));

    struct hexlib_tensor tens[2];
    memset(tens, 0, sizeof(tens));
    for (int i = 0; i < 2; i++) {
        tens[i].bi     = (uint32_t) i;
        tens[i].offset = 0;
        tens[i].nbytes = (uint32_t) nbytes;
        tens[i].dtype  = 1;   /* hexlib.runtime.wire.DTYPE_ID["fp16"] */
        tens[i].layout = 0;   /* hexlib.runtime.wire.LAYOUT_ID["row_major"] */
        tens[i].ne[0]  = SELF_TEST_N;
        tens[i].ne[1]  = 1;
        tens[i].ne[2]  = 1;
        tens[i].ne[3]  = 1;
        tens[i].data   = 0;   /* DSP-side scratch. Host writes 0. */
    }
    memcpy(blob + off_tensors, tens, sizeof(tens));

    struct hexlib_op_desc op;
    memset(&op, 0, sizeof(op));
    op.kind  = HEXLIB_KIND_SCALE;
    op.flags = 0;
    /* `factor` is a float attr, so its wire slot carries the caller's IEEE-754
     * bit pattern (0.125f from run_self_test, 0.0f from
     * run_coherency_check), not the integer 0 that `(int32_t) factor` would
     * silently produce -- genentry.py's generated entry reads it back as
     * `((const float *) a->params)[0]`, a raw reinterpretation, not a
     * numeric conversion. */
    union { float f; int32_t i; } factor_bits;
    factor_bits.f = factor;
    op.params[0] = factor_bits.i;
    for (int i = 1; i < HEXLIB_MAX_PARAMS; i++) {
        op.params[i] = 0;
    }
    for (int i = 0; i < HEXLIB_MAX_SRC; i++) {
        op.src[i] = 0xFFFF;
    }
    for (int i = 0; i < HEXLIB_MAX_DST; i++) {
        op.dst[i] = 0xFFFF;
    }
    op.src[0] = 0;   /* tensor 0: x */
    op.dst[0] = 1;   /* tensor 1: y */
    memcpy(blob + off_ops, &op, sizeof(op));

    *out_len = total;
    return blob;
}

/* ==========================================================================
 * --unmapped -- THE LOAD-BEARING CHECK.
 *
 * On the simulator, host and DSP share one address space: `HAP_mmap` is
 * `return (void*)(uintptr_t)fd;` and `rpcmem_to_fd` is
 * `return (int)(uintptr_t)po;` there, so the whole pointer -> fd -> map ->
 * base chain is an IDENTITY FUNCTION and the "mapped" address is always the
 * real host pointer. No comparison of VALUES can tell a correct DSP
 * implementation apart from one that simply read the host's own address --
 * which would work perfectly on the simulator and fail instantly on real
 * hardware. The ONLY thing that discriminates is a table lookup:
 * `hexlib_bufs_map` (skel_bufs.c) consults a table that only
 * `hexlib_bufs_register` populates, and that only happens in response to a
 * genuine `hexlib_iface_mmap` call. So this mode allocates rpcmem and gets
 * an fd exactly as `hexlib_alloc()` (buffers.c) does, but DELIBERATELY SKIPS
 * the `hexlib_iface_mmap` registration call -- mirroring
 * `hexlib/runtime/simhost/simhost.c`'s own `--unmapped`, which withholds the
 * identical call for the identical reason (see that file's header comment).
 * `hexlib_dispatch_batch` (skel_dispatch.c) must then refuse with
 * `HEXLIB_DSP_ERR_UNMAPPED` (7) as the batch's TOP-LEVEL status -- no new
 * print statement is needed for that refusal to be visible: run_self_test's
 * own `status != HEXLIB_DSP_OK` branch already reports it and returns
 * HEXLIB_EXIT_OP_FAILED (4).
 *
 * buffers.c is out of scope for this change (only this file may move) and
 * `hexlib_alloc()` has no knob for skipping its registration call, so this
 * is a small, local duplicate of its allocation sequence -- not an edit to
 * it. The CPU-side rpcmem_alloc/rpcmem_to_fd/fastrpc_mmap sequence still
 * runs in full ("the host allocates its rpcmem buffer and gets its fd as
 * usual"); only the DSP-side registration is withheld.
 * ========================================================================*/
static int alloc_maybe_unmapped(hexlib_ctx *ctx, hexlib_buf **out, size_t size,
                                int skip_dsp_register) {
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

    int rc = hexlib_fastrpc_mmap(ctx->domain, fd, ptr, 0, size, FASTRPC_MAP_FD);
    if (rc != 0) {
        fprintf(stderr,
                "hexlib: fastrpc_mmap(fd=%d, size=%zu) failed (rc %d)\n",
                fd, size, rc);
        hexlib_rpcmem_free(ptr);
        return -1;
    }

    if (skip_dsp_register) {
        /* DELIBERATELY NOT REGISTERED WITH THE SKEL. hexlib_bufs_map()'s
         * table lookup (skel_bufs.c) has nothing to find for this fd, so a
         * batch that references it must be refused with
         * HEXLIB_DSP_ERR_UNMAPPED (7) -- never silently succeed by reading a
         * host address, which is the one failure mode the simulator's
         * identity-mapped HAP_mmap/rpcmem_to_fd cannot rule out. See the
         * file header comment above this function. */
        printf("hexlib: --unmapped: fd %d deliberately not registered with the skel\n", fd);
    } else {
        /* Registers the SAME fd with the DSP-side skel (hexlib_iface_mmap ->
         * hexlib_bufs_register -> HAP_mmap in skel_bufs.c) -- the ordinary
         * path, identical to hexlib_alloc()'s own second mapping call. */
        int arc = hexlib_iface_mmap(ctx->handle, (uint32_t) fd, (uint32_t) size);
        if (arc != AEE_SUCCESS) {
            fprintf(stderr, "hexlib: hexlib_iface_mmap(fd=%d) failed (rc %d)\n", fd, arc);
            hexlib_fastrpc_munmap(ctx->domain, fd, ptr, size);
            hexlib_rpcmem_free(ptr);
            return -1;
        }
    }

    hexlib_buf *buf = (hexlib_buf *) calloc(1, sizeof(*buf));
    if (buf == NULL) {
        if (!skip_dsp_register) {
            hexlib_iface_munmap(ctx->handle, (uint32_t) fd);
        }
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

/* Mirror of hexlib_free() (buffers.c), for a buffer allocated by
 * alloc_maybe_unmapped() above. `was_unmapped` must match the
 * `skip_dsp_register` the buffer was allocated with -- calling
 * hexlib_iface_munmap() on an fd that was never registered would just be
 * one more no-op RPC, but skipping this parameter entirely and always
 * calling it would silently paper over a mismatch between allocation and
 * teardown, which is exactly the kind of asymmetry this file's callers must
 * get right by construction rather than by accident. */
static void free_maybe_unmapped(hexlib_ctx *ctx, hexlib_buf *buf, int was_unmapped) {
    if (buf == NULL) {
        return;
    }
    if (!was_unmapped) {
        hexlib_iface_munmap(ctx->handle, (uint32_t) buf->fd);
    }
    hexlib_fastrpc_munmap(ctx->domain, buf->fd, buf->ptr, buf->size);
    hexlib_rpcmem_free(buf->ptr);
    free(buf);
}

/* `unmapped`: when true, both self-test buffers are allocated via
 * alloc_maybe_unmapped() with DSP-side registration withheld -- see that
 * function's header comment. The rest of this function is otherwise
 * unchanged; the DSP is expected to refuse with HEXLIB_DSP_ERR_UNMAPPED (7),
 * which the existing `status != HEXLIB_DSP_OK` branch below already reports
 * and turns into HEXLIB_EXIT_OP_FAILED (4). */
static int run_self_test(int unmapped) {
    hexlib_ctx *ctx = NULL;
    if (hexlib_open(&ctx, CDSP_DOMAIN_ID) != 0) {
        fprintf(stderr, "hexlib: --self-test: could not open a CDSP session\n");
        return HEXLIB_EXIT_SESSION_FAILED;
    }

    size_t nbytes = (size_t) SELF_TEST_N * sizeof(__fp16);
    hexlib_buf *bx = NULL, *by = NULL;
    int alloc_failed = unmapped
        ? (alloc_maybe_unmapped(ctx, &bx, nbytes, 1) != 0 ||
           alloc_maybe_unmapped(ctx, &by, nbytes, 1) != 0)
        : (hexlib_alloc(ctx, &bx, nbytes) != 0 ||
           hexlib_alloc(ctx, &by, nbytes) != 0);
    if (alloc_failed) {
        fprintf(stderr, "hexlib: --self-test: buffer allocation failed\n");
        if (unmapped) {
            free_maybe_unmapped(ctx, bx, 1);
            free_maybe_unmapped(ctx, by, 1);
        } else {
            hexlib_free(ctx, bx);
            hexlib_free(ctx, by);
        }
        hexlib_close(ctx);
        return HEXLIB_EXIT_SESSION_FAILED;
    }

    /* Deliberately not a single repeated value: exercises the full range
     * scale_fp16 handles, body and scalar tail alike. Scaling by a power of
     * two (0.125f = 2^-3) only shifts the exponent field -- no mantissa bit
     * is lost -- so the expected result is BIT-EXACT, not approximate. Any
     * difference at all is therefore a marshalling bug, never a precision
     * one; see kernels/scale_fp16/kernel_api.h. */
    __fp16 *x = (__fp16 *) bx->ptr;
    for (int i = 0; i < SELF_TEST_N; i++) {
        x[i] = (__fp16) ((float) ((i % 17) - 8) * 0.5f);
    }

    size_t batch_len = 0;
    uint8_t *batch = build_scale_batch(bx->fd, by->fd, nbytes, SELF_TEST_FACTOR, &batch_len);
    if (batch == NULL) {
        fprintf(stderr, "hexlib: --self-test: out of memory building the batch\n");
        if (unmapped) {
            free_maybe_unmapped(ctx, bx, 1);
            free_maybe_unmapped(ctx, by, 1);
        } else {
            hexlib_free(ctx, bx);
            hexlib_free(ctx, by);
        }
        hexlib_close(ctx);
        return HEXLIB_EXIT_SESSION_FAILED;
    }

    size_t rsp_cap = sizeof(struct hexlib_batch_rsp_hdr) + sizeof(struct hexlib_op_result);
    uint8_t *rsp = (uint8_t *) calloc(1, rsp_cap);
    size_t rsp_len = 0;
    int rc = hexlib_invoke(ctx, batch, batch_len, rsp, rsp_cap, &rsp_len);

    uint32_t status = 0;
    int exit_code = HEXLIB_EXIT_OK;
    if (rc != 0 || !response_is_valid(rsp, rsp_len, &status)) {
        fprintf(stderr,
                "hexlib: --self-test: no valid response from the DSP (rc=%d) "
                "-- absence of a response is a failure, never a pass\n", rc);
        exit_code = HEXLIB_EXIT_NO_RESPONSE;
    } else if (status != HEXLIB_DSP_OK) {
        fprintf(stderr, "hexlib: --self-test: batch status %u, not HEXLIB_DSP_OK\n",
                status);
        exit_code = HEXLIB_EXIT_OP_FAILED;
    } else {
        const struct hexlib_op_result *result =
            (const struct hexlib_op_result *) (rsp + sizeof(struct hexlib_batch_rsp_hdr));
        if (rsp_len < sizeof(struct hexlib_batch_rsp_hdr) + sizeof(*result) ||
            result->status != HEXLIB_DSP_OK) {
            fprintf(stderr, "hexlib: --self-test: op result missing or not OK\n");
            exit_code = HEXLIB_EXIT_OP_FAILED;
        } else {
            const __fp16 *y = (const __fp16 *) by->ptr;
            int mismatches = 0;
            for (int i = 0; i < SELF_TEST_N; i++) {
                __fp16 expect = (__fp16) ((float) x[i] * SELF_TEST_FACTOR);
                if (memcmp(&expect, &y[i], sizeof(__fp16)) != 0) {
                    if (mismatches < 5) {
                        fprintf(stderr, "hexlib: --self-test: mismatch at index %d\n", i);
                    }
                    mismatches++;
                }
            }
            if (mismatches != 0) {
                fprintf(stderr, "hexlib: --self-test: %d/%d values not bit-exact\n",
                        mismatches, SELF_TEST_N);
                exit_code = HEXLIB_EXIT_MISMATCH;
            } else {
                printf("hexlib: --self-test: PASS (%d values, bit-exact)\n", SELF_TEST_N);
                /* The response header's own PCYCLE-measured total (see
                 * skel_dispatch.c) -- the only DSP-measured cycle count this
                 * binary can report at all, and the execution-proof signal
                 * --coherency-check's discriminator depends on. Previously
                 * validated by response_is_valid() above and read fresh here
                 * rather than threaded through as an extra out-parameter. */
                struct hexlib_batch_rsp_hdr full_hdr;
                memcpy(&full_hdr, rsp, sizeof(full_hdr));
                printf("hexlib: --self-test: cycles_total=%llu\n",
                       (unsigned long long) full_hdr.cycles_total);
            }
        }
    }

    free(rsp);
    free(batch);
    if (unmapped) {
        free_maybe_unmapped(ctx, bx, 1);
        free_maybe_unmapped(ctx, by, 1);
    } else {
        hexlib_free(ctx, bx);
        hexlib_free(ctx, by);
    }
    hexlib_close(ctx);
    return exit_code;
}

/* ==========================================================================
 * --coherency-check -- distinguishes a cache-coherency miss from a
 * marshalling/dispatch bug. Design doc §6.1 (corrected 2026-08-11).
 *
 * WHY THIS EXISTS. `buffers.c` maps rpcmem with FASTRPC_MAP_FD, which
 * <remote.h> documents as putting cache maintenance on US; `rpcmem`
 * allocates CACHED memory by default; and there is no CPU-side flush or
 * invalidate call anywhere in the SDK. So a DSP write that never becomes
 * visible to the CPU is a real possibility on real hardware, and it
 * presents EXACTLY like a marshalling bug: status OK, wrong bytes. This is
 * the one place marshalling is exercised at all (see main.c's own file
 * header), so the two failure modes would otherwise confound each other
 * with no cheaper way to tell them apart.
 *
 * THE SENTINEL ALONE IS NOT ENOUGH -- READ THIS BEFORE CHANGING ANYTHING
 * BELOW. An earlier version of this design pre-wrote a sentinel into the
 * output buffer and ran scale_fp16 with factor=0.0 so the correct result is
 * bit-exact zero, then just checked whether the sentinel survived. That
 * FAILS TO DISCRIMINATE: if dispatch silently no-ops and still returns
 * HEXLIB_DSP_OK -- a marshalling bug, not a coherency one -- the observable
 * is IDENTICAL to a coherency miss (status OK, sentinel intact). What
 * actually separates the two is an execution-proof signal: `cycles_total`,
 * the DSP's own PCYCLE-measured total around the kernel call
 * (skel_dispatch.c), which is exactly zero unless the kernel genuinely ran.
 *
 *     cycles 0,  sentinel intact      -> the kernel never ran: a dispatch bug
 *     cycles >0, sentinel intact      -> it ran; the write never reached the
 *                                        host: COHERENCY
 *     cycles >0, sentinel overwritten -> both fine, for THIS direction
 *
 * This function prints BOTH the cycles_total line and the COHERENCY
 * verdict line unconditionally (once the batch status and op status are
 * both confirmed OK), so all three rows of that table are distinguishable
 * from stdout alone -- never just "the bad thing is absent" (see this
 * file's project-wide discipline on that, stated in the header above main()).
 *
 * WHAT THIS DOES NOT PROVE -- DO NOT READ MORE INTO A PASS THAN THIS.
 * This exercises only the DSP-write -> host-read direction (the DSP writes
 * `y`, the CPU reads it back afterwards). A host-write -> DSP-read miss (the
 * CPU writes `x`, the DSP reads something stale from ITS cache) is a
 * different direction through the same cache hierarchy and is NOT covered
 * here at all. Nor is this kernel-independent: it says something about
 * scale_fp16's one write pattern and this one buffer size, not about every
 * kernel or every buffer size hexlib might ever dispatch.
 * ========================================================================*/
static int run_coherency_check(void) {
    hexlib_ctx *ctx = NULL;
    if (hexlib_open(&ctx, CDSP_DOMAIN_ID) != 0) {
        fprintf(stderr, "hexlib: --coherency-check: could not open a CDSP session\n");
        return HEXLIB_EXIT_SESSION_FAILED;
    }

    size_t nbytes = (size_t) SELF_TEST_N * sizeof(__fp16);
    hexlib_buf *bx = NULL, *by = NULL;
    if (hexlib_alloc(ctx, &bx, nbytes) != 0 || hexlib_alloc(ctx, &by, nbytes) != 0) {
        fprintf(stderr, "hexlib: --coherency-check: buffer allocation failed\n");
        hexlib_free(ctx, bx);
        hexlib_free(ctx, by);
        hexlib_close(ctx);
        return HEXLIB_EXIT_SESSION_FAILED;
    }

    /* `x` need not be anything special -- factor=0.0 makes the correct
     * result bit-exact zero regardless of its contents, for any finite,
     * non-NaN input. Reused shape from run_self_test purely for a
     * reasonable non-degenerate input. */
    __fp16 *x = (__fp16 *) bx->ptr;
    for (int i = 0; i < SELF_TEST_N; i++) {
        x[i] = (__fp16) ((float) ((i % 17) - 8) * 0.5f);
    }

    /* THE SENTINEL. Written into the OUTPUT buffer, before invoke, so that
     * only the DSP's own write to `y` -- or the CPU's failure to observe it
     * -- can change what this side reads back. */
    __fp16 *y = (__fp16 *) by->ptr;
    for (int i = 0; i < SELF_TEST_N; i++) {
        y[i] = (__fp16) COHERENCY_SENTINEL;
    }

    size_t batch_len = 0;
    uint8_t *batch = build_scale_batch(bx->fd, by->fd, nbytes, COHERENCY_FACTOR, &batch_len);
    if (batch == NULL) {
        fprintf(stderr, "hexlib: --coherency-check: out of memory building the batch\n");
        hexlib_free(ctx, bx);
        hexlib_free(ctx, by);
        hexlib_close(ctx);
        return HEXLIB_EXIT_SESSION_FAILED;
    }

    size_t rsp_cap = sizeof(struct hexlib_batch_rsp_hdr) + sizeof(struct hexlib_op_result);
    uint8_t *rsp = (uint8_t *) calloc(1, rsp_cap);
    size_t rsp_len = 0;
    int rc = hexlib_invoke(ctx, batch, batch_len, rsp, rsp_cap, &rsp_len);

    uint32_t status = 0;
    int exit_code = HEXLIB_EXIT_OK;
    if (rc != 0 || !response_is_valid(rsp, rsp_len, &status)) {
        fprintf(stderr,
                "hexlib: --coherency-check: no valid response from the DSP (rc=%d) "
                "-- absence of a response is a failure, never a pass\n", rc);
        exit_code = HEXLIB_EXIT_NO_RESPONSE;
    } else if (status != HEXLIB_DSP_OK) {
        fprintf(stderr, "hexlib: --coherency-check: batch status %u, not HEXLIB_DSP_OK\n",
                status);
        exit_code = HEXLIB_EXIT_OP_FAILED;
    } else {
        const struct hexlib_op_result *result =
            (const struct hexlib_op_result *) (rsp + sizeof(struct hexlib_batch_rsp_hdr));
        if (rsp_len < sizeof(struct hexlib_batch_rsp_hdr) + sizeof(*result) ||
            result->status != HEXLIB_DSP_OK) {
            fprintf(stderr, "hexlib: --coherency-check: op result missing or not OK\n");
            exit_code = HEXLIB_EXIT_OP_FAILED;
        } else {
            /* STATUS OK, PROVEN: marshalling and dispatch both genuinely
             * succeeded (both the batch-level status and this op's own
             * status say so). Only now is reading the sentinel back
             * meaningful at all -- see this function's own header comment. */
            struct hexlib_batch_rsp_hdr full_hdr;
            memcpy(&full_hdr, rsp, sizeof(full_hdr));

            const __fp16 *yr = (const __fp16 *) by->ptr;
            __fp16 zero = (__fp16) 0.0f;
            int overwritten = 1;
            for (int i = 0; i < SELF_TEST_N; i++) {
                if (memcmp(&yr[i], &zero, sizeof(__fp16)) != 0) {
                    overwritten = 0;
                    break;
                }
            }

            /* Both lines, always -- see the file header on why cycles_total
             * must be printed unconditionally rather than only on failure:
             * it is what tells a genuine coherency miss apart from a
             * dispatch bug, and a test reading only the COHERENCY line could
             * not make that distinction on its own. */
            printf("hexlib: --coherency-check: cycles_total=%llu\n",
                   (unsigned long long) full_hdr.cycles_total);
            if (overwritten) {
                printf("COHERENCY sentinel_overwritten\n");
            } else {
                printf("COHERENCY sentinel_unchanged\n");
                exit_code = HEXLIB_EXIT_COHERENCY_MISS;
            }
        }
    }

    free(rsp);
    free(batch);
    hexlib_free(ctx, bx);
    hexlib_free(ctx, by);
    hexlib_close(ctx);
    return exit_code;
}

static uint8_t *read_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }
    uint8_t *buf = (uint8_t *) malloc((size_t) n > 0 ? (size_t) n : 1);
    if (buf == NULL) {
        fclose(f);
        return NULL;
    }
    size_t got = fread(buf, 1, (size_t) n, f);
    fclose(f);
    if (got != (size_t) n) {
        free(buf);
        return NULL;
    }
    *len_out = (size_t) n;
    return buf;
}

/* The general path: `--batch <f>` is a wire-format template built ahead of
 * time (buffer sizes and every tensor/op already filled in; each
 * hexlib_buf_desc's `fd` is a placeholder this function overwrites once it
 * has actually allocated rpcmem for it -- `base` in the template is already
 * required to be 0, same as everywhere else on this side of the wire).
 *
 * CONVENTION, NOT PROTOCOL: this CLI treats every buffer the template
 * declares except the last as an input, filled in order from `--in`, and the
 * last as the output, written to `--out`. The wire format itself has no
 * concept of "input" vs "output" buffer -- that only exists in how an op's
 * src/dst reference tensors -- so this is a one-shot-CLI simplification, not
 * something skel_dispatch.c or wire.py know about. */
static int run_batch_file(const char *batch_path, const char *in_path,
                          const char *out_path) {
    size_t tmpl_len = 0;
    uint8_t *tmpl = read_file(batch_path, &tmpl_len);
    if (tmpl == NULL || tmpl_len < sizeof(struct hexlib_batch_hdr)) {
        fprintf(stderr, "hexlib: --batch: could not read %s\n", batch_path);
        free(tmpl);
        return HEXLIB_EXIT_USAGE;
    }

    struct hexlib_batch_hdr hdr;
    memcpy(&hdr, tmpl, sizeof(hdr));
    if (hdr.magic != HEXLIB_BATCH_MAGIC) {
        fprintf(stderr, "hexlib: --batch: %s is not a hexlib batch (bad magic)\n",
                batch_path);
        free(tmpl);
        return HEXLIB_EXIT_USAGE;
    }
    if (hdr.n_bufs == 0 || hdr.n_bufs > HEXLIB_MAX_BUFS ||
        (uint64_t) hdr.off_bufs + (uint64_t) hdr.n_bufs * sizeof(struct hexlib_buf_desc) > tmpl_len) {
        fprintf(stderr, "hexlib: --batch: malformed buffer table in %s\n", batch_path);
        free(tmpl);
        return HEXLIB_EXIT_USAGE;
    }

    hexlib_ctx *ctx = NULL;
    if (hexlib_open(&ctx, CDSP_DOMAIN_ID) != 0) {
        fprintf(stderr, "hexlib: --batch: could not open a CDSP session\n");
        free(tmpl);
        return HEXLIB_EXIT_SESSION_FAILED;
    }

    hexlib_buf **bufs = (hexlib_buf **) calloc(hdr.n_bufs, sizeof(hexlib_buf *));
    struct hexlib_buf_desc *descs = (struct hexlib_buf_desc *) (tmpl + hdr.off_bufs);

    size_t in_len = 0;
    uint8_t *in_data = read_file(in_path, &in_len);
    if (in_data == NULL) {
        fprintf(stderr, "hexlib: --batch: could not read %s\n", in_path);
        free(bufs);
        free(tmpl);
        hexlib_close(ctx);
        return HEXLIB_EXIT_USAGE;
    }

    int ok = 1;
    size_t in_off = 0;
    for (uint32_t i = 0; i < hdr.n_bufs && ok; i++) {
        size_t sz = (size_t) descs[i].size;
        if (hexlib_alloc(ctx, &bufs[i], sz) != 0) {
            fprintf(stderr, "hexlib: --batch: failed to allocate buffer %u (%zu bytes)\n",
                    i, sz);
            ok = 0;
            break;
        }
        if (i + 1 < hdr.n_bufs) {   /* an input, per the convention above */
            if (in_off + sz > in_len) {
                fprintf(stderr,
                        "hexlib: --batch: %s is shorter than the inputs the "
                        "batch template declares\n", in_path);
                ok = 0;
                break;
            }
            memcpy(bufs[i]->ptr, in_data + in_off, sz);
            in_off += sz;
        }
        /* Patch the real fd into the working copy of the buffer table.
         * `base` stays 0 -- hexlib_buf_to_desc() never sets anything else. */
        struct hexlib_buf_desc d;
        hexlib_buf_to_desc(bufs[i], &d);
        memcpy(&descs[i], &d, sizeof(d));
    }
    free(in_data);

    int exit_code = HEXLIB_EXIT_OK;
    uint8_t *rsp = NULL;

    if (!ok) {
        exit_code = HEXLIB_EXIT_USAGE;
    } else {
        size_t rsp_cap = sizeof(struct hexlib_batch_rsp_hdr)
                        + (size_t) hdr.n_ops * sizeof(struct hexlib_op_result);
        rsp = (uint8_t *) calloc(1, rsp_cap);
        size_t rsp_len = 0;
        int rc = hexlib_invoke(ctx, tmpl, tmpl_len, rsp, rsp_cap, &rsp_len);

        uint32_t status = 0;
        if (rc != 0 || !response_is_valid(rsp, rsp_len, &status)) {
            /* NO OUTPUT FILE IS WRITTEN ON THIS PATH. See the file header --
             * an absent, truncated, or wrong-magic response must never be
             * mistaken for a result worth saving. */
            fprintf(stderr,
                    "hexlib: --batch: no valid response from the DSP -- "
                    "writing no output file\n");
            exit_code = HEXLIB_EXIT_NO_RESPONSE;
        } else if (status != HEXLIB_DSP_OK) {
            fprintf(stderr,
                    "hexlib: --batch: batch status %u, not HEXLIB_DSP_OK -- "
                    "writing no output file\n", status);
            exit_code = HEXLIB_EXIT_OP_FAILED;
        } else {
            /* ONLY NOW, after the magic AND the status are both confirmed
             * good, does anything get written to disk. */
            hexlib_buf *out_buf = bufs[hdr.n_bufs - 1];
            FILE *f = fopen(out_path, "wb");
            if (f == NULL || fwrite(out_buf->ptr, 1, out_buf->size, f) != out_buf->size) {
                fprintf(stderr, "hexlib: --batch: could not write %s\n", out_path);
                exit_code = HEXLIB_EXIT_USAGE;
            } else {
                printf("hexlib: --batch: wrote %zu bytes to %s\n", out_buf->size, out_path);
            }
            if (f != NULL) {
                fclose(f);
            }
        }
    }

    free(rsp);
    if (bufs != NULL) {
        for (uint32_t i = 0; i < hdr.n_bufs; i++) {
            if (bufs[i] != NULL) {
                hexlib_free(ctx, bufs[i]);
            }
        }
        free(bufs);
    }
    free(tmpl);
    hexlib_close(ctx);
    return exit_code;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "--caps") == 0) {
        print_caps();
        return HEXLIB_EXIT_OK;
    }
    if (argc >= 2 && strcmp(argv[1], "--self-test") == 0) {
        /* Two independent modifiers, either optional, checked past argv[1]:
         * `--unmapped` (run_self_test's own unmapped-buffer path) and
         * `--coherency-check` (a distinct function, since it needs a
         * pre-written sentinel and a different scale factor). If both are
         * given, --coherency-check wins and --unmapped is ignored -- that
         * combination is not part of this project's on-device test plan and
         * is left unspecified rather than given a third code path. */
        int unmapped = 0, coherency = 0;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--unmapped") == 0) {
                unmapped = 1;
            } else if (strcmp(argv[i], "--coherency-check") == 0) {
                coherency = 1;
            }
        }
        if (coherency) {
            return run_coherency_check();
        }
        return run_self_test(unmapped);
    }
    if (argc >= 2 && strcmp(argv[1], "--batch") == 0) {
        const char *batch_path = NULL, *in_path = NULL, *out_path = NULL;
        for (int i = 1; i + 1 < argc; i += 2) {
            if (strcmp(argv[i], "--batch") == 0) {
                batch_path = argv[i + 1];
            } else if (strcmp(argv[i], "--in") == 0) {
                in_path = argv[i + 1];
            } else if (strcmp(argv[i], "--out") == 0) {
                out_path = argv[i + 1];
            }
        }
        if (batch_path == NULL || in_path == NULL || out_path == NULL) {
            usage(argv[0]);
            return HEXLIB_EXIT_USAGE;
        }
        return run_batch_file(batch_path, in_path, out_path);
    }

    usage(argv[0]);
    return HEXLIB_EXIT_USAGE;
}

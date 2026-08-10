/* hexlib/runtime/simhost/simhost.c -- the host side, for the simulator.
 *
 * TASK 7B UPDATE: this file is now compiled into a QuRT-hosted SHARED OBJECT
 * (build_sim_so, hexlib/runtime/build.py), dlopen'd by the SDK's own prebuilt
 * `run_main_on_hexagon_sim` under a real booted QuRT kernel, instead of a
 * standalone `--force-dynamic` qexe (task 7's build_sim_qexe, retired -- see
 * build.py's module docstring for why). This file's OWN code did not change;
 * only how it gets packaged and launched did. It still has a plain `main()`
 * that `run_main_on_hexagon`'s own dsp-side driver calls after dlopen.
 *
 * ============================================================================
 * WHAT A SIMULATOR RUN OF THIS FILE DOES NOT PROVE -- READ THIS FIRST.
 *
 * This file calls hexlib_iface_open/_start/_mmap/_invoke/_stop/_close as
 * PLAIN C FUNCTIONS, bound by the linker DIRECTLY to skel.c's definitions --
 * simhost.o and the skel archive are both compiled into the SAME .so, so
 * this is still an ordinary intra-module call, not a qaic-marshalled one.
 * The qaic-generated stub (hexlib_iface_stub.c) -- the code that would
 * actually marshal these calls into a `remote_arg` scalar/buffer list and
 * drive them through `remote_handle64_open`/`_invoke` -- is DELIBERATELY NOT
 * LINKED INTO THIS .SO AT ALL. It defines the exact same function names as
 * skel.c's DSP-side implementation (confirmed by running qaic and reading
 * both generated files back), so linking both into one module is a
 * duplicate-symbol error, not merely redundant. The SDK's own calculator
 * example makes the identical choice: `calculator_q_C_SRCS` in
 * examples/calculator/hexagon.min never includes calculator_stub.c either.
 * Packaging this file as a shared object rather than a standalone executable
 * does NOT change this -- it changes how VTCM's own weak symbols get
 * resolved (dynamically, against the host process, at dlopen time -- see
 * build_sim_so's comment in build.py), not whether the qaic stub is linked
 * (it still is not).
 *
 * CONSEQUENCE: a simulator run through this file exercises hexlib's OWN
 * code -- batch parsing (hexlib_dispatch_batch), the buffer table
 * (hexlib_bufs_register/_map), the kernel dispatch table
 * (hexlib_kernel_table), kernel correctness, PCYCLE accounting, and now (as
 * of task 7b) real VTCM acquisition -- but it does NOT exercise qaic's
 * argument marshaling/demarshaling at all. That is a real gap against this
 * project's own design spec, which describes the simulator path as
 * exercising "a qaic stub/skel invoke": what actually happens here is a
 * plain function call, and the marshaling layer is completely bypassed.
 * Marshaling is only exercised on a real device, where the stub and skel
 * genuinely live in separate processes and the call cannot avoid the wire.
 * ============================================================================
 *
 * WHY THIS EXISTS. On a device the host is an aarch64 Android binary. On the
 * simulator there is no aarch64, so the "host" is Hexagon code in the same
 * module as the skel. Originally (task 7) that module was a monolithic
 * standalone qexe, the SDK's own BUILD_QEXES pattern (examples/calculator's
 * calculator_q). As of task 7b it is a shared object instead, because a
 * standalone qexe can never satisfy VTCM's real client/server manager (real
 * QuRT thread/clock primitives it structurally cannot host) -- see
 * .superpowers/sdd/2026-08-10-silicon-path-runtime/
 * investigation-sim-vtcm-and-marshalling.md.
 *
 * WHY THIS FILE CALLS hexlib_iface_open/start/mmap/invoke/stop/close DIRECTLY,
 * NOT THROUGH THE QAIC-GENERATED STUB. Reading calculator_q's own link line and
 * its generated calculator_stub.c/calculator_skel.c settled it: the generated
 * STUB (hexlib_iface_stub.c) defines the SAME function names
 * (hexlib_iface_open, _start, _mmap, _invoke, ...) as the DEVELOPER'S skel-side
 * implementation in skel.c -- on a device these live in two different ELFs
 * (host APK vs. DSP .so) so the names never collide, but statically linking
 * both into ONE module would be a duplicate-symbol error. calculator's own
 * hexagon.min never compiles calculator_stub.c into calculator_q either: only
 * the generated *_skel.c (an unused, harmless archive member here) and the
 * developer's *_imp.c (which implements calculator_open/_close/_sum/_max
 * directly) go into calculator_q's link. calculator_test.c's calls to
 * calculator_open/_sum resolve straight to that developer implementation --
 * there is no marshaling, no remote_handle64_open/_invoke, on this path at all
 * (confirmed by `hexagon-nm` on rtld.a/test_util.a/atomic.a: none of them
 * define remote_handle64_open/_close/_invoke). This file follows the same
 * shape: it calls hexlib_iface_open/etc. as plain C functions, which the
 * linker binds directly to skel.c's definitions.
 *
 * FILE I/O UNDER THE QURT-HOSTED PACKAGING -- READS AND WRITES ARE NOT
 * SYMMETRIC, CONFIRMED EMPIRICALLY (task 7b). Under task 7's standalone qexe,
 * relative fopen() paths resolved through hexagon-sim's own `--usefs <dir>`
 * angel-mode redirection for both reads and writes, so hexlib_in.bin/
 * hexlib_batch.bin/hexlib_out.bin/hexlib_rsp.bin all lived in one place. Under
 * this file's new QuRT-hosted packaging, relative fopen() READS of
 * hexlib_batch.bin/hexlib_in.bin still resolve through --usefs correctly
 * (verified: a batch that exists ONLY under --usefs's directory, nowhere
 * else, is read and executed correctly). But relative fopen(..., "wb") WRITES
 * of hexlib_rsp.bin/hexlib_out.bin land in the REAL launching process's own
 * working directory instead -- QuRT's own POSIX filesystem layer, not the
 * simulator's angel-mode redirection, appears to own file creation once a
 * real QuRT kernel is booted, and it does not consult --usefs the same way.
 * CONSEQUENCE FOR CALLERS (Task 8): a harness that wants the response/output
 * files to land next to the batch/input files it wrote MUST launch the
 * hexagon-sim subprocess with its OWN working directory set to the same
 * directory passed as --usefs (e.g. Python's subprocess `cwd=` kwarg) --
 * this file cannot fix this from its own side, because it does not know at
 * compile time what directory a future caller will use as --usefs.
 *
 * THE ONE THING TO BE CAREFUL ABOUT. Host and DSP are one address space here.
 * This file must never hand the skel a pointer; it registers an fd with
 * rpcmem_alloc()+rpcmem_to_fd() and sends offsets, exactly as the device host
 * does. `--unmapped` exercises the negative case, which is the test that makes
 * a simulator pass transferable: it deliberately skips hexlib_iface_mmap, so
 * the skel must refuse (HEXLIB_DSP_ERR_UNMAPPED), not silently read the host's
 * address the way a shared-address-space bug would let it. THIS BEHAVIOR MUST
 * NOT CHANGE: `--unmapped` is load-bearing for Task 8's own discriminator
 * test, which proves a skel that leaned on the shared address space would
 * pass a request whose buffer was never mapped -- do not "fix" this into
 * mapping anyway.
 *
 * hexlib_iface_invoke HAS NO "resultLenOut" PARAMETER (see skel.c's own header
 * comment: `rout sequence<octet> result` marshals only a capacity). The
 * response is self-describing -- hexlib_batch_rsp_hdr.n_ops says how many
 * hexlib_op_result entries follow -- so that is what this file uses to decide
 * how many bytes of the response buffer are meaningful.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "hexlib_dsp.h"
#include "hexlib_iface.h"
#include "rpcmem.h"

#define MAX_BLOB (16 * 1024 * 1024)

static unsigned char g_batch[65536];
static unsigned char g_rsp[65536];

static long read_file(const char *path, void *dst, long cap) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    long n = (long) fread(dst, 1, (size_t) cap, f);
    fclose(f);
    return n;
}

int main(int argc, char **argv) {
    int want_unmapped = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--unmapped") == 0) want_unmapped = 1;
    }

    remote_handle64 h = 0;
    int rc = hexlib_iface_open(hexlib_iface_URI, &h);
    if (rc != 0) {
        printf("SIMHOST error=open rc=%d\n", rc);
        return 2;
    }

    rc = hexlib_iface_start(h, 1, 1, 1, (uint64) MAX_BLOB);
    if (rc != 0) {
        printf("SIMHOST error=start rc=%d\n", rc);
        return 3;
    }

    uint32 arch = 0, nthr = 0, nhvx = 0, nhmx = 0;
    uint64 vtcm = 0;
    rc = hexlib_iface_hwinfo(h, &arch, &nthr, &nhvx, &nhmx, &vtcm);
    if (rc != 0) {
        printf("SIMHOST error=hwinfo rc=%d\n", rc);
        return 4;
    }
    printf("SIMHOST hwinfo arch=%u threads=%u vtcm=%llu\n",
           (unsigned int) arch, (unsigned int) nthr, (unsigned long long) vtcm);

    long blen = read_file("hexlib_batch.bin", g_batch, (long) sizeof(g_batch));
    if (blen <= 0) {
        printf("SIMHOST error=no_batch\n");
        return 5;
    }

    /* The payload buffer. rpcmem gives an fd, which is the ONLY thing the skel
     * is told; it maps that fd itself and computes every address. */
    void *data = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, MAX_BLOB);
    if (!data) {
        printf("SIMHOST error=rpcmem_alloc\n");
        return 6;
    }
    long dlen = read_file("hexlib_in.bin", data, MAX_BLOB);
    if (dlen < 0) {
        printf("SIMHOST error=no_input\n");
        rpcmem_free(data);
        return 7;
    }
    int fd = rpcmem_to_fd(data);

    if (!want_unmapped) {
        rc = hexlib_iface_mmap(h, (uint32) fd, (uint32) MAX_BLOB);
        if (rc != 0) {
            printf("SIMHOST error=mmap rc=%d\n", rc);
            rpcmem_free(data);
            return 8;
        }
    } else {
        /* DELIBERATELY NOT MAPPED. The skel must refuse. If it returns a
         * result anyway, it read the host's address -- which works here and
         * would fail on silicon. This is the discriminator. */
        printf("SIMHOST note=fd_deliberately_unmapped\n");
    }

    /* The batch was built by the host with fd 0 as a placeholder; patch in the
     * real fd. Offsets are unchanged -- they are all this side ever sends. */
    struct hexlib_batch_hdr hdr;
    memcpy(&hdr, g_batch, sizeof(hdr));
    for (uint32_t i = 0; i < hdr.n_bufs; i++) {
        struct hexlib_buf_desc b;
        size_t off = hdr.off_bufs + i * sizeof(b);
        memcpy(&b, g_batch + off, sizeof(b));
        b.fd = (uint32_t) fd;
        b.base = 0;   /* never an address, on any path */
        memcpy(g_batch + off, &b, sizeof(b));
    }

    rc = hexlib_iface_invoke(h, g_batch, (int) blen, g_rsp, (int) sizeof(g_rsp));
    if (rc != 0) {
        printf("SIMHOST error=invoke rc=%d\n", rc);
        hexlib_iface_stop(h);
        hexlib_iface_close(h);
        rpcmem_free(data);
        return 9;
    }

    struct hexlib_batch_rsp_hdr rh;
    memcpy(&rh, g_rsp, sizeof(rh));
    uint64_t want = (uint64_t) sizeof(rh) +
                    (uint64_t) rh.n_ops * (uint64_t) sizeof(struct hexlib_op_result);
    uint32_t rsp_len = (want > sizeof(g_rsp)) ? (uint32_t) sizeof(g_rsp) : (uint32_t) want;

    printf("SIMHOST invoke rc=%d rsp_len=%u status=%u n_ops=%u cycles=%llu\n",
           rc, (unsigned int) rsp_len, (unsigned int) rh.status,
           (unsigned int) rh.n_ops, (unsigned long long) rh.cycles_total);

    FILE *rf = fopen("hexlib_rsp.bin", "wb");
    if (rf) { fwrite(g_rsp, 1, rsp_len, rf); fclose(rf); }

    if (rh.status == HEXLIB_DSP_OK) {
        FILE *of = fopen("hexlib_out.bin", "wb");
        if (of) { fwrite(data, 1, (size_t) dlen, of); fclose(of); }
    }

    hexlib_iface_stop(h);
    hexlib_iface_close(h);
    rpcmem_free(data);
    printf("SIMHOST done\n");
    return rh.status == HEXLIB_DSP_OK ? 0 : 1;
}

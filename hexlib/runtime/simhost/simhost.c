/* hexlib/runtime/simhost/simhost.c -- the host side, for the simulator.
 *
 * WHY THIS EXISTS. On a device the host is an aarch64 Android binary. On the
 * simulator there is no aarch64, so the "host" is Hexagon code in the same ELF
 * as the skel. That is the SDK's own BUILD_QEXES pattern (examples/calculator's
 * calculator_q), verified directly against that example at v75 on this
 * toolchain: it prints "Sum = 32640 / Pass: 2 Fail: 0" and exits 0.
 *
 * WHY THIS FILE CALLS hexlib_iface_open/start/mmap/invoke/stop/close DIRECTLY,
 * NOT THROUGH THE QAIC-GENERATED STUB. Reading calculator_q's own link line and
 * its generated calculator_stub.c/calculator_skel.c settled it: the generated
 * STUB (hexlib_iface_stub.c) defines the SAME function names
 * (hexlib_iface_open, _start, _mmap, _invoke, ...) as the DEVELOPER'S skel-side
 * implementation in skel.c -- on a device these live in two different ELFs
 * (host APK vs. DSP .so) so the names never collide, but statically linking
 * both into ONE qexe would be a duplicate-symbol error. calculator's own
 * hexagon.min never compiles calculator_stub.c into calculator_q either: only
 * the generated *_skel.c (an unused, harmless archive member here) and the
 * developer's *_imp.c (which implements calculator_open/_close/_sum/_max
 * directly) go into calculator_q's link. calculator_test.c's calls to
 * calculator_open/_sum resolve straight to that developer implementation --
 * there is no marshaling, no remote_handle64_open/_invoke, on this path at all
 * (confirmed by `hexagon-nm` on rtld.a/test_util.a/atomic.a: none of them
 * define remote_handle64_open/_close/_invoke). hexlib_q follows the same
 * shape: this file calls hexlib_iface_open/etc. as plain C functions, which
 * the linker binds directly to skel.c's definitions.
 *
 * IT SPEAKS THE PROTOCOL THAT ALREADY EXISTS. hexlib_in.bin / hexlib_out.bin,
 * the same files `hexlib/exec/hexagon.py` already writes and reads for the
 * standalone-ELF path -- host file I/O works in a standalone sim ELF and was
 * verified directly. So the acceptance test is one that already passes by
 * another route, and any difference is the new path's fault.
 *
 * THE ONE THING TO BE CAREFUL ABOUT. Host and DSP are one address space here.
 * This file must never hand the skel a pointer; it registers an fd with
 * rpcmem_alloc()+rpcmem_to_fd() and sends offsets, exactly as the device host
 * does. `--unmapped` exercises the negative case, which is the test that makes
 * a simulator pass transferable: it deliberately skips hexlib_iface_mmap, so
 * the skel must refuse (HEXLIB_DSP_ERR_UNMAPPED), not silently read the host's
 * address the way a shared-address-space bug would let it.
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

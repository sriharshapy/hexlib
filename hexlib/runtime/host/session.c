/* hexlib/runtime/host/session.c -- open a session on the CDSP, unsigned PD,
 * arch cross-checked rather than assumed.
 *
 * CDSP ONLY, NEVER ADSP. Measured on the target device (SM8650): CDSP is
 * domain 3 and reports UNSIGNED_PD_SUPPORT = 1. ADSP is a v73 part on the
 * same SoC and reports UNSIGNED_PD_SUPPORT = 0 -- opening it instead would
 * not fail, it would silently produce a measurement from different hardware.
 * hexlib_open (below) refuses any domain that is not CDSP_DOMAIN_ID outright.
 *
 * NO LITERAL REQUEST IDS. A wrong request-id constant does not fail loudly --
 * it queries something else on the DSP, or comes back with an error that
 * reads exactly like "unsupported", which is indistinguishable from the
 * answer a genuinely unsupported query would give. (This project hardcoded
 * DSPRPC_GET_DSP_INFO as 11 once, by counting an enum in a doc comment; the
 * real value, from <remote.h>'s own `enum handle_control_req_id`, is 2.)
 * Every request id and every capability attribute in this file is therefore
 * spelled by name from <remote.h> -- `enum handle_control_req_id`, `enum
 * remote_dsp_attributes`, `enum session_control_req_id` -- never typed as a
 * bare number.
 *
 * THE ARCH IS QUERIED, NEVER ASSUMED, ON BOTH SIDES OF THE WIRE, AND THE TWO
 * ARE CROSS-CHECKED. `hexlib_query_caps` asks the DRIVER what silicon this
 * is (ARCH_VER via DSPRPC_GET_DSP_INFO); `hexlib_iface_hwinfo` asks the SKEL
 * what it was compiled for (__HEXAGON_ARCH__, baked in at Task 7 build time).
 * The two must agree -- disagreement means the wrong skel .so is loaded for
 * this part, a version-skew bug, not a hardware fact -- so hexlib_open
 * fails rather than proceeding on a mismatched measurement.
 *
 * THE TWO SIDES ARE IN DIFFERENT ENCODINGS -- DECODE, NEVER COMPARE RAW.
 * `arch` (skel hwinfo) is plain decimal: __HEXAGON_ARCH__ is 75 on the
 * measured target (see skel.c, test_dsp_sim.py). `caps.arch_ver` (driver
 * ARCH_VER) is NOT the same number in the same base: on the identical part
 * it reads 35957 = 0x8c75 (see device/qdc/test_on_device.py, job.py's own
 * measured-facts header). The low byte packs the arch as two BCD digits --
 * 0x75 means digits 7 and 5, i.e. 75, not the integer 0x75 = 117 and
 * certainly not 35957. Comparing the raw values, as an earlier draft of
 * this file did, is unconditionally false for every real device: no session
 * could ever open. `hexlib_decode_bcd_arch` below does the same decode
 * llama.cpp's `htpdrv_get_arch` does (ggml-hexagon/htp-drv.cpp:412-413, MIT;
 * see ATTRIBUTION.md) -- `val = arch_ver & 0xff; arch = (val >> 4) * 10 +
 * (val & 0x0f)` -- and hexlib_open compares ITS output against `arch`, never
 * `caps.arch_ver` directly.
 */
#include "hexlib_host.h"

#include <remote.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hexlib_dsp.h"     /* struct hexlib_batch_rsp_hdr / hexlib_op_result */
#include "hexlib_iface.h"   /* qaic-generated from runtime/idl/hexlib_iface.idl */

int hexlib_query_caps(int domain, struct hexlib_caps *out) {
    memset(out, 0, sizeof(*out));

    struct {
        enum remote_dsp_attributes attr;
        uint32_t                  *dst;
    } queries[] = {
        { DOMAIN_SUPPORT,      &out->domain_support },
        { UNSIGNED_PD_SUPPORT, &out->unsigned_pd_support },
        { HVX_SUPPORT_128B,    &out->hvx_support_128b },
        { VTCM_PAGE,           &out->vtcm_page },
        { VTCM_COUNT,          &out->vtcm_count },
        { ARCH_VER,            &out->arch_ver },
        { HMX_SUPPORT_DEPTH,   &out->hmx_support_depth },
    };

    for (size_t i = 0; i < sizeof(queries) / sizeof(queries[0]); i++) {
        struct remote_dsp_capability cap;
        memset(&cap, 0, sizeof(cap));
        cap.domain       = (uint32_t) domain;
        cap.attribute_ID = (uint32_t) queries[i].attr;

        /* DSPRPC_GET_DSP_INFO, from <remote.h>'s own `enum
         * handle_control_req_id` -- see the file header. */
        int rc = hexlib_remote_handle_control(DSPRPC_GET_DSP_INFO, &cap, sizeof(cap));
        if (rc != 0) {
            fprintf(stderr,
                    "hexlib: DSPRPC_GET_DSP_INFO attribute %u failed (rc %d)\n",
                    (unsigned) queries[i].attr, rc);
            return -1;
        }
        *queries[i].dst = cap.capability;
    }
    return 0;
}

/* Must run BEFORE the handle is opened -- once the PD exists, it is already
 * signed or unsigned. Request id from <remote.h>'s `enum
 * session_control_req_id`, never a literal. */
static int enable_unsigned_pd(int domain) {
    struct remote_rpc_control_unsigned_module req;
    memset(&req, 0, sizeof(req));
    req.domain = domain;
    req.enable = 1;   /* Measured UNSIGNED_PD_SUPPORT = 1 on CDSP; see caller. */

    int rc = hexlib_remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                           &req, sizeof(req));
    if (rc != 0) {
        fprintf(stderr,
                "hexlib: DSPRPC_CONTROL_UNSIGNED_MODULE failed (rc %d)\n", rc);
    }
    return rc;
}

/* Pure BCD-nibble decode of the driver's ARCH_VER capability -- byte-for-byte
 * ported from llama.cpp's htpdrv_get_arch (ggml-hexagon/htp-drv.cpp:412-413,
 * MIT; see ATTRIBUTION.md). ARCH_VER's low byte packs the arch as two BCD
 * digits (0x8c75 -> low byte 0x75 -> nibbles 7 and 5 -> 75), not the plain
 * integer __HEXAGON_ARCH__ encodes -- see this file's own header comment for
 * why comparing the raw values can never agree. Kept as its own pure
 * function (no I/O, no globals, no side effects) so it can be extracted and
 * unit-tested directly against the one measured value this project has on
 * record (0x8c75 -> 75) rather than only asserted by source pattern -- see
 * hexlib/tests/test_session_arch_decode.py. */
static uint32_t hexlib_decode_bcd_arch(uint32_t arch_ver) {
    uint32_t val = arch_ver & 0xff;
    return (val >> 4) * 10 + (val & 0x0f);
}

int hexlib_open(hexlib_ctx **out, int domain) {
    *out = NULL;

    /* Refuse anything that is not CDSP outright. ADSP is a real, openable
     * domain on the same device -- opening it would not fail, it would
     * silently measure different hardware. See the file header. */
    if (domain != CDSP_DOMAIN_ID) {
        fprintf(stderr,
                "hexlib: refusing domain %d -- only CDSP_DOMAIN_ID (%d) is "
                "supported; ADSP is a v73 part and must never be substituted\n",
                domain, CDSP_DOMAIN_ID);
        return -1;
    }

    if (hexlib_drv_init() != 0) {
        return -1;
    }

    struct hexlib_caps caps;
    if (hexlib_query_caps(domain, &caps) != 0) {
        return -1;
    }
    if (!caps.unsigned_pd_support) {
        fprintf(stderr,
                "hexlib: CDSP reports UNSIGNED_PD_SUPPORT = 0 on this device "
                "(the measured target reports 1) -- refusing rather than "
                "silently taking a signed-PD path that has never been "
                "measured\n");
        return -1;
    }

    if (enable_unsigned_pd(domain) != 0) {
        return -1;
    }

    /* hexlib_iface_URI is qaic-generated (hexlib_iface.h); CDSP_DOMAIN is
     * <remote.h>'s own domain-suffix macro (&_dom=cdsp). Adjacent
     * string-literal concatenation -- neither half is retyped by hand, so
     * this is built, not a domain suffix written out by hand again here. */
    static const char uri[] = hexlib_iface_URI CDSP_DOMAIN;

    hexlib_ctx *ctx = (hexlib_ctx *) calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        return -1;
    }
    ctx->domain = domain;

    int rc = hexlib_iface_open(uri, &ctx->handle);
    if (rc != 0) {
        fprintf(stderr, "hexlib: hexlib_iface_open(%s) failed (rc %d)\n", uri, rc);
        free(ctx);
        return -1;
    }

    rc = hexlib_iface_start(ctx->handle, /* sess_id */ 0, /* n_hvx */ 0,
                            /* n_hmx */ 0, /* max_vmem: unbounded for now */ 0);
    if (rc != AEE_SUCCESS) {
        /* DECODED, NOT PRINTED RAW. `skel.c:58-72` tags a real DSP status into
         * the AEE return with HEXLIB_AEE_FROM_STATUS precisely so a VTCM
         * contention failure can be told apart from a signing failure or a
         * missing skel -- and until now NOTHING ON THE HOST READ IT. The macros
         * appeared only in the header and in a test probe, so the detail was on
         * the wire and every cause printed the same bare negative number, which
         * is the operator confusion the tag exists to remove. */
        if (HEXLIB_AEE_IS_STATUS(rc)) {
            fprintf(stderr,
                    "hexlib: hexlib_iface_start failed (rc %d) -- the DSP "
                    "reported status %d: %s\n",
                    rc, HEXLIB_AEE_STATUS(rc),
                    hexlib_dsp_status_name(HEXLIB_AEE_STATUS(rc)));
        } else {
            fprintf(stderr,
                    "hexlib: hexlib_iface_start failed (rc %d) -- no DSP status "
                    "tag, so this came from qaic or the RPC layer, not from the "
                    "skel's own code\n", rc);
        }
        hexlib_iface_close(ctx->handle);
        free(ctx);
        return -1;
    }

    uint32_t arch = 0, n_threads = 0, n_hvx = 0, n_hmx = 0;
    /* `uint64`, not `uint64_t`: hexlib_iface_hwinfo's qaic-generated
     * prototype (AEEStdDef.h's `unsigned __int64`) is a distinct type from
     * <stdint.h>'s uint64_t on an LP64 target even though both are 64 bits,
     * and passing the wrong one is a real pointer-type mismatch, not
     * pedantry -- confirmed by `-fsyntax-only` against the real generated
     * header (see the task report). */
    uint64 vtcm_size = 0;
    rc = hexlib_iface_hwinfo(ctx->handle, &arch, &n_threads, &n_hvx, &n_hmx, &vtcm_size);
    if (rc != AEE_SUCCESS) {
        fprintf(stderr, "hexlib: hexlib_iface_hwinfo failed (rc %d)\n", rc);
        hexlib_iface_stop(ctx->handle);
        hexlib_iface_close(ctx->handle);
        free(ctx);
        return -1;
    }

    /* CROSS-CHECK: the arch the DRIVER reports (queried above, from the CDSP
     * firmware itself) against the arch the SKEL reports (what THIS .so was
     * compiled for). See the file header -- disagreement is a version-skew
     * bug and must fail, not merely log.
     *
     * THE DRIVER'S VALUE IS DECODED FIRST -- see hexlib_decode_bcd_arch()
     * and this file's own header comment. Comparing `arch` against
     * `caps.arch_ver` directly (its raw, BCD-packed encoding) would be
     * unconditionally false on every real device -- e.g. 75 != 35957 -- and
     * every session would refuse before measuring anything. */
    uint32_t driver_arch = hexlib_decode_bcd_arch(caps.arch_ver);
    if (arch != driver_arch) {
        fprintf(stderr,
                "hexlib: arch mismatch -- driver ARCH_VER raw=%u (0x%04x) "
                "decodes to %u, skel hwinfo reports %u; refusing to run a "
                "mismatched binary\n",
                caps.arch_ver, caps.arch_ver, driver_arch, arch);
        hexlib_iface_stop(ctx->handle);
        hexlib_iface_close(ctx->handle);
        free(ctx);
        return -1;
    }

    ctx->arch      = arch;
    ctx->n_threads = n_threads;
    ctx->n_hvx     = n_hvx;
    ctx->n_hmx     = n_hmx;
    ctx->vtcm_size = vtcm_size;
    ctx->started   = 1;

    *out = ctx;
    return 0;
}

int hexlib_close(hexlib_ctx *ctx) {
    if (ctx == NULL) {
        return 0;
    }
    int rc = AEE_SUCCESS;
    if (ctx->started) {
        int src = hexlib_iface_stop(ctx->handle);
        if (src != AEE_SUCCESS) {
            rc = src;
        }
        ctx->started = 0;
    }
    int crc = hexlib_iface_close(ctx->handle);
    if (crc != 0 && rc == AEE_SUCCESS) {
        rc = crc;
    }
    free(ctx);
    return rc == AEE_SUCCESS ? 0 : -1;
}

int hexlib_hwinfo(hexlib_ctx *ctx, uint32_t *arch, uint32_t *n_threads,
                  uint32_t *n_hvx, uint32_t *n_hmx, uint64_t *vtcm_size) {
    *arch      = ctx->arch;
    *n_threads = ctx->n_threads;
    *n_hvx     = ctx->n_hvx;
    *n_hmx     = ctx->n_hmx;
    *vtcm_size = ctx->vtcm_size;
    return 0;
}

int hexlib_invoke(hexlib_ctx *ctx, const void *batch, size_t batch_len,
                  void *rsp, size_t rsp_cap, size_t *rsp_len) {
    *rsp_len = 0;
    if (!ctx->started) {
        return -1;
    }

    /* THE FIRST CODE IN THIS PROJECT TO EXERCISE QAIC'S REAL ARGUMENT
     * MARSHALLING. Every simulator run through Task 8 called skel.c's
     * hexlib_iface_invoke as a plain C function in the same address space
     * (see runtime/build.py's build_sim_so); the qaic stub was never
     * linked there. Here it is: this call goes through the generated
     * hexlib_iface_stub.c, which marshals `batch`/`result` into a
     * remote_arg[] and calls remote_handle64_invoke() for real. */
    int rc = hexlib_iface_invoke(ctx->handle, (const unsigned char *) batch,
                                 (int) batch_len, (unsigned char *) rsp,
                                 (int) rsp_cap);
    if (rc != AEE_SUCCESS) {
        return -1;
    }

    /* NO resultLenOut. `rout sequence<octet> result` marshals only a
     * CAPACITY (see hexlib_iface.h / ATTRIBUTION.md) -- there is no wire
     * channel back to the host for "how many bytes are meaningful". The
     * response is self-describing instead: hexlib_batch_rsp_hdr.n_ops says
     * how many hexlib_op_result entries follow. Compute the real length from
     * THAT, never from anything qaic handed back (it handed back nothing). */
    if (rsp_cap < sizeof(struct hexlib_batch_rsp_hdr)) {
        return -1;
    }
    struct hexlib_batch_rsp_hdr hdr;
    memcpy(&hdr, rsp, sizeof(hdr));
    size_t need = sizeof(hdr) + (size_t) hdr.n_ops * sizeof(struct hexlib_op_result);
    *rsp_len = need <= rsp_cap ? need : rsp_cap;
    return 0;
}

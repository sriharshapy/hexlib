/* hexlib/runtime/skel/hexlib_dsp.h -- the DSP-side contract.
 *
 * THE HOST WRITES FDS AND OFFSETS. THE DSP WRITES ADDRESSES. `hexlib_buf_desc.base`
 * and `hexlib_tensor.data` are scratch fields this side fills in; a host that put
 * an address in either would produce code that works on the simulator -- where
 * host and DSP share one address space -- and fails instantly on silicon. There is
 * no field on the wire for an address, which is what makes a simulator result
 * transferable.
 *
 * Adapted in shape from llama.cpp ggml-hexagon's htp-ops.h (MIT); see
 * ATTRIBUTION.md. Two things are deliberately NOT adapted: ne/nb strides (hexlib
 * uses an enumerated layout, so un-repacked weights are a plan-time error rather
 * than silent corruption, and strides cannot describe a VTCM-resident tile of a
 * DDR tensor) and the ggml opcode enum.
 */
#ifndef HEXLIB_DSP_H
#define HEXLIB_DSP_H

#include <stddef.h>
#include <stdint.h>

#define HEXLIB_BATCH_MAGIC   0x424C5848u   /* 'HXLB' little-endian */
#define HEXLIB_BATCH_VERSION 1

#define HEXLIB_MAX_BUFS 8
#define HEXLIB_MAX_SRC 6
#define HEXLIB_MAX_DST 4
#define HEXLIB_MAX_PARAMS 16
#define HEXLIB_MAX_TENSORS 512

/* OK IS 1, NOT 0. A response buffer that is never written is all zeros, and zero
 * must not read as success -- this project has been bitten four times by absence
 * reported as success, including a device-farm job that ran no tests and passed. */
enum hexlib_dsp_status {
    HEXLIB_DSP_OK = 1,
    HEXLIB_DSP_ERR_INTERNAL = 2,
    HEXLIB_DSP_ERR_BAD_MAGIC = 3,
    HEXLIB_DSP_ERR_BAD_VERSION = 4,
    HEXLIB_DSP_ERR_TRUNCATED = 5,
    HEXLIB_DSP_ERR_INVAL_PARAMS = 6,
    HEXLIB_DSP_ERR_UNMAPPED = 7,
    HEXLIB_DSP_ERR_NO_MMAP_SLOT = 8,
    HEXLIB_DSP_ERR_MMAP_FAILED = 9,
    HEXLIB_DSP_ERR_NO_KERNEL = 10,
    HEXLIB_DSP_ERR_VTCM_TOO_SMALL = 11,
    HEXLIB_DSP_ERR_VTCM_RECLAIMED = 12,
    HEXLIB_DSP_ERR_REQUIRES = 13,
    HEXLIB_DSP_ERR_NOT_STARTED = 14,
};

/* The status as text, for the one place a human reads it: the host's error
 * output. `switch` rather than a string array indexed by the value, so the
 * compiler warns on a status added to the enum and not to this, and so an
 * out-of-range value cannot index past the end.
 *
 * WHY THIS EXISTS AT ALL. `HEXLIB_AEE_FROM_STATUS` below tags a real DSP status
 * into an AEEResult so that a VTCM-contention failure can be told apart from a
 * signing failure or a missing skel -- and for a while nothing on the host
 * decoded it. `HEXLIB_AEE_IS_STATUS`/`HEXLIB_AEE_STATUS` appeared only in this
 * header and in a test probe, so the detail crossed the wire and every cause
 * printed the same bare negative number, which is exactly the operator
 * confusion the tag was added to remove. `session.c` decodes it now.
 *
 * The names match `hexlib.runtime.wire.STATUS` string for string, minus the
 * `HEXLIB_DSP_` prefix, and a compiled test binds the two tables. */
static inline const char *hexlib_dsp_status_name(int s) {
    switch (s) {
    case HEXLIB_DSP_OK:                 return "OK";
    case HEXLIB_DSP_ERR_INTERNAL:       return "ERR_INTERNAL";
    case HEXLIB_DSP_ERR_BAD_MAGIC:      return "ERR_BAD_MAGIC";
    case HEXLIB_DSP_ERR_BAD_VERSION:    return "ERR_BAD_VERSION";
    case HEXLIB_DSP_ERR_TRUNCATED:      return "ERR_TRUNCATED";
    case HEXLIB_DSP_ERR_INVAL_PARAMS:   return "ERR_INVAL_PARAMS";
    case HEXLIB_DSP_ERR_UNMAPPED:       return "ERR_UNMAPPED";
    case HEXLIB_DSP_ERR_NO_MMAP_SLOT:   return "ERR_NO_MMAP_SLOT";
    case HEXLIB_DSP_ERR_MMAP_FAILED:    return "ERR_MMAP_FAILED";
    case HEXLIB_DSP_ERR_NO_KERNEL:      return "ERR_NO_KERNEL";
    case HEXLIB_DSP_ERR_VTCM_TOO_SMALL: return "ERR_VTCM_TOO_SMALL";
    case HEXLIB_DSP_ERR_VTCM_RECLAIMED: return "ERR_VTCM_RECLAIMED";
    case HEXLIB_DSP_ERR_REQUIRES:       return "ERR_REQUIRES";
    case HEXLIB_DSP_ERR_NOT_STARTED:    return "ERR_NOT_STARTED";
    default:                            return "UNKNOWN";
    }
}

/* CARRY A STATUS OUT THROUGH AN AEEResult, for the one call that has no response
 * buffer to put it in. `invoke` returns its status inside the response blob, but
 * `start` fails before any blob exists, so a bare AEE_EFAILED there flattened
 * every VTCM outcome into the single result the host already prints for a dozen
 * unrelated causes -- leaving a device operator unable to tell VTCM contention
 * from a signing failure, a URI error, or a missing skel.
 *
 * 0x8FA0xxxx sits in the AEE "reserved for OEM/vendor" high half, so it is
 * nonzero (every `if (rc != AEE_SUCCESS)` still fails) and does not collide with
 * AEE_EFAILED or the AEE_E* range. Decode with HEXLIB_AEE_STATUS; test with
 * HEXLIB_AEE_IS_STATUS first, because a failure from qaic or the RPC layer
 * itself will not carry this tag. */
#define HEXLIB_AEE_STATUS_TAG 0x8FA00000u
#define HEXLIB_AEE_FROM_STATUS(s) ((int) (HEXLIB_AEE_STATUS_TAG | ((unsigned) (s) & 0xFFu)))
#define HEXLIB_AEE_IS_STATUS(r) (((unsigned) (r) & 0xFFFFFF00u) == HEXLIB_AEE_STATUS_TAG)
#define HEXLIB_AEE_STATUS(r) ((int) ((unsigned) (r) & 0xFFu))

struct hexlib_batch_hdr {
    uint32_t magic;
    uint32_t version;
    uint32_t total_size;
    uint32_t n_bufs;
    uint32_t n_tensors;
    uint32_t n_ops;
    uint32_t off_bufs;
    uint32_t off_tensors;
    uint32_t off_ops;
    uint32_t flags;
};

struct hexlib_buf_desc {
    uint64_t base;   /* DSP-SIDE SCRATCH. Host writes 0. */
    uint64_t size;
    uint32_t fd;
    uint32_t flags;
};

/* THE ENUMERATED LAYOUT, spelled once. These MUST equal
 * hexlib.runtime.wire.LAYOUT_ID, which is what the host serializes through, and
 * hexlib/tests/test_wire_struct_layout.py compares the two tables so a value
 * added on one side cannot drift from the other.
 *
 * Named rather than left as bare integers because `tens[i].layout = 0` in
 * main.c's hand-built batch was a literal 0 with only a comment tying it to
 * `LAYOUT_ID["row_major"]` -- and unlike the `dtype` literal beside it, which
 * genentry's emitted `a->dtype[..] != 1u` guard catches loudly at run time,
 * nothing checked layout at all. Inserting a layout ahead of row_major would
 * have left `pack_batch` emitting 1 while main.c kept emitting 0, and
 * `--self-test` would still print `PASS (4100 values, bit-exact)` while
 * declaring the buffer as something else entirely. Latent only because no
 * kernel reads `a->layout` YET: `q4_0_repacked` is the matmul weight layout,
 * and the whole reason this field is an enum rather than ne/nb strides is so
 * that un-repacked weights are a plan-time error and not silent corruption. */
#define HEXLIB_LAYOUT_ROW_MAJOR    0u
#define HEXLIB_LAYOUT_TILED_32X32  1u
#define HEXLIB_LAYOUT_Q4_0_REPACKED 2u

struct hexlib_tensor {
    uint32_t bi;
    uint32_t offset;
    uint32_t nbytes;
    uint32_t dtype;
    uint32_t layout;
    uint32_t ne[4];
    uint32_t data;   /* DSP-SIDE SCRATCH. Host writes 0. */
    uint32_t pad;
};

struct hexlib_op_desc {
    uint32_t kind;
    uint32_t flags;
    int32_t  params[HEXLIB_MAX_PARAMS];
    uint16_t src[HEXLIB_MAX_SRC];
    uint16_t dst[HEXLIB_MAX_DST];
};

struct hexlib_batch_rsp_hdr {
    uint32_t magic;
    uint32_t version;
    uint32_t status;
    uint32_t n_ops;
    uint64_t cycles_total;
    uint32_t arch;
    uint32_t pad;
};

struct hexlib_op_result {
    uint32_t kind;
    uint32_t status;
    uint64_t cycles;
};

/* The kernel ABI. `int`, not `void`: a kernel that cannot serve a request says so
 * rather than producing a plausible wrong answer. */
typedef struct {
    void       *buf[HEXLIB_MAX_BUFS];
    uint32_t    ne[HEXLIB_MAX_BUFS][4];
    uint32_t    dtype[HEXLIB_MAX_BUFS];
    uint32_t    layout[HEXLIB_MAX_BUFS];
    uint32_t    n_buf;
    uint8_t    *vtcm;
    size_t      vtcm_size;
    const void *params;
    uint32_t    n_threads;
} hexlib_args;

typedef int (*hexlib_kernel_fn)(const hexlib_args *);

struct hexlib_kernel_entry {
    uint32_t          kind;
    const char       *name;
    hexlib_kernel_fn  fn;
};

/* Generated by hexlib/runtime/genentry.py. */
extern const struct hexlib_kernel_entry hexlib_kernel_table[];
extern const uint32_t hexlib_kernel_table_len;

#endif /* HEXLIB_DSP_H */

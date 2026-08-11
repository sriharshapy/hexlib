# hexlib/tests/test_genentry_entry_probe.py
"""BEHAVIOURAL test of the GENERATED DSP entry points: compile them with a host
C compiler, call them with a hand-built `hexlib_args`, and check what they
return.

WHY THIS EXISTS RATHER THAN MORE SOURCE ASSERTIONS. test_runtime_genentry.py
checks the emitted TEXT ("this `if` is present, in this order"). Text cannot
answer the only question that matters about a guard: does it fire? The output
dtype check has been emitted since the generator was written and CANNOT fire
through `hexlib/exec/dsp.py`, because that serializer fills the field it reads
from the same `spec.out_dtype` the check was generated from -- a check that
compares a constant with itself. So "the `if` is there" and "the guard works"
are genuinely different claims here, and the new per-input dtype checks needed
the second one.

The proof is offline. Same recipe as test_wire_struct_layout.py,
test_session_arch_decode.py and test_coherency_lane_classification.py: emit a
small C program, compile it with a host `cc`, RUN it, and compare real returned
values against what Python expects. No SDK, no simulator, no device -- and
`hexagon-sim` could not answer this question much better anyway, since driving a
deliberately-wrong dtype through it needs a hand-built blob either way.

WHAT IS REAL HERE AND WHAT IS A STAND-IN. Real: the entry source, verbatim from
`genentry.emit_entry`, and `hexlib_dsp.h` itself (so `hexlib_args`'s real layout,
the real `dtype[]` array, and the real status enum). A stand-in: `kernel_api.h`,
written below, which typedefs `hexlib_hf` as `unsigned short` and declares the
three kernel prototypes. The real per-kernel headers typedef it as `__fp16`,
which mainstream x86 gcc does not accept -- and nothing here depends on the
type's arithmetic, only on the entry's control flow before the call. The kernels
themselves are recording stubs, because "the kernel was not called" is half of
every assertion below.

TWO PROBES, AND THE SECOND ONE EXISTS BECAUSE THE FIRST'S CLAIM WAS HALF TRUE.
`test_buffers_are_packed_sources_then_destinations` says the src-then-dst
contract is "PINNED FROM BOTH SIDES AT ONCE". It was not: only `genentry`'s
generated entries were compiled, so it pinned `genentry`'s `out_idx = n_in` and
nothing else. `skel_dispatch.c`'s fill loop -- the OTHER side, and the only
statement anywhere that `a->buf[]` holds sources followed by destinations -- was
never built here, and INVERTING IT (destinations packed first) left this file at
6 passed. The claim was written by the same hand that wrote the test and it was
simply wrong.

So there is now a second fixture, `both_sides`, that compiles the REAL
`skel_dispatch.c` and `skel_bufs.c` together with a generated entry and drives a
REAL batch blob from `hexlib.runtime.wire.pack_batch` through
`hexlib_dispatch_batch`. Nothing about the ordering is asserted textually: the
recording kernel reports which ADDRESS it received as its input and which as its
output, and those addresses are checked against the fd-plus-offset arithmetic the
batch declared. Inverting either side -- the fill loop or `out_idx` -- makes the
kernel receive the other tensor and fails it. That is what pinning both sides at
once means, and it also makes this the one offline test that drives the host
serializer, the DSP-side fd-to-address mapping, the dispatcher and a generated
entry in a single run.

THE STANDING-IN GOES ONE LEVEL FURTHER FOR THAT SECOND PROBE, and here is
exactly how far. Stubbed: `HAP_farf.h` (FARF discards its arguments),
`HAP_perf.h` (a monotonic counter), `HAP_mem.h` (HAP_mmap2 returns a small
distinct fake address per fd, so `hexlib_tensor.data` -- a uint32_t -- can hold
it), and `hexlib_vtcm_acquire`/`hexlib_vtcm_release`, which this probe is not
about. Real and compiled from the repo: skel_dispatch.c, skel_bufs.c,
hexlib_dsp.h, skel_internal.h, the generated entry. The addresses are never
dereferenced -- the kernel stub records the pointer and returns -- so a fake
mapping is enough to make identity meaningful, which is the only property under
test.
"""
import pathlib
import re
import shutil
import struct
import subprocess

import pytest

from hexlib.exec import runner as rn
from hexlib.runtime import genentry as ge
from hexlib.runtime import wire
from hexlib.runtime.wire import DTYPE_ID, STATUS

SKEL = pathlib.Path("hexlib/runtime/skel")

HOST_CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
needs_cc = pytest.mark.skipif(
    HOST_CC is None,
    reason=(
        "no host C compiler found (tried: cc, gcc, clang); the generated "
        "entries' dtype guards are then covered only by the source assertions "
        "in test_runtime_genentry.py, which can see that an `if` was emitted "
        "but not that it fires. Install a host C compiler to restore this."
    ),
)

# A stand-in kernel_api.h -- see the module docstring on why hexlib_hf is not
# __fp16 here.
KERNEL_API_H = """\
#ifndef PROBE_KERNEL_API_H
#define PROBE_KERNEL_API_H
typedef unsigned short hexlib_hf;
void scale_fp16(const hexlib_hf *x, hexlib_hf *y, int n, float factor);
void cast_f32_f16(const float *x, hexlib_hf *y, int n);
void add_fp16(const hexlib_hf *a, const hexlib_hf *b, hexlib_hf *y, int n);
#endif
"""

PROBE_C = """\
#include "hexlib_dsp.h"
#include "kernel_api.h"
#include <stdio.h>
#include <string.h>

/* Recording stubs. "the kernel was NOT called" is half of every assertion. */
static int g_calls;
static int g_n;
static float g_factor;

/* WHICH BUFFER EACH ARGUMENT ACTUALLY WAS. The src-then-dst packing contract is
 * stated in exactly one place -- skel_dispatch.c's fill loop -- and mirrored by
 * genentry's `out_idx = n_in`; nothing checked that the two agree. Recording the
 * pointers turns an "outputs first" refactor from a silent swap into a failure. */
static const void *g_in0;
static const void *g_in1;
static const void *g_out;

void scale_fp16(const hexlib_hf *x, hexlib_hf *y, int n, float factor) {
    g_calls++; g_n = n; g_factor = factor; g_in0 = x; g_out = y;
}
void cast_f32_f16(const float *x, hexlib_hf *y, int n) {
    g_calls++; g_n = n; g_in0 = x; g_out = y;
}
void add_fp16(const hexlib_hf *a, const hexlib_hf *b, hexlib_hf *y, int n) {
    g_calls++; g_n = n; g_in0 = a; g_in1 = b; g_out = y;
}

extern int scale_fp16_entry(const hexlib_args *);
extern int cast_f32_f16_entry(const hexlib_args *);
extern int add_fp16_entry(const hexlib_args *);

static unsigned char b0[4096], b1[4096], b2[4096];
static float params[4] = { 0.125f, 0.0f, 0.0f, 0.0f };

/* Every buffer supplied, every extent 17 (so the derived n is checkable and is
 * not 0 by accident), params holding the scale factor. Each case then breaks
 * exactly one thing. */
static void base(hexlib_args *a, unsigned int n_buf) {
    memset(a, 0, sizeof(*a));
    a->n_buf = n_buf;
    a->buf[0] = b0; a->buf[1] = b1; a->buf[2] = b2;
    for (unsigned int i = 0; i < HEXLIB_MAX_BUFS; i++) {
        a->ne[i][0] = 17; a->ne[i][1] = 1; a->ne[i][2] = 1; a->ne[i][3] = 1;
    }
    a->params = params;
}

/* putchar(10) emits the newline: this C is carried inside a Python
 * string literal, so an escape sequence here round-trips badly. */
static void report_order(const char *label, int ok) {
    printf("%s=%d", label, ok ? 1 : 0);
    putchar(10);
}

static void report(const char *label, int rc) {
    printf("case=%s rc=%d calls=%d n=%d\\n", label, rc, g_calls, g_n);
}

int main(void) {
    hexlib_args a;
    int rc;

    g_calls = 0; g_n = -1;
    base(&a, 2); a.dtype[0] = ID_FP16; a.dtype[1] = ID_FP16;
    rc = scale_fp16_entry(&a);
    report("scale_ok", rc);
    report_order("scale_order",
                 g_in0 == (const void *) b0 && g_out == (const void *) b1);
    printf("factor_ok=%d\\n", g_factor == 0.125f ? 1 : 0);

    g_calls = 0; g_n = -1;
    base(&a, 2); a.dtype[0] = ID_FP32; a.dtype[1] = ID_FP16;
    rc = scale_fp16_entry(&a);
    report("scale_input_fp32", rc);

    g_calls = 0; g_n = -1;
    base(&a, 2); a.dtype[0] = ID_FP16; a.dtype[1] = ID_FP32;
    rc = scale_fp16_entry(&a);
    report("scale_output_fp32", rc);

    g_calls = 0; g_n = -1;
    base(&a, 2); a.dtype[0] = ID_FP32; a.dtype[1] = ID_FP16;
    rc = cast_f32_f16_entry(&a);
    report("cast_ok", rc);

    g_calls = 0; g_n = -1;
    base(&a, 2); a.dtype[0] = ID_FP16; a.dtype[1] = ID_FP16;
    rc = cast_f32_f16_entry(&a);
    report("cast_input_fp16", rc);

    g_calls = 0; g_n = -1;
    base(&a, 3); a.dtype[0] = ID_FP16; a.dtype[1] = ID_FP16; a.dtype[2] = ID_FP16;
    rc = add_fp16_entry(&a);
    report("add_ok", rc);
    report_order("add_order",
                 g_in0 == (const void *) b0 && g_in1 == (const void *) b1
                 && g_out == (const void *) b2);

    g_calls = 0; g_n = -1;
    base(&a, 3); a.dtype[0] = ID_FP16; a.dtype[1] = ID_FP32; a.dtype[2] = ID_FP16;
    rc = add_fp16_entry(&a);
    report("add_second_input_fp32", rc);

    g_calls = 0; g_n = -1;
    base(&a, 1); a.dtype[0] = ID_FP16; a.dtype[1] = ID_FP16;
    rc = scale_fp16_entry(&a);
    report("scale_one_buffer", rc);

    g_calls = 0; g_n = -1;
    base(&a, 2); a.dtype[0] = ID_FP16; a.dtype[1] = ID_FP16; a.buf[1] = 0;
    rc = scale_fp16_entry(&a);
    report("scale_null_output", rc);

    return 0;
}
"""

CASE_RE = re.compile(r"case=(\S+) rc=(-?\d+) calls=(-?\d+) n=(-?\d+)")


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    """Compile the real generated entries plus the probe, run it once, and
    return {label: (rc, calls, n)} together with the factor flag.

    The dtype IDS are passed in with -D from `wire.DTYPE_ID` rather than typed
    into the C, so this cannot silently test a stale table."""
    if HOST_CC is None:
        pytest.skip("no host C compiler")
    d = tmp_path_factory.mktemp("entryprobe")
    (d / "kernel_api.h").write_text(KERNEL_API_H)
    (d / "probe.c").write_text(PROBE_C)

    sources = [str(d / "probe.c")]
    for kind in ("scale", "cast", "add"):
        spec = rn.SPECS[kind]
        name = spec.kernel_dir.split("/")[-1]
        path = d / f"{name}_entry.c"
        path.write_text(ge.emit_entry(kind, spec))
        sources.append(str(path))

    exe = str(d / "probe.exe")
    cmd = [
        HOST_CC, "-std=c11", "-O0",
        f"-DID_FP16={DTYPE_ID['fp16']}u", f"-DID_FP32={DTYPE_ID['fp32']}u",
        "-I", str(d), "-I", str(SKEL.resolve()),
        *sources, "-o", exe,
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    assert cp.returncode == 0, f"probe did not compile:\n{cp.stderr}"
    run = subprocess.run([exe], capture_output=True, text=True)
    assert run.returncode == 0, f"probe crashed:\n{run.stdout}\n{run.stderr}"

    cases = {
        m.group(1): (int(m.group(2)), int(m.group(3)), int(m.group(4)))
        for m in CASE_RE.finditer(run.stdout)
    }
    assert len(cases) == 9, f"probe printed {sorted(cases)}:\n{run.stdout}"
    cases["_factor_ok"] = ("factor_ok=1" in run.stdout, 0, 0)
    cases["_scale_order"] = ("scale_order=1" in run.stdout, 0, 0)
    cases["_add_order"] = ("add_order=1" in run.stdout, 0, 0)
    return cases


@needs_cc
def test_a_well_formed_request_reaches_the_kernel(probe):
    """The control, and it has to come first: every refusal below is only
    meaningful because the accepted case is accepted. `n` proves the extent is
    derived from `ne` on the DSP side, and the factor proves the params blob is
    read as float bits and not as an int."""
    for label in ("scale_ok", "cast_ok", "add_ok"):
        rc, calls, n = probe[label]
        assert rc == STATUS["OK"], f"{label} returned {rc}"
        assert calls == 1, f"{label} did not call its kernel"
        assert n == 17, f"{label} derived n={n} from ne, expected 17"
    assert probe["_factor_ok"][0], "the attr scalar did not arrive as 0.125f"


@needs_cc
def test_an_input_declared_with_the_wrong_dtype_is_refused_before_the_cast(probe):
    """THE FINDING, PROVEN BY RETURN VALUE. `scale_fp16` casts buf[0] to
    `const hexlib_hf *`. A batch declaring that buffer fp32 -- reachable from
    `run_raw`, from main.c's hand-built blob, or from a future planner -- used to
    be read at half stride: half the tensor, HEXLIB_DSP_OK, plausible values.
    Now the entry refuses it and never calls the kernel.

    THIS IS THE FALSIFIABLE ONE. Delete the input dtype check from
    `genentry._dtype_check`/`emit_entry` and this test fails with rc=1 and
    calls=1 -- unlike the output check, which no serializer can violate."""
    rc, calls, _ = probe["scale_input_fp32"]
    assert rc == STATUS["ERR_REQUIRES"], (
        f"an fp32 buffer cast to hexlib_hf* returned {rc}, not ERR_REQUIRES"
    )
    assert calls == 0, "the kernel ran on a buffer of the wrong dtype"

    rc, calls, _ = probe["cast_input_fp16"]
    assert rc == STATUS["ERR_REQUIRES"], f"cast's fp32 input check returned {rc}"
    assert calls == 0


@needs_cc
def test_the_dtype_check_is_per_buffer_not_just_the_first_one(probe):
    """`add` has two inputs. A guard written for buf[0] alone would leave the
    right-hand operand -- the one carrying the learned pos_embed in this
    graph -- unchecked."""
    rc, calls, _ = probe["add_second_input_fp32"]
    assert rc == STATUS["ERR_REQUIRES"], (
        f"add's SECOND input was cast without a dtype check (rc={rc})"
    )
    assert calls == 0


@needs_cc
def test_an_output_declared_with_the_wrong_dtype_is_refused_too(probe):
    """The same hazard on the write side: an fp32 output buffer written through
    `hexlib_hf *` gets half of it filled and half left as whatever was there.
    Unreachable through `hexlib/exec/dsp.py` (which stamps this field from
    `spec.out_dtype`), reachable from any hand-built batch -- which is exactly
    what this probe is."""
    rc, calls, _ = probe["scale_output_fp32"]
    assert rc == STATUS["ERR_REQUIRES"], f"returned {rc}"
    assert calls == 0


@needs_cc
def test_the_structural_checks_still_come_first(probe):
    """A dtype check on a buffer the batch never supplied would be reading
    uninitialised `hexlib_args` fields. The count and null checks must still be
    the ones that answer these, with their own distinct status."""
    rc, calls, _ = probe["scale_one_buffer"]
    assert rc == STATUS["ERR_INVAL_PARAMS"], f"n_buf=1 returned {rc}"
    assert calls == 0
    rc, calls, _ = probe["scale_null_output"]
    assert rc == STATUS["ERR_INVAL_PARAMS"], f"a null output returned {rc}"
    assert calls == 0


@needs_cc
def test_buffers_are_packed_sources_then_destinations(probe):
    """THE CONTRACT, PINNED FROM BOTH SIDES AT ONCE.

    `skel_dispatch.c`'s fill loop is the only statement anywhere that `a->buf[]`
    holds sources followed by destinations, and `genentry.py` hardcodes the
    mirror image as `out_idx = n_in`. Neither referenced the other and no test
    compared them, so an "outputs first" refactor could swap input and output
    pointers in every generated entry at once: `scale_fp16` would write into its
    own input and return the zero-filled output region, at the right length,
    with status OK. Only the @sdk-gated numeric test would have noticed, and CI
    does not run it.

    This checks the POINTERS the kernel actually received, so it fails on the
    swap rather than on the spelling of any particular index expression. b0/b1/b2
    are distinct static arrays, which is what makes identity meaningful.

    ONE SIDE, HONESTLY LABELLED. The heading above used to say both sides were
    pinned here. They were not: this probe hands `hexlib_args` to the entry
    DIRECTLY, so it pins `genentry`'s half of the contract -- that the entry
    reads its inputs from indices 0..n_in-1 and its output from index n_in -- and
    nothing about how `a->buf[]` came to be filled. `skel_dispatch.c`'s fill loop
    was not compiled by this file at all, and inverting it left every test here
    passing. `test_the_dispatcher_and_the_generated_entry_agree_on_src_then_dst`
    below is the other side, and the two together are what the heading claims."""
    assert probe["_scale_order"][0], (
        "scale_fp16 must receive buf[0] as its input and buf[1] as its output "
        "(1 source, then 1 destination)"
    )
    assert probe["_add_order"][0], (
        "add_fp16 must receive buf[0] and buf[1] as its two inputs and buf[2] as "
        "its output -- the destination sits at index n_in, not index 0"
    )


# ==============================================================================
# THE OTHER SIDE OF THE SAME CONTRACT: skel_dispatch.c's fill loop, compiled and
# driven with a real batch blob. See the module docstring for what is stubbed.
# ==============================================================================

_STUB_HEADERS = {
    # FARF discards its arguments: nothing here reads a device log.
    "HAP_farf.h": (
        "#ifndef HEXLIB_PROBE_HAP_FARF_H\n"
        "#define HEXLIB_PROBE_HAP_FARF_H\n"
        "#define FARF(...) do { } while (0)\n"
        "#endif\n"
    ),
    # A monotonic counter, so the PCYCLE bracket produces a nonzero delta and
    # the response's cycles_total is checkable without a real counter.
    "HAP_perf.h": (
        "#ifndef HEXLIB_PROBE_HAP_PERF_H\n"
        "#define HEXLIB_PROBE_HAP_PERF_H\n"
        "static unsigned long long hexlib_probe_pcycles;\n"
        "static inline unsigned long long HAP_perf_get_pcycles(void) {\n"
        "    hexlib_probe_pcycles += 1287; return hexlib_probe_pcycles;\n"
        "}\n"
        "#endif\n"
    ),
    # A distinct small fake address per fd. SMALL ON PURPOSE: hexlib_tensor.data
    # is a uint32_t, so a real 64-bit host address would be truncated by
    # skel_bufs.c's own (uint32_t) cast and identity would stop meaning
    # anything. Nothing dereferences these.
    "HAP_mem.h": (
        "#ifndef HEXLIB_PROBE_HAP_MEM_H\n"
        "#define HEXLIB_PROBE_HAP_MEM_H\n"
        "#include <stddef.h>\n"
        "#define HAP_PROT_READ 1\n"
        "#define HAP_PROT_WRITE 2\n"
        "#define HEXLIB_PROBE_BASE(fd) "
        "(0x01000000u + 0x00010000u * (unsigned) (fd))\n"
        "static inline void *HAP_mmap2(void *a, size_t l, int p, int f,\n"
        "                              int fd, long o) {\n"
        "    (void) a; (void) l; (void) p; (void) f; (void) o;\n"
        "    return (void *) (size_t) HEXLIB_PROBE_BASE(fd);\n"
        "}\n"
        "static inline void *HAP_mmap(void *a, int l, int p, int f,\n"
        "                             int fd, long o) {\n"
        "    (void) a; (void) l; (void) p; (void) f; (void) o;\n"
        "    return (void *) (size_t) HEXLIB_PROBE_BASE(fd);\n"
        "}\n"
        "static inline int HAP_munmap2(void *a, size_t l) {\n"
        "    (void) a; (void) l; return 0;\n"
        "}\n"
        "static inline int HAP_munmap(void *a, int l) {\n"
        "    (void) a; (void) l; return 0;\n"
        "}\n"
        "#endif\n"
    ),
}

# Only the one kernel this probe drives, so `kernel_api.h` above is not reused
# (it declares three).
_DISPATCH_KERNEL_API_H = """\
#ifndef PROBE_DISPATCH_KERNEL_API_H
#define PROBE_DISPATCH_KERNEL_API_H
typedef unsigned short hexlib_hf;
void scale_fp16(const hexlib_hf *x, hexlib_hf *y, int n, float factor);
#endif
"""

_DISPATCH_PROBE_C = r"""
#include "skel_internal.h"
#include "kernel_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Recording stub. The addresses are never dereferenced -- WHICH buffer arrived
 * where is the entire question. */
static int g_calls;
static const void *g_in0;
static const void *g_out;
static int g_n;
static float g_factor;

void scale_fp16(const hexlib_hf *x, hexlib_hf *y, int n, float factor) {
    g_calls++; g_in0 = x; g_out = y; g_n = n; g_factor = factor;
}

extern int scale_fp16_entry(const hexlib_args *);

/* The real table shape from hexlib_dsp.h, with the one generated entry. */
const struct hexlib_kernel_entry hexlib_kernel_table[] = {
    { PROBE_KIND_SCALE, "scale_fp16", scale_fp16_entry },
};
const uint32_t hexlib_kernel_table_len = 1;

/* Not what this probe is about; skel_vtcm.c needs the real HAP_compute_res. */
int hexlib_vtcm_acquire(struct hexlib_ctx *c) { (void) c; return HEXLIB_DSP_OK; }
void hexlib_vtcm_release(struct hexlib_ctx *c) { (void) c; }

static struct hexlib_ctx g_ctx;
static unsigned char g_batch[65536];
/* uint64-aligned: the dispatcher casts rsp + sizeof(hdr) to
 * struct hexlib_op_result *, which contains a uint64_t. */
static unsigned long long g_rsp[1024];

int main(int argc, char **argv) {
    FILE *f;
    size_t len;
    int i;
    uint32_t rsp_len = 0;
    int rc;

    if (argc < 3) return 2;
    f = fopen(argv[1], "rb");
    if (!f) return 3;
    len = fread(g_batch, 1, sizeof(g_batch), f);
    fclose(f);

    memset(&g_ctx, 0, sizeof(g_ctx));
    /* Register every fd the batch names, exactly as hexlib_iface_mmap would. */
    for (i = 2; i < argc; i++) {
        unsigned fd = (unsigned) strtoul(argv[i], 0, 10);
        printf("register fd=%u rc=%d\n", fd,
               hexlib_bufs_register(&g_ctx, fd, PROBE_BUF_SIZE));
    }
    /* The DSP-side mapping table, so the expected addresses are read out of the
     * real skel_bufs.c state rather than recomputed by the test. */
    for (i = 0; i < HEXLIB_MAX_MMAPS; i++) {
        if (g_ctx.mmap[i].size) {
            printf("mapped fd=%d base=%llu\n", (int) g_ctx.mmap[i].fd,
                   (unsigned long long) g_ctx.mmap[i].base);
        }
    }

    g_ctx.started = 1;
    rc = hexlib_dispatch_batch(&g_ctx, g_batch, (uint32_t) len,
                               (unsigned char *) g_rsp,
                               (uint32_t) sizeof(g_rsp), &rsp_len);
    printf("dispatch rc=%d rsp_len=%u calls=%d n=%d factor_ok=%d\n",
           rc, rsp_len, g_calls, g_n, g_factor == 0.125f ? 1 : 0);
    printf("in0=%llu out=%llu\n",
           (unsigned long long) (size_t) g_in0,
           (unsigned long long) (size_t) g_out);
    printf("rsp=");
    for (i = 0; i < (int) rsp_len; i++) {
        printf("%02x", ((unsigned char *) g_rsp)[i]);
    }
    printf("\n");
    return 0;
}
"""

# Distinct fds and distinct NONZERO offsets: the input's address and the
# output's must be different numbers for identity to prove anything, and an
# offset of 0 on both would make a base-only bug invisible.
_FD_IN, _FD_OUT = 11, 22
_OFF_IN, _OFF_OUT = 128, 256
_BUF_SIZE = 4096
_NE = (17, 1, 1, 1)
_FACTOR = 0.125  # a power of two, exact in fp16 -- same value run_self_test uses


@pytest.fixture(scope="module")
def both_sides(tmp_path_factory):
    """Compile skel_dispatch.c + skel_bufs.c + a generated entry, build a real
    batch with `wire.pack_batch`, run it through `hexlib_dispatch_batch`, and
    return what the kernel saw plus the raw response bytes."""
    if HOST_CC is None:
        pytest.skip("no host C compiler")
    d = tmp_path_factory.mktemp("dispatchprobe")
    for name, text in _STUB_HEADERS.items():
        (d / name).write_text(text)
    (d / "kernel_api.h").write_text(_DISPATCH_KERNEL_API_H)
    (d / "probe.c").write_text(_DISPATCH_PROBE_C)
    (d / "scale_entry.c").write_text(ge.emit_entry("scale", rn.SPECS["scale"]))

    kind = ge.KIND_ID["scale"]
    exe = str(d / "probe.exe")
    cmd = [
        HOST_CC, "-std=c11", "-O0",
        f"-DPROBE_KIND_SCALE={kind}u", f"-DPROBE_BUF_SIZE={_BUF_SIZE}u",
        # The arch this "binary" was built for (hexlib_write_rsp_hdr reads it)
        # and the HVX level that selects skel_bufs.c's HAP_mmap2 branch -- the
        # v75 branch, which is the one that runs on the target part.
        "-D__HEXAGON_ARCH__=75", "-D__HVX_ARCH__=75",
        "-I", str(d), "-I", str(SKEL.resolve()),
        str(d / "probe.c"), str(d / "scale_entry.c"),
        str((SKEL / "skel_dispatch.c").resolve()),
        str((SKEL / "skel_bufs.c").resolve()),
        "-o", exe,
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    assert cp.returncode == 0, (
        "the skel dispatch probe did not compile. THIS FIXTURE IMPOSES A REAL "
        "CONSTRAINT AND THAT IS DELIBERATE: skel_dispatch.c and skel_bufs.c must "
        "stay buildable by a plain host C compiler, i.e. straight-line C with no "
        "Hexagon intrinsics and no inline asm, with every SDK dependency behind "
        "one of the stubbed headers above. That is already true and is worth "
        "keeping -- it is the same property that makes the cycle counter go "
        "through HAP_perf_get_pcycles() rather than a hand-rolled `c15:14` read. "
        "If a kernel-side intrinsic genuinely belongs in one of these two files, "
        "it needs to move behind a helper this probe can stub, not be absorbed "
        f"by deleting this test:\n{cp.stderr}"
    )

    bufs = [wire.BufDesc(fd=_FD_IN, size=_BUF_SIZE),
            wire.BufDesc(fd=_FD_OUT, size=_BUF_SIZE)]
    nbytes = 2 * _NE[0]
    tensors = [
        wire.TensorDesc(bi=0, offset=_OFF_IN, nbytes=nbytes, dtype="fp16",
                        layout="row_major", ne=_NE),
        wire.TensorDesc(bi=1, offset=_OFF_OUT, nbytes=nbytes, dtype="fp16",
                        layout="row_major", ne=_NE),
    ]
    factor_bits = struct.unpack("<i", struct.pack("<f", _FACTOR))[0]
    ops = [wire.OpDesc(kind=kind, params=(factor_bits,), src=(0,), dst=(1,))]
    (d / "batch.bin").write_bytes(wire.pack_batch(bufs, tensors, ops))

    run = subprocess.run(
        [exe, str(d / "batch.bin"), str(_FD_IN), str(_FD_OUT)],
        capture_output=True, text=True,
    )
    assert run.returncode == 0, (
        f"the skel dispatch probe exited {run.returncode}:\n"
        f"{run.stdout}\n{run.stderr}"
    )

    out = {"bases": {}}
    for line in run.stdout.splitlines():
        if line.startswith("mapped "):
            m = re.match(r"mapped fd=(-?\d+) base=(\d+)", line)
            out["bases"][int(m.group(1))] = int(m.group(2))
        elif line.startswith("dispatch "):
            out.update({k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", line)})
        elif line.startswith("in0="):
            m = re.match(r"in0=(\d+) out=(\d+)", line)
            out["in0"], out["out"] = int(m.group(1)), int(m.group(2))
        elif line.startswith("rsp="):
            out["rsp"] = bytes.fromhex(line[4:])
    for key in ("rc", "calls", "n", "in0", "out", "rsp"):
        assert key in out, f"the probe printed no {key}:\n{run.stdout}"
    assert out["bases"], f"the probe registered no fds:\n{run.stdout}"
    return out


@needs_cc
def test_the_dispatcher_and_the_generated_entry_agree_on_src_then_dst(both_sides):
    """THE CONTRACT, NOW GENUINELY PINNED FROM BOTH SIDES AT ONCE.

    `skel_dispatch.c` walks `op.src` then `op.dst` into one `a->buf[]`;
    `genentry` reads the output back out at `out_idx = n_in`. Neither references
    the other. Invert either and `scale_fp16` writes into its own input and
    returns the untouched output region -- at the right length, with status OK,
    on real silicon. Only the @sdk-gated numeric test would have noticed, and CI
    does not run it.

    What is checked is the ADDRESS the kernel received for each argument,
    against the fd-plus-offset arithmetic the batch declared, with the bases read
    out of skel_bufs.c's own mapping table. No index expression, no field name
    and no source text is matched, so this fails on the swap itself rather than
    on how anyone spelled it."""
    base_in = both_sides["bases"][_FD_IN]
    base_out = both_sides["bases"][_FD_OUT]
    assert base_in != base_out, "the two fds must map to different addresses"

    assert both_sides["calls"] == 1, "the kernel was not called exactly once"
    assert both_sides["in0"] == base_in + _OFF_IN, (
        f"scale_fp16 received {both_sides['in0']} as its INPUT; the batch's one "
        f"source tensor is at fd {_FD_IN} + {_OFF_IN} = {base_in + _OFF_IN}. "
        f"(The output tensor is at {base_out + _OFF_OUT} -- if that is what "
        f"arrived, sources and destinations are packed the other way round on "
        f"one of the two sides, and the kernel is reading what it should be "
        f"writing.)"
    )
    assert both_sides["out"] == base_out + _OFF_OUT, (
        f"scale_fp16 received {both_sides['out']} as its OUTPUT, expected "
        f"{base_out + _OFF_OUT} -- the destination sits at buf[n_in], filled "
        f"from op.dst after op.src"
    )


@needs_cc
def test_the_dispatcher_resolves_a_tensor_from_the_dsp_side_mapping_only(both_sides):
    """The corollary, and the invariant skel_bufs.c exists for: the address the
    kernel got is `base + offset` where `base` came from the DSP's OWN mmap
    table, keyed by fd. `wire.pack_batch` writes zero into the `base` wire slot
    and there is no field for a host address, so an implementation that leaned on
    one could not even be expressed here -- what this adds is that the address
    actually used is the mapped one, measured, rather than 0 + offset (upstream's
    silent fallthrough) or the offset alone."""
    base_in = both_sides["bases"][_FD_IN]
    assert both_sides["in0"] not in (0, _OFF_IN), (
        "the kernel's pointer must be a real mapped address plus the offset, "
        "not the offset alone or a zero base"
    )
    assert both_sides["in0"] - _OFF_IN == base_in
    assert both_sides["n"] == _NE[0], (
        f"the extent must be derived from the tensor's own ne on the DSP side; "
        f"got n={both_sides['n']}, expected {_NE[0]}"
    )
    assert both_sides["factor_ok"] == 1, (
        "the op's params blob must reach the kernel as float bits -- it is "
        "carried as int32[] on the wire and cast on the DSP side"
    )


@needs_cc
def test_the_response_the_dispatcher_wrote_is_what_wire_py_unpacks(both_sides):
    """END TO END, IN BOTH DIRECTIONS, IN ONE RUN: `wire.pack_batch` built the
    blob, the real dispatcher walked it, and the response bytes it wrote go back
    through `wire.unpack_response`. test_wire_struct_layout.py pins the two
    descriptions of these bytes against each other by size and offset; this pins
    them against a real dispatcher's real output.

    `cycles_total > 0` because the probe's HAP_perf stub advances -- which checks
    that the PCYCLE bracket's delta actually reaches the response header, the
    thing `hexlib/cli.py` refuses a device job over."""
    rsp = wire.unpack_response(both_sides["rsp"])
    assert rsp.status == STATUS["OK"], f"batch status {rsp.status}"
    assert rsp.n_ops == 1
    assert rsp.arch == 75, (
        "the response must carry the arch the skel was BUILT for "
        "(__HEXAGON_ARCH__), never a caller-supplied value"
    )
    assert len(rsp.results) == 1
    assert rsp.results[0].kind == ge.KIND_ID["scale"]
    assert rsp.results[0].status == STATUS["OK"]
    assert rsp.results[0].cycles > 0
    assert rsp.cycles_total == rsp.results[0].cycles, (
        "one op, so the batch total is that op's own bracket"
    )
    assert both_sides["rc"] == STATUS["OK"]

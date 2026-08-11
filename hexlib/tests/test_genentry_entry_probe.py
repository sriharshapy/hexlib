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
"""
import pathlib
import re
import shutil
import subprocess

import pytest

from hexlib.exec import runner as rn
from hexlib.runtime import genentry as ge
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
    are distinct static arrays, which is what makes identity meaningful."""
    assert probe["_scale_order"][0], (
        "scale_fp16 must receive buf[0] as its input and buf[1] as its output "
        "(1 source, then 1 destination)"
    )
    assert probe["_add_order"][0], (
        "add_fp16 must receive buf[0] and buf[1] as its two inputs and buf[2] as "
        "its output -- the destination sits at index n_in, not index 0"
    )

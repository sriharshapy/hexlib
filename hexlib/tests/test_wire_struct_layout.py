# hexlib/tests/test_wire_struct_layout.py
"""BEHAVIOURAL test for the HOST/DSP STRUCT SEAM: every `struct` format in
hexlib/runtime/wire.py against the real C structs in
hexlib/runtime/skel/hexlib_dsp.h, compiled by a host C compiler.

WHY THIS EXISTS. `wire.py` serializes a batch with hand-written `struct` format
strings (`_HDR`, `_BUF`, `_TENSOR`, `_OP`, `_RSP_HDR`, `_RESULT`); the DSP casts
the same bytes to `struct hexlib_batch_hdr`, `struct hexlib_buf_desc`,
`struct hexlib_tensor`, `struct hexlib_op_desc`, `struct hexlib_batch_rsp_hdr`
and `struct hexlib_op_result`. Nothing checked that those two descriptions of
the same bytes agreed. They do agree today -- every size and every offset was
confirmed by compiling the header -- so this file is not fixing a live bug; it
is closing the gap that would let one appear silently.

AND IT WOULD BE SILENT. Reorder `hexlib_tensor`'s `dtype` and `layout` fields
and every test in this repo stayed green while the DSP read a tensor's dtype as
its layout: a q4_0 weight interpreted as row_major, or vice versa. That is not
a crash and not a compile error -- it is a plausible wrong answer, produced at
full speed. test_runtime_wire.py's `test_c_header_agrees_with_python_on_every_
constant` checks the #defines and the status enum; the LAYOUT was never checked
by anything, in either direction.

THIS MATTERS MORE FROM HERE ON. The batch format is about to carry a 308-op
encoder plan rather than one op, so `_OP`'s 92 bytes get multiplied by 308 and
every offset in it is exercised 308 times per invoke. A one-field disagreement
that is survivable-looking with a single op is a garbled plan at that size.

WHAT THIS DOES, AND WHY IT IS THE SAME RECIPE AS ITS TWO SIBLINGS. Follows
test_session_arch_decode.py and test_coherency_lane_classification.py exactly:
generate a small standalone C program, compile it with a host C compiler, RUN
it, and compare real measured numbers against what Python claims -- rather than
asserting on source text. Here the program `#include`s hexlib_dsp.h itself
(never a retyped copy of the structs) and prints `sizeof` for each struct plus
`offsetof` and the member `sizeof` for every field. Same HOST_CC lookup and the
same `needs_cc` skipif as both siblings, for the same reason: a machine with no
host compiler must skip cleanly and say what it lost, not fail and not silently
pass.

THE BRIDGE, AND WHY IT IS NOT A HAND-MAINTAINED SECOND COPY. `wire.py`'s
formats carry no field NAMES, and the C structs carry no format string, so
something has to pair them: `WIRE_STRUCTS` below lists, per struct, the C field
names in the order the host writes them and how many format elements each field
occupies. It deliberately does NOT restate the types. Every type, size and
offset on the Python side is derived from `wire.py`'s own format strings by
`_flatten`, and `_assert_every_format_element_is_accounted_for` fails if a
field is added to or removed from a format without being added here -- so this
table cannot quietly drift out of agreement with wire.py, it can only stop
compiling against it.

`<` MEANS NO PADDING, WHICH IS THE ONE ASSUMPTION WORTH NAMING. Every format
in wire.py is little-endian-with-no-alignment (`<`), so Python's offsets are
just cumulative sizes. The C structs are naturally aligned by the compiler. The
two agree only because every field in every wire struct happens to be laid out
so that natural alignment introduces no padding (e.g. `hexlib_batch_rsp_hdr`
has exactly four uint32 before its uint64, so the uint64 lands on 16). That is
a real property of these structs and not a general one -- it is exactly what
this file measures rather than assumes.

COMPILER-INDEPENDENT HALF. `test_the_header_declares_every_wire_field_in_the_
order_the_host_writes_them` runs with no `cc` on PATH and catches the
dtype/layout reorder on its own, by reading the field order out of the header
text with `csource.block_from` (comment-aware and comment-blanked, so a field
name left behind in a comment cannot satisfy it). It cannot catch a TYPE change
or a padding change -- `uint32_t offset` becoming `uint64_t offset` keeps the
order intact -- which is what the compiled tests below are for.

THE THIRD DESCRIPTION OF THE SAME BYTES, WHICH NOTHING CHECKED AT ALL. Every
test above compares the C structs against wire.py's FORMAT STRINGS. There is a
third description in play and it is the one that actually runs: the ARGUMENT
ORDER of `pack_batch`'s `struct.pack` calls, and the unpacking order in
`unpack_response`. A format string says "eleven uint32 in a row"; it does not say
which value goes in which. Swapping `dtype` and `layout` in
`pack_batch`'s tensor `struct.pack(...)` call -- so every tensor's dtype is
written into the DSP's `layout` field and vice versa -- left the whole offline
suite at 810 passed. It was caught only by the @sdk-gated `test_dsp_sim.py`,
which needs the Hexagon SDK and which CI does not run: on any machine without
the SDK, and in CI, a q4_0 weight read as row_major (or vice versa) was
completely unpinned. The layout tests here could not see it, because the bytes
still had the right SIZE at the right OFFSETS -- they just meant different
things.

`test_pack_batch_writes_every_value_into_the_field_it_belongs_to` closes that,
with no compiler needed. It packs a batch in which every field of every record
holds a DISTINCT recognizable value, then reads each field back at the offset
this file's own bridge table implies and checks it is the value that field was
given. Any two fields exchanged in a `struct.pack` call swaps two distinct
values and fails. `test_unpack_response_reads_every_field_from_the_slot_the_dsp_
wrote_it_in` is the same idea in the other direction, on the response path.
"""
import pathlib
import re
import shutil
import struct
import subprocess
import sys

import pytest

from hexlib.runtime import wire
from hexlib.tests import csource

DSP_H = pathlib.Path("hexlib/runtime/skel/hexlib_dsp.h")

HOST_CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
needs_cc = pytest.mark.skipif(
    HOST_CC is None,
    reason=(
        "no host C compiler found (tried: cc, gcc, clang); the BEHAVIOURAL "
        "host/DSP struct-layout test is skipped, and only the weaker "
        "field-ORDER check in this same file "
        "(test_the_header_declares_every_wire_field_in_the_order_the_host_"
        "writes_them) covers this -- which cannot see a field's TYPE or "
        "alignment change. Install a host C compiler to restore it."
    ),
)


class _Wire:
    """One wire struct: the name of wire.py's format string, the C struct it
    describes, and the C field names in the order the host writes them paired
    with how many format elements each consumes (4 for `ne[4]`, 16 for
    `params[16]`, 1 for a scalar). No types: those come from wire.py."""

    def __init__(self, fmt_attr, c_name, fields):
        self.fmt_attr = fmt_attr
        self.c_name = c_name
        self.fields = fields

    @property
    def fmt(self):
        return getattr(wire, self.fmt_attr)

    def __repr__(self):
        return f"{self.c_name} (wire.{self.fmt_attr})"


WIRE_STRUCTS = (
    _Wire("_HDR", "hexlib_batch_hdr", (
        ("magic", 1), ("version", 1), ("total_size", 1), ("n_bufs", 1),
        ("n_tensors", 1), ("n_ops", 1), ("off_bufs", 1), ("off_tensors", 1),
        ("off_ops", 1), ("flags", 1),
    )),
    _Wire("_BUF", "hexlib_buf_desc", (
        # `base` first, and it is DSP-side scratch the host writes 0 into --
        # see wire.py's module docstring. Its OFFSET being right is what makes
        # "the host cannot express an address" true on the wire and not just in
        # the dataclass.
        ("base", 1), ("size", 1), ("fd", 1), ("flags", 1),
    )),
    _Wire("_TENSOR", "hexlib_tensor", (
        ("bi", 1), ("offset", 1), ("nbytes", 1), ("dtype", 1), ("layout", 1),
        ("ne", 4), ("data", 1), ("pad", 1),
    )),
    _Wire("_OP", "hexlib_op_desc", (
        ("kind", 1), ("flags", 1), ("params", wire.MAX_PARAMS),
        ("src", wire.MAX_SRC), ("dst", wire.MAX_DST),
    )),
    _Wire("_RSP_HDR", "hexlib_batch_rsp_hdr", (
        ("magic", 1), ("version", 1), ("status", 1), ("n_ops", 1),
        ("cycles_total", 1), ("arch", 1), ("pad", 1),
    )),
    _Wire("_RESULT", "hexlib_op_result", (
        ("kind", 1), ("status", 1), ("cycles", 1),
    )),
)

_COUNTED_CODE = re.compile(r"(\d*)([a-zA-Z?])")


def _flatten(fmt):
    """Expand a struct format into one code per element: `"<II16i6H4H"` ->
    `["I","I","i"*16...,"H"*6...,"H"*4...]`. Repeat counts are the only thing
    that makes a format's element count differ from its character count, and
    getting that wrong is how a bridge table silently stops lining up.

    Refuses anything but a `<` prefix: the whole comparison below assumes no
    alignment padding on the Python side, so a format that quietly changed to
    `@` or `=` must fail loudly here rather than produce offsets that look
    plausible."""
    assert fmt.startswith("<"), (
        f"{fmt!r} is not little-endian-no-padding ('<'); every offset computed "
        "in this file assumes that, so a change of byte-order character must "
        "be dealt with here explicitly, not absorbed"
    )
    out = []
    for count, code in _COUNTED_CODE.findall(fmt[1:]):
        out.extend([code] * (int(count) if count else 1))
    return out


def _python_layout(w):
    """What wire.py's own format implies: (total size, {field: (offset, size)}).
    Every number here comes from `struct.calcsize` over a slice of the REAL
    format string -- nothing is typed in."""
    codes = _flatten(w.fmt)
    layout = {}
    i = 0
    for name, n in w.fields:
        layout[name] = (
            struct.calcsize("<" + "".join(codes[:i])),
            struct.calcsize("<" + "".join(codes[i:i + n])),
        )
        i += n
    assert i == len(codes), (
        f"{w!r}: this file's field table accounts for {i} of the "
        f"{len(codes)} elements in wire.{w.fmt_attr} -- a field was added to "
        "or removed from the format without being added here, so the "
        "comparison below would have checked a prefix and called it a match"
    )
    return struct.calcsize(w.fmt), layout


def test_every_wire_format_element_is_accounted_for_by_this_files_field_table():
    """Runs with or without a compiler. `_python_layout`'s own trailing
    assertion is the point: it is what stops this file's bridge table from
    silently describing a subset of wire.py's formats."""
    for w in WIRE_STRUCTS:
        total, layout = _python_layout(w)
        assert total > 0
        assert len(layout) == len(w.fields)


def test_the_size_constants_wire_py_exports_match_its_own_formats():
    """Cheap, but it pins the pairing the rest of this file (and pack_batch's
    own offset arithmetic) relies on: HDR_SIZE really is calcsize(_HDR), etc."""
    for attr, size_attr in (
        ("_HDR", "HDR_SIZE"), ("_BUF", "BUF_SIZE"), ("_TENSOR", "TENSOR_SIZE"),
        ("_OP", "OP_SIZE"), ("_RSP_HDR", "RSP_HDR_SIZE"),
        ("_RESULT", "RESULT_SIZE"),
    ):
        assert getattr(wire, size_attr) == struct.calcsize(getattr(wire, attr))


# ==============================================================================
# WHAT pack_batch / unpack_response ACTUALLY PUT IN EACH SLOT. No compiler
# needed: this is Python's own serializer checked against this file's field
# table, which the compiled tests above have already checked against the C.
# ==============================================================================

_BY_NAME = {w.c_name: w for w in WIRE_STRUCTS}


def _read_record(c_name, blob, offset):
    """`{field: value}` for one record of `struct c_name` at `offset` in `blob`,
    unpacked with wire.py's own format and named by this file's field table.
    Array fields come back as tuples. This is the only place the two are
    paired, which is what makes a swapped pair of `struct.pack` arguments
    visible: the format cannot tell them apart, the names can."""
    w = _BY_NAME[c_name]
    values = struct.unpack_from(w.fmt, blob, offset)
    out, i = {}, 0
    for name, n in w.fields:
        out[name] = values[i] if n == 1 else tuple(values[i:i + n])
        i += n
    assert i == len(values)
    return out


def _assert_distinct(record, name, exempt=()):
    """Every scalar in `record` must be a DIFFERENT value, or a swap of the two
    that match would pass. `exempt` names fields whose value is fixed by the
    contract (the DSP-side scratch slots, which are both required to be 0) and
    so cannot be made distinct."""
    scalars = {k: v for k, v in record.items()
               if isinstance(v, int) and k not in exempt}
    assert len(set(scalars.values())) == len(scalars), (
        f"this test's own {name} values are not all distinct, so a swapped pair "
        f"of fields would pass it: {scalars!r}"
    )


# Chosen so that within every record every scalar differs from every other --
# including `version` (1) against `n_ops`, which is why there are two ops and
# three buffers rather than one of each.
_PACK_BUFS = (
    dict(fd=11, size=4096, flags=5),
    dict(fd=22, size=8192, flags=6),
    dict(fd=33, size=2048, flags=7),
)
_PACK_TENSORS = (
    dict(bi=0, offset=32, nbytes=16, dtype="fp16", layout="q4_0_repacked",
         ne=(3, 5, 7, 9)),
    # THE EXHAUSTIVELY-CHECKED ONE: every scalar in it is a different number,
    # and its dtype and layout ids differ from each other AND from the other
    # tensor's, so neither can be a constant and the two cannot be exchanged.
    dict(bi=2, offset=64, nbytes=128, dtype="int32", layout="tiled_32x32",
         ne=(11, 13, 17, 19)),
    dict(bi=1, offset=256, nbytes=512, dtype="fp32", layout="row_major",
         ne=(23, 29, 31, 37)),
    dict(bi=1, offset=1024, nbytes=48, dtype="q4_0", layout="q4_0_repacked",
         ne=(41, 43, 47, 53)),
)
_PACK_OPS = (
    dict(kind=9, flags=3, params=(101, 102, 103), src=(1, 2), dst=(0,)),
    dict(kind=10, flags=4, params=(201, 202), src=(0, 3), dst=(1,)),
)


@pytest.fixture(scope="module")
def packed():
    """One real `pack_batch` blob built from the tables above."""
    bufs = [wire.BufDesc(**b) for b in _PACK_BUFS]
    tensors = [wire.TensorDesc(**t) for t in _PACK_TENSORS]
    ops = [wire.OpDesc(**o) for o in _PACK_OPS]
    return wire.pack_batch(bufs, tensors, ops)


def test_pack_batch_writes_every_value_into_the_field_it_belongs_to(packed):
    """THE THIRD DESCRIPTION, CHECKED. See the module docstring: swapping
    `dtype` and `layout` in pack_batch's tensor pack call was 810-green offline
    and caught only by the SDK-gated simulator test CI does not run.

    Every assertion here is a whole-record equality, not a field-by-field spot
    check, so a field this test forgot cannot be the one that drifts -- and
    `_assert_distinct` refuses to let the test pass on values that could not
    tell a swap apart in the first place."""
    hdr = _read_record("hexlib_batch_hdr", packed, 0)
    _assert_distinct(hdr, "header", exempt=("flags",))
    assert hdr == {
        "magic": wire.BATCH_MAGIC,
        "version": wire.BATCH_VERSION,
        "total_size": len(packed),
        "n_bufs": len(_PACK_BUFS),
        "n_tensors": len(_PACK_TENSORS),
        "n_ops": len(_PACK_OPS),
        "off_bufs": wire.HDR_SIZE,
        "off_tensors": wire.HDR_SIZE + wire.BUF_SIZE * len(_PACK_BUFS),
        "off_ops": (wire.HDR_SIZE + wire.BUF_SIZE * len(_PACK_BUFS)
                    + wire.TENSOR_SIZE * len(_PACK_TENSORS)),
        "flags": 0,
    }

    for i, b in enumerate(_PACK_BUFS):
        rec = _read_record("hexlib_buf_desc", packed,
                           hdr["off_bufs"] + wire.BUF_SIZE * i)
        _assert_distinct(rec, f"buffer {i}", exempt=("base",))
        assert rec == {"base": 0, "size": b["size"], "fd": b["fd"],
                       "flags": b["flags"]}, (
            f"buffer {i}: pack_batch put the values somewhere other than the "
            f"fields they name. `base` must be 0 -- see wire.py's docstring: "
            f"there is no field for a host address, and the DSP fills this one"
        )

    for i, t in enumerate(_PACK_TENSORS):
        rec = _read_record("hexlib_tensor", packed,
                           hdr["off_tensors"] + wire.TENSOR_SIZE * i)
        if i == 1:
            _assert_distinct(rec, f"tensor {i}", exempt=("data", "pad"))
        assert rec == {
            "bi": t["bi"], "offset": t["offset"], "nbytes": t["nbytes"],
            "dtype": wire.DTYPE_ID[t["dtype"]],
            "layout": wire.LAYOUT_ID[t["layout"]],
            "ne": t["ne"], "data": 0, "pad": 0,
        }, (
            f"tensor {i}: the DSP reads these eleven uint32 by NAME "
            f"(hexlib_dsp.h) and pack_batch wrote them in a different order. "
            f"dtype/layout exchanged is a q4_0 weight read as row_major -- a "
            f"plausible wrong answer at full speed, not a crash"
        )

    for i, op in enumerate(_PACK_OPS):
        rec = _read_record("hexlib_op_desc", packed,
                           hdr["off_ops"] + wire.OP_SIZE * i)
        assert rec == {
            "kind": op["kind"],
            "flags": op["flags"],
            "params": (tuple(op["params"])
                       + (0,) * (wire.MAX_PARAMS - len(op["params"]))),
            "src": (tuple(op["src"])
                    + (0xFFFF,) * (wire.MAX_SRC - len(op["src"]))),
            "dst": (tuple(op["dst"])
                    + (0xFFFF,) * (wire.MAX_DST - len(op["dst"]))),
        }, (
            f"op {i}: `kind` is what skel_dispatch.c matches the kernel table "
            f"on and src/dst are what its fill loop walks in that order, so an "
            f"exchange here dispatches the wrong kernel or feeds it the wrong "
            f"buffers"
        )


def test_unpack_response_reads_every_field_from_the_slot_the_dsp_wrote_it_in():
    """THE RETURN PATH, SAME PROPERTY. `unpack_response` names its fields by
    tuple position (`magic, version, status, n_ops, cycles, arch, _ = ...`), so
    exchanging two of those names is invisible to any format-string comparison
    -- and reading `status` out of the `n_ops` slot would report a two-op batch
    as ERR_INTERNAL, or a failure as success.

    The bytes are built HERE, in this file's own field order (the order the
    compiled tests above have already checked against hexlib_dsp.h), rather than
    by wire.py -- otherwise a matching pair of mistakes on both sides would
    cancel out and this would pass."""
    hdr_fields = {
        "magic": wire.BATCH_MAGIC, "version": wire.BATCH_VERSION,
        "status": wire.STATUS["ERR_REQUIRES"], "n_ops": 2,
        "cycles_total": 1287, "arch": 75, "pad": 0,
    }
    _assert_distinct(hdr_fields, "response header", exempt=("pad",))
    results = (
        {"kind": 9, "status": wire.STATUS["OK"], "cycles": 101},
        {"kind": 10, "status": wire.STATUS["ERR_REQUIRES"], "cycles": 202},
    )
    for i, r in enumerate(results):
        _assert_distinct(r, f"result {i}")

    def _pack(c_name, values):
        w = _BY_NAME[c_name]
        flat = []
        for name, n in w.fields:
            v = values[name]
            flat.extend(v if n > 1 else [v])
        return struct.pack(w.fmt, *flat)

    blob = _pack("hexlib_batch_rsp_hdr", hdr_fields)
    for r in results:
        blob += _pack("hexlib_op_result", r)

    rsp = wire.unpack_response(blob)
    assert rsp.status == hdr_fields["status"], (
        "unpack_response read `status` out of a different slot than the one "
        "hexlib_dsp.h declares it in"
    )
    assert rsp.n_ops == hdr_fields["n_ops"]
    assert rsp.cycles_total == hdr_fields["cycles_total"]
    assert rsp.arch == hdr_fields["arch"]
    assert len(rsp.results) == len(results)
    for got, want in zip(rsp.results, results):
        assert (got.kind, got.status, got.cycles) == (
            want["kind"], want["status"], want["cycles"]
        ), (
            "an OpResult's kind/status/cycles came back in a different order "
            "than the DSP wrote them"
        )
    assert not rsp.ok, "a non-OK batch status must not read as ok"


@pytest.fixture(scope="module")
def header_source():
    """Comment-BLANKED header text. Every payload check in this file runs
    against this, never the raw text, so a struct field name that survives only
    inside a comment cannot satisfy an assertion -- see csource.py's module
    docstring for the mutation that made this the default."""
    return csource.code_only(DSP_H.read_text())


def _declared_fields(header, c_name):
    """The field names of `struct c_name`, in declaration order, sliced out of
    the header with `csource.block_from` (the shared comment-aware slicer, not
    a fourth private copy -- see csource.py's docstring). Array declarators are
    reduced to their name, so `uint32_t ne[4];` reads as `ne`."""
    marker = f"struct {c_name} {{"
    assert marker in header, f"the header no longer declares `struct {c_name}`"
    block = csource.block_from(header, header.index(marker))
    return [m.group(1) for m in re.finditer(r"\b(\w+)\s*(?:\[[^\]]*\])?\s*;", block)]


def test_the_header_declares_every_wire_field_in_the_order_the_host_writes_them(
    header_source,
):
    """COMPILER-INDEPENDENT -- runs even when `needs_cc` skips everything else,
    so a field REORDER (the dtype/layout swap that motivated this file) is never
    invisible on a machine with no host compiler. Order, and exactly these
    fields: an extra field in the C struct that the host never writes is just as
    much a seam break as a missing one, because it shifts everything after it."""
    for w in WIRE_STRUCTS:
        expected = [name for name, _ in w.fields]
        assert _declared_fields(header_source, w.c_name) == expected, (
            f"{w!r}: the C struct's fields are not the fields wire."
            f"{w.fmt_attr} writes, in that order. Host and DSP disagree about "
            "what the same bytes mean -- which is a wrong answer at full "
            "speed, not a compile error"
        )


def _emit_probe_c():
    """A standalone C program that includes the REAL header and prints what the
    compiler actually laid out. Generated from WIRE_STRUCTS so a field added
    there is probed automatically.

    `(unsigned long)` + `%lu` rather than `%zu`: the host compiler here is
    mingw gcc, whose `%zu` support depends on which stdio it was built against
    (see test_coherency_lane_classification.py's note on this same toolchain
    lacking `__fp16`). Every number printed is an offset or a small size, so a
    32-bit-safe cast costs nothing and removes the variable."""
    lines = [
        "#include <stddef.h>",
        "#include <stdio.h>",
        '#include "hexlib_dsp.h"',
        "int main(void) {",
    ]
    for w in WIRE_STRUCTS:
        lines.append(
            f'    printf("{w.c_name} . %lu 0\\n", '
            f"(unsigned long) sizeof(struct {w.c_name}));"
        )
        for name, _ in w.fields:
            lines.append(
                f'    printf("{w.c_name} {name} %lu %lu\\n", '
                f"(unsigned long) offsetof(struct {w.c_name}, {name}), "
                f"(unsigned long) sizeof(((struct {w.c_name} *) 0)->{name}));"
            )
    lines += ["    return 0;", "}", ""]
    return "\n".join(lines)


@pytest.fixture(scope="module")
def measured_layout(tmp_path_factory):
    """Compile and RUN the probe, and return
    {c_struct: (sizeof, {field: (offset, member_sizeof)})} as the host compiler
    really laid it out. `-I` points at the header's own directory so the
    `#include` resolves to the checked-in file and nothing else."""
    tmp_path = tmp_path_factory.mktemp("wire_layout")
    c_path = tmp_path / "probe.c"
    c_path.write_text(_emit_probe_c())
    exe = tmp_path / ("probe.exe" if sys.platform == "win32" else "probe")

    compile_result = subprocess.run(
        [HOST_CC, "-o", str(exe), str(c_path), "-I", str(DSP_H.parent.resolve())],
        capture_output=True, text=True,
    )
    assert compile_result.returncode == 0, (
        "compiling the hexlib_dsp.h struct-layout probe failed -- the header "
        "itself may not compile, which is a finding, not a reason to skip:\n"
        f"{compile_result.stdout}\n{compile_result.stderr}"
    )

    run_result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run_result.returncode == 0, (
        f"the struct-layout probe exited {run_result.returncode}:\n"
        f"{run_result.stdout}\n{run_result.stderr}"
    )

    out = {}
    for line in run_result.stdout.split("\n"):
        parts = line.split()
        if len(parts) != 4:
            continue
        c_name, field, a, b = parts
        sizeof, fields = out.setdefault(c_name, (None, {}))
        if field == ".":
            out[c_name] = (int(a), fields)
        else:
            fields[field] = (int(a), int(b))
    assert out, (
        "the struct-layout probe compiled and ran but printed nothing "
        f"parseable -- refusing to read that as agreement:\n{run_result.stdout}"
    )
    return out


@needs_cc
@pytest.mark.parametrize("w", WIRE_STRUCTS, ids=lambda w: w.c_name)
def test_the_c_struct_is_the_size_wire_py_serializes(w, measured_layout):
    """A size disagreement means the DSP reads op N at the wrong address for
    every N > 0 -- and with a 308-op plan that is 307 garbled ops behind one
    correct one."""
    expected_size, _ = _python_layout(w)
    assert w.c_name in measured_layout, f"the probe printed nothing for {w!r}"
    measured_size, _ = measured_layout[w.c_name]
    assert measured_size == expected_size, (
        f"sizeof(struct {w.c_name}) is {measured_size} on the host compiler, "
        f"but wire.{w.fmt_attr} serializes {expected_size} bytes"
    )


@needs_cc
@pytest.mark.parametrize("w", WIRE_STRUCTS, ids=lambda w: w.c_name)
def test_every_c_field_is_at_the_offset_and_width_wire_py_writes(w, measured_layout):
    """THE DECISIVE ONE. Offset AND member width, per field. Offset catches a
    reorder or unexpected padding; width catches a type change that a reorder
    check cannot see (`uint32_t offset` -> `uint64_t offset` shifts nothing
    before it and everything after)."""
    _, expected = _python_layout(w)
    _, measured = measured_layout[w.c_name]
    for name, (exp_off, exp_size) in expected.items():
        assert name in measured, f"the probe printed no offset for {w.c_name}.{name}"
        got_off, got_size = measured[name]
        assert got_off == exp_off, (
            f"offsetof(struct {w.c_name}, {name}) is {got_off} in the C "
            f"header, but wire.{w.fmt_attr} writes that field at {exp_off}. "
            "The DSP is reading a different field than the host wrote"
        )
        assert got_size == exp_size, (
            f"sizeof(struct {w.c_name}.{name}) is {got_size} in the C header, "
            f"but wire.{w.fmt_attr} writes {exp_size} bytes there"
        )


@needs_cc
def test_the_dsp_side_scratch_fields_are_where_the_host_writes_its_zeros(
    measured_layout,
):
    """`hexlib_buf_desc.base` and `hexlib_tensor.data` are DSP-side scratch;
    the host writes zeros into those exact wire slots (see wire.py's
    pack_batch) so no host address can cross. test_runtime_wire.py's
    `test_host_writes_zero_into_tensor_data` reads `data` back at a HARDCODED
    `off_t + 9 * 4` -- correct today, and true only because of this layout.
    Pin the offsets that hardcoding depends on, so if the header moves `data`
    that test starts checking a different field's bytes and this one says why."""
    _, buf = measured_layout["hexlib_buf_desc"]
    assert buf["base"][0] == 0, "the host's zero must land on `base`"
    _, tensor = measured_layout["hexlib_tensor"]
    assert tensor["data"] == (9 * 4, 4), (
        "hexlib_tensor.data is no longer the 10th uint32 -- "
        "test_runtime_wire.py::test_host_writes_zero_into_tensor_data reads "
        f"it at a hardcoded offset 36 and would now read {tensor['data']}"
    )


# ---------------------------------------------------------------------------
# The layout enum, on both sides at once
# ---------------------------------------------------------------------------


@needs_cc
def test_the_layout_enum_has_the_same_values_in_c_as_on_the_wire(tmp_path):
    """`HEXLIB_LAYOUT_*` in the header vs `wire.LAYOUT_ID`, COMPILED.

    `main.c` used to write `tens[i].layout = 0` as a bare literal whose only
    tie to `LAYOUT_ID["row_major"]` was a trailing comment, and unlike the
    `dtype` literal beside it nothing checked layout anywhere: `grep -c layout
    hexlib/tests/test_host_source.py` was 0. Inserting a value ahead of
    row_major would have left `pack_batch` emitting 1 while `main.c` kept
    emitting 0, and `--self-test` would still have printed `PASS (4100 values,
    bit-exact)` over a buffer the batch declared as a different layout.

    Compiled rather than grepped for the reason this whole file exists: a
    source assertion over `#define` lines is satisfied by a comment, and was
    twice satisfied by a string literal. The preprocessor's own numbers are the
    only thing that cannot be faked. Every entry in `LAYOUT_ID` must have a
    macro, so ADDING a Python-side layout without adding the C one fails here
    too -- which is the direction the next kernel takes (`q4_0_repacked`).
    """
    names = {k: "HEXLIB_LAYOUT_" + k.upper() for k in wire.LAYOUT_ID}
    lines = ['#include <stdio.h>', '#include "hexlib_dsp.h"', "int main(void) {"]
    for key, macro in names.items():
        lines.append(f'    printf("{key} %lu' + r'\n' + f'", (unsigned long) {macro});')
    lines += ["    return 0;", "}", ""]

    c_path = tmp_path / "layout_probe.c"
    c_path.write_text("\n".join(lines))
    exe = tmp_path / ("lp.exe" if sys.platform == "win32" else "lp")
    cc = subprocess.run(
        [HOST_CC, "-o", str(exe), str(c_path), "-I", str(DSP_H.parent.resolve())],
        capture_output=True, text=True,
    )
    assert cc.returncode == 0, (
        "the layout-enum probe did not compile. Every name in wire.LAYOUT_ID "
        "must have a HEXLIB_LAYOUT_<NAME> macro in hexlib_dsp.h -- a layout "
        "that exists only on the Python side is one main.c cannot spell:\n"
        f"{cc.stdout}\n{cc.stderr}"
    )
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, f"layout probe exited {run.returncode}"

    got = {}
    for line in run.stdout.split("\n"):
        parts = line.split()
        if len(parts) == 2:
            got[parts[0]] = int(parts[1])

    assert got == dict(wire.LAYOUT_ID), (
        f"the C layout macros are {got} but wire.LAYOUT_ID is "
        f"{dict(wire.LAYOUT_ID)}. These cross the wire in "
        "hexlib_tensor.layout; a mismatch is a correctly-shaped wrong answer, "
        "not a compile error."
    )


# ---------------------------------------------------------------------------
# The status names, on both sides at once
# ---------------------------------------------------------------------------


@needs_cc
def test_the_status_names_agree_between_c_and_the_wire_table(tmp_path):
    """`hexlib_dsp_status_name` vs `wire.STATUS`, COMPILED.

    The host prints this string when the DSP tags a real status into an
    AEEResult (`session.c`, via `HEXLIB_AEE_IS_STATUS`). Before that decode
    existed, every cause of a failed `start()` printed the same bare negative
    number -- a VTCM contention failure looked exactly like a signing failure or
    a missing skel, which is the operator confusion the tag was added to remove.
    A wrong NAME here is a different flavour of the same problem: it sends
    someone to debug the wrong subsystem.

    Compiled rather than grepped, for the reason this whole file exists: a source
    assertion over `case` labels is satisfiable by a comment, and twice was by a
    string literal. This drives the real function with every value in
    `wire.STATUS` and compares what it actually returns.

    Both directions are checked. Every wire status must have a name (a status
    added to the Python table and not the switch), and the switch must not
    invent one for a value the table does not have -- the `default` returns
    "UNKNOWN", so an out-of-range value is reported as unknown rather than
    silently reading a neighbouring string.
    """
    names = sorted(wire.STATUS.items(), key=lambda kv: kv[1])
    lines = ['#include <stdio.h>', '#include "hexlib_dsp.h"', "int main(void) {"]
    for _, value in names:
        lines.append(f'    printf("%d %s' + r'\n' + f'", {value}, '
                     f"hexlib_dsp_status_name({value}));")
    # And one value deliberately outside the enum.
    lines.append('    printf("%d %s' + r'\n' + '", 999, hexlib_dsp_status_name(999));')
    lines += ["    return 0;", "}", ""]

    c_path = tmp_path / "status_probe.c"
    c_path.write_text("\n".join(lines))
    exe = tmp_path / ("sp.exe" if sys.platform == "win32" else "sp")
    cc = subprocess.run(
        [HOST_CC, "-o", str(exe), str(c_path), "-I", str(DSP_H.parent.resolve())],
        capture_output=True, text=True,
    )
    assert cc.returncode == 0, (
        "the status-name probe did not compile -- hexlib_dsp.h itself may not, "
        f"which is a finding and not a reason to skip:\n{cc.stdout}\n{cc.stderr}"
    )
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, f"status probe exited {run.returncode}"

    got = {}
    for line in run.stdout.split("\n"):
        parts = line.split()
        if len(parts) == 2:
            got[int(parts[0])] = parts[1]

    for name, value in names:
        assert got.get(value) == name, (
            f"wire.STATUS says {value} is {name!r} but hexlib_dsp_status_name "
            f"returns {got.get(value)!r}. The host prints this string to tell an "
            f"operator which subsystem failed."
        )
    assert got.get(999) == "UNKNOWN", (
        f"a status outside the enum returned {got.get(999)!r}; it must be "
        f"reported as unknown rather than resolving to some other name"
    )

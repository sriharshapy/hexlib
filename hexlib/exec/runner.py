"""One declarative description per kernel, instead of one Python function per kernel.

`hexagon.scale_backend` was written for a single input and a single float attr.
Generalising it is not optional work: `matmul_epilogue` takes three inputs, one
of them block-quantized, plus an activation selector, and discovering the gaps in
a bespoke marshaller inside the hardest kernel in the set is the expensive way to
find them.

THE WIRE FORMAT, little-endian:

    hexlib_in.bin    scalars, in the order `RunnerSpec.scalars` lists them
                     then each input's payload, in the order `inputs` lists them,
                     each in its own storage dtype, C-contiguous
    hexlib_out.bin   the output payload, in `out_dtype`

WHAT IS PINNED, AND BY WHAT. The agreement between this table and a kernel's
`runner.c` is the whole contract, and two comments that can drift apart is not a
contract. So each kernel's test asserts the pair agrees, and every scalar a
runner reads has to appear in the spec that feeds it -- a mismatch shows up as a
byte offset error on the first call, which is loud, rather than as plausible
wrong numbers.

WHY SCALARS COME FIRST. The payloads are variable length and the scalars say how
long they are. A runner has to know `n` before it can read `x[n]`.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

# Storage dtypes on the wire. Deliberately a separate table from
# exec.vtcm.STORAGE_DTYPE even though they agree today: this one describes bytes
# in a file that a C program parses, and changing it is an ABI change.
WIRE_DTYPE: dict[str, np.dtype] = {
    "fp32": np.dtype("<f4"),
    "fp16": np.dtype("<f2"),
    "int32": np.dtype("<i4"),
}

# BLOCK-QUANTIZED STORAGE, which is not a numpy dtype and never will be.
#
# A q4_0 tensor is a stream of 18-byte blocks -- one fp16 scale then 32 4-bit
# values, `llama.cpp`'s `block_q4_0` (see hexlib/graph/ir.py, which is the
# authority on the byte count and enforces that the last dimension is a multiple
# of 32). There is no numpy scalar dtype describing one element of that, so these
# dtypes cannot go in WIRE_DTYPE and `__post_init__` used to refuse them outright:
# "block-quantized inputs are staged as raw bytes and need their own path". This
# is that path, and 75 of the encoder's 308 plan steps need it.
#
# A raw input is staged VERBATIM from a `RawTensor`. hexlib does not quantize
# here; whatever produced the checkpoint did.
#
# ITS `ne` IS STILL THE LOGICAL ELEMENT SHAPE, not the byte shape, because
# `hexlib_tensor` carries `nbytes` and `ne[]` as separate fields and `wire.py`
# never asserts `nbytes == prod(ne) * itemsize` -- it only checks the buffer
# actually holds `offset + nbytes`. So a (768, 768) q4_0 weight says ne=(768,768)
# and nbytes=331776, both true. Writing the byte shape into `ne` would be a lie on
# the wire, and every `dim:` scalar and every kernel reading `a->ne` would inherit
# it.
WIRE_RAW: frozenset[str] = frozenset({"q4_0"})

_STRUCT_CODE = {"int": "i", "float": "f"}

# THE LAYOUT NAMES ARE IMPORTED, NOT RESPELLED, unlike WIRE_DTYPE above. That
# table is deliberately independent because it maps names to numpy dtypes and
# only agrees with wire.py's by coincidence of naming; this one is the same set
# of names for the same field, and a second copy of it is precisely the drift
# that left main.c emitting layout 0 while pack_batch could emit something else.
# wire.py imports nothing but the stdlib, so this cannot cycle.
from hexlib.runtime.wire import LAYOUT_ID as WIRE_LAYOUT  # noqa: E402


@dataclass(frozen=True)
class RawTensor:
    """An already-quantized input: opaque bytes plus its LOGICAL element shape.

    A separate type rather than a numpy array, for two reasons. `np.ndarray` does
    not accept arbitrary attributes, so the logical shape cannot simply be
    attached to a uint8 array -- and more importantly, a bare uint8 array would
    make the dangerous mistake silent. Handing `payload` an fp32 weight array and
    staging its bytes as though they were q4_0 blocks yields either a buffer of
    the wrong size or, when the sizes happen to line up, a correctly-shaped
    answer computed from noise; numpy never objects, because `.tobytes()` on any
    array is always willing. Requiring a distinct type at the call site means the
    caller has to have known what they were passing.

    `shape` is the ELEMENT shape -- (768, 768) for a q4_0 weight, not the 331776
    bytes it occupies. That is what goes in `ne` on the wire; `nbytes` carries the
    byte count separately, and `wire.py` never conflates them.
    """

    dtype: str
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self) -> None:
        from hexlib.graph import ir

        if self.dtype not in WIRE_RAW:
            raise ValueError(
                f"RawTensor is for block-quantized storage only; {self.dtype!r} "
                f"is not in WIRE_RAW ({sorted(WIRE_RAW)})"
            )
        # ir.nbytes is the authority on q4_0's 18-bytes-per-32-elements block and
        # already refuses a last dimension that is not a multiple of 32. Asking it
        # here rather than recomputing means the two cannot disagree, and it is
        # what makes a truncated or mis-shaped weight a construction-time error
        # instead of a wrong answer.
        want = ir.nbytes(tuple(self.shape), self.dtype)
        if len(self.data) != want:
            raise ValueError(
                f"a {self.dtype} tensor of logical shape {tuple(self.shape)} is "
                f"{want} bytes of blocks, but {len(self.data)} were given"
            )

    @property
    def nbytes(self) -> int:
        return len(self.data)

    @property
    def size(self) -> int:
        """Element count, so `numel:` scalars read the LOGICAL count."""
        n = 1
        for d in self.shape:
            n *= int(d)
        return n


def raw_bytes(kind: str, idx: int, dtype: str, value) -> bytes:
    """The already-quantized bytes of one raw input, verbatim."""
    if not isinstance(value, RawTensor):
        raise ValueError(
            f"{kind}: input {idx} is declared {dtype!r}, which is staged as raw "
            f"quantized bytes, so it must be a RawTensor and not a "
            f"{type(value).__name__}. hexlib does not quantize here -- pass the "
            f"already-quantized bytes with their logical shape."
        )
    if value.dtype != dtype:
        raise ValueError(
            f"{kind}: input {idx} is declared {dtype!r} but the RawTensor says "
            f"{value.dtype!r}"
        )
    return value.data


@dataclass(frozen=True)
class Scalar:
    """One value in the header.

    `source` is either 'attr:<name>' (read from the op's attrs), 'numel:<i>'
    (the element count of input i), or 'dim:<i>:<axis>' (one dimension of input
    i). Those three cover every kernel in the encoder without letting a spec
    smuggle in arbitrary host-side computation, which would put logic somewhere
    no kernel test looks.
    """

    source: str
    ctype: str = "int"

    def value(self, arrays: tuple[np.ndarray, ...], attrs: Mapping[str, Any]) -> Any:
        kind, _, rest = self.source.partition(":")
        if kind == "attr":
            if rest not in attrs:
                raise KeyError(
                    f"runner scalar wants attr {rest!r}; op attrs are {sorted(attrs)}"
                )
            return attrs[rest]
        if kind == "numel":
            return int(arrays[int(rest)].size)
        if kind == "dim":
            idx, _, axis = rest.partition(":")
            return int(arrays[int(idx)].shape[int(axis)])
        raise ValueError(
            f"unknown runner scalar source {self.source!r}; expected attr:, "
            "numel: or dim:"
        )


@dataclass(frozen=True)
class RunnerSpec:
    kind: str
    kernel_dir: str
    inputs: tuple[str, ...]
    out_dtype: str
    scalars: tuple[Scalar, ...] = ()
    # Attr values this kernel is only valid for. An op kind is not always one
    # kernel: `transpose` covers three distinct permutations in this graph, and
    # handing a perm(0,2,1) op to the perm(1,0,2) kernel would produce a
    # perfectly-shaped, silently WRONG layout. Every shape check downstream
    # passes, so nothing but the values would catch it. Checked before the
    # kernel is invoked, and a mismatch is an error rather than a fallback.
    requires: tuple[tuple[str, Any], ...] = ()
    # The wire layout each buffer must be declared as, inputs then output, or ()
    # for "row_major throughout" -- which every kernel shipped so far is. Named
    # per buffer rather than per kernel because the matmul this leads to takes a
    # `q4_0_repacked` weight beside a row-major activation, and `LAYOUT_ID`
    # already carries that value. See `genentry._layout_check` for what enforces
    # it on the DSP and why an unchecked enum is a comment, not a mechanism.
    layouts: tuple[str, ...] = ()
    # Output shape comes from the graph, not from the kernel: the op's `infer`
    # already declared it and the executor checks it. A kernel that returned a
    # different length fails the byte-count check in the backend.
    out_shape_from: str = "declared"
    notes: str = ""

    def buf_layouts(self) -> tuple[str, ...]:
        """The layout of every buffer, inputs then output, always fully spelled.

        `layouts=()` means row_major throughout, which is what every kernel
        shipped so far is -- so the default keeps the declaration short without
        making "unspecified" a third possibility anything downstream has to
        handle. Callers get one buffer per buffer, in the order
        `skel_dispatch.c` walks src then dst.
        """
        if not self.layouts:
            return ("row_major",) * (len(self.inputs) + 1)
        return self.layouts

    def check_requires(self, attrs: Mapping[str, Any]) -> None:
        for key, want in self.requires:
            got = attrs.get(key)
            if got != want:
                raise ValueError(
                    f"{self.kernel_dir} implements {self.kind} only for "
                    f"{key}={want!r}, but this op has {key}={got!r}. Dispatching "
                    "it here would produce a correctly-shaped wrong answer."
                )

    def __post_init__(self) -> None:
        for dtype in self.inputs:
            if dtype not in WIRE_DTYPE and dtype not in WIRE_RAW:
                raise ValueError(
                    f"{self.kind}: {dtype!r} is neither a dense wire dtype "
                    f"({sorted(WIRE_DTYPE)}) nor a raw block-quantized one "
                    f"({sorted(WIRE_RAW)})"
                )
        # THE OUTPUT MAY NOT BE RAW, and that is a real restriction rather than an
        # oversight. `decode` reads the result back through `np.frombuffer` with a
        # numpy dtype, and no kernel in this encoder writes a quantized result --
        # the weights arrive quantized and everything computed is fp16. Allowing it
        # would mean a `decode` that cannot decode.
        if self.out_dtype not in WIRE_DTYPE:
            raise ValueError(
                f"{self.kind}: out_dtype {self.out_dtype!r} is not a dense wire "
                f"dtype. A kernel may READ block-quantized bytes (see WIRE_RAW) "
                f"but not write them: the result has to be decodable."
            )
        for s in self.scalars:
            if s.ctype not in _STRUCT_CODE:
                raise ValueError(f"{self.kind}: unknown scalar ctype {s.ctype!r}")
        # Refused at construction, not at generate time: an unknown layout name
        # would reach `genentry._layout_check` as a KeyError from a dict lookup
        # inside an f-string, which says nothing about which spec is wrong. A
        # short `layouts` is the worse error of the two -- it silently leaves the
        # output buffer, or an input, with no guard at all.
        if self.layouts:
            want = len(self.inputs) + 1
            if len(self.layouts) != want:
                raise ValueError(
                    f"{self.kind}: layouts has {len(self.layouts)} entries but "
                    f"this kernel has {len(self.inputs)} input(s) plus one "
                    f"output = {want}. Every buffer must be named, in src-then-"
                    f"dst order, or leave layouts=() for row_major throughout."
                )
            for layout in self.layouts:
                if layout not in WIRE_LAYOUT:
                    raise ValueError(
                        f"{self.kind}: {layout!r} is not a layout on the wire; "
                        f"known layouts are {sorted(WIRE_LAYOUT)}"
                    )

    def header(
        self, arrays: tuple[np.ndarray, ...], attrs: Mapping[str, Any]
    ) -> bytes:
        fmt = "<" + "".join(_STRUCT_CODE[s.ctype] for s in self.scalars)
        values = []
        for s in self.scalars:
            v = s.value(arrays, attrs)
            values.append(int(v) if s.ctype == "int" else float(v))
        return struct.pack(fmt, *values)

    def payload(self, arrays: tuple[np.ndarray, ...]) -> bytes:
        if len(arrays) != len(self.inputs):
            raise ValueError(
                f"{self.kind} takes {len(self.inputs)} inputs, got {len(arrays)}"
            )
        out = bytearray()
        for i, (array, dtype) in enumerate(zip(arrays, self.inputs)):
            if dtype in WIRE_RAW:
                out += raw_bytes(self.kind, i, dtype, array)
            else:
                out += np.ascontiguousarray(array, dtype=WIRE_DTYPE[dtype]).tobytes()
        return bytes(out)

    def encode(
        self, arrays: tuple[np.ndarray, ...], attrs: Mapping[str, Any]
    ) -> bytes:
        return self.header(arrays, attrs) + self.payload(arrays)

    def decode(self, raw: bytes, shape: tuple[int, ...]) -> np.ndarray:
        dtype = WIRE_DTYPE[self.out_dtype]
        expect = int(np.prod(shape)) * dtype.itemsize if shape else dtype.itemsize
        if len(raw) != expect:
            raise ValueError(
                f"{self.kind}: output is {len(raw)} bytes, expected {expect} for "
                f"shape {shape} of {self.out_dtype}"
            )
        return np.frombuffer(raw, dtype=dtype).reshape(shape).astype(np.float32)


# --- the encoder's kernels ------------------------------------------------
#
# One entry per op kind that has a kernel directory with a runner.c. An op kind
# absent from here falls back to the registry's reference implementation, which
# is why the encoder runs at every stage of this work rather than only at the
# end.

SPECS: dict[str, RunnerSpec] = {
    "scale": RunnerSpec(
        kind="scale",
        kernel_dir="kernels/scale_fp16",
        inputs=("fp16",),
        out_dtype="fp16",
        scalars=(Scalar("numel:0", "int"), Scalar("attr:factor", "float")),
        notes="12 ops. factor is 1/sqrt(head_dim) = 0.125, exact in fp16.",
    ),
    "add": RunnerSpec(
        kind="add",
        kernel_dir="kernels/add_fp16",
        inputs=("fp16", "fp16"),
        out_dtype="fp16",
        scalars=(Scalar("numel:0", "int"),),
        notes=(
            "25 ops. 24 are fp16+fp16; the 25th takes an fp32 right operand "
            "(the learned pos_embed) and is rounded to fp16 on the wire rather "
            "than given a second kernel."
        ),
    ),
    "cast": RunnerSpec(
        kind="cast",
        kernel_dir="kernels/cast_f32_f16",
        inputs=("fp32",),
        out_dtype="fp16",
        scalars=(Scalar("numel:0", "int"),),
        requires=(("dtype", "fp16"),),
        notes=(
            "1 op, [256,1536] fp32 -> fp16, the host/activation dtype boundary. "
            "`requires` pins the target dtype: `cast` is a general op kind and a "
            "cast to anything else would be a different kernel."
        ),
    ),
    "transpose": RunnerSpec(
        kind="transpose",
        kernel_dir="kernels/transpose_th_fp16",
        inputs=("fp16",),
        out_dtype="fp16",
        scalars=(
            Scalar("dim:0:0", "int"),
            Scalar("dim:0:1", "int"),
            Scalar("dim:0:2", "int"),
        ),
        requires=(("perm", (1, 0, 2)),),
        notes=(
            "48 of the graph's 60 transposes: perm (1,0,2) in both directions, "
            "[256,12,64]->[12,256,64] (36 ops) and [12,256,64]->[256,12,64] "
            "(12 ops). One kernel covers both, because swapping axes 0 and 1 is "
            "the same operation with the dims passed the other way round.\n"
            "The remaining 12 are perm (0,2,1), which transposes the INNERMOST "
            "two axes -- no contiguous run survives, so it is a genuinely "
            "different kernel. `requires` refuses them rather than returning a "
            "correctly-shaped wrong answer."
        ),
    ),
    "layernorm": RunnerSpec(
        kind="layernorm",
        kernel_dir="kernels/layernorm_fp16",
        inputs=("fp16", "fp32", "fp32"),
        out_dtype="fp16",
        scalars=(
            Scalar("dim:0:0", "int"),    # R
            Scalar("dim:0:1", "int"),    # C
            Scalar("attr:eps", "float"),
        ),
        notes=(
            "All 25 layernorm ops in the encoder share ONE signature -- checked "
            "against the built graph, not assumed: x=(256,768) fp16, weight and "
            "bias both (768,) fp32, eps=1e-06. Because the input is rank 2, R and "
            "C are each a single dimension, so no new Scalar source was needed.\n"
            "THE KERNEL EXISTED AND GATED GREEN FOR A DAY WITHOUT THIS SPEC, and "
            "the spec -- not a runner.c -- is what makes an op dispatchable on the "
            "DSP batch path. Until this landed, KIND_ID['layernorm'] = 3 was "
            "reachable on the wire and answered ERR_NO_KERNEL, while STATE.md "
            "counted its 25 ops as covered.\n"
            "First kernel here with THREE inputs, and the first with MIXED input "
            "dtypes: fp16 data against fp32 affine parameters. That makes it the "
            "first real exercise of the generated entry's per-input dtype check, "
            "which previously only ever saw buffers of one type.\n"
            "Its 111088 cycles are a first rung, not a result -- both reductions "
            "are still scalar. Note the gate measured R=4; the encoder needs "
            "R=256, which the kernel takes as a parameter and no harness has run."
        ),
    ),
}


def spec_for(kind: str) -> RunnerSpec | None:
    return SPECS.get(kind)

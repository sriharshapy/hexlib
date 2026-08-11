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

_STRUCT_CODE = {"int": "i", "float": "f"}


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
    # Output shape comes from the graph, not from the kernel: the op's `infer`
    # already declared it and the executor checks it. A kernel that returned a
    # different length fails the byte-count check in the backend.
    out_shape_from: str = "declared"
    notes: str = ""

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
        for dtype in self.inputs + (self.out_dtype,):
            if dtype not in WIRE_DTYPE:
                raise ValueError(
                    f"{self.kind}: {dtype!r} has no wire form; block-quantized "
                    "inputs are staged as raw bytes and need their own path"
                )
        for s in self.scalars:
            if s.ctype not in _STRUCT_CODE:
                raise ValueError(f"{self.kind}: unknown scalar ctype {s.ctype!r}")

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
        for array, dtype in zip(arrays, self.inputs):
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

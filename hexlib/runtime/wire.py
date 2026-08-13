# hexlib/runtime/wire.py
"""The batch blob the host sends and the response it gets back.

WHY A BATCH AND NOT ONE OP PER CALL. A batch descriptor -- buffers, tensors and
ops together, tensors addressed as (buffer index, offset) -- is a serialized
`Compiled`. A single-op descriptor is a function call. Only the first can carry a
plan, and carrying a plan on the DSP is M2, which this work exists to unblock. A
one-op batch is the trivial instance, so nothing is made harder by starting here.

Shape adapted from llama.cpp ggml-hexagon's htp_opbatch_req (MIT). See
ATTRIBUTION.md.

THERE IS NO FIELD FOR AN ADDRESS. `BufDesc` has no `base` and `TensorDesc` has no
`data`; the serializer writes zeros into those wire slots and the DSP fills them.
Under the simulator the host and the DSP share one address space, so an
implementation that leaned on a host pointer would pass locally and fail on
silicon. Removing the field is what stops that being expressible.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

BATCH_MAGIC = 0x424C5848  # 'HXLB'
BATCH_VERSION = 1

MAX_BUFS = 8
MAX_SRC = 6
MAX_DST = 4
MAX_PARAMS = 16
MAX_TENSORS = 512

STATUS = {
    "OK": 1,
    "ERR_INTERNAL": 2,
    "ERR_BAD_MAGIC": 3,
    "ERR_BAD_VERSION": 4,
    "ERR_TRUNCATED": 5,
    "ERR_INVAL_PARAMS": 6,
    "ERR_UNMAPPED": 7,
    "ERR_NO_MMAP_SLOT": 8,
    "ERR_MMAP_FAILED": 9,
    "ERR_NO_KERNEL": 10,
    "ERR_VTCM_TOO_SMALL": 11,
    "ERR_VTCM_RECLAIMED": 12,
    "ERR_REQUIRES": 13,
    "ERR_NOT_STARTED": 14,
    "ERR_CACHE": 15,
}
STATUS_NAME = {v: k for k, v in STATUS.items()}

# Keyed by the SAME strings as `hexlib.exec.runner.WIRE_DTYPE` and
# `hexlib.runtime.genentry._CTYPE` -- "int32", not "i32". The ids are the ABI
# (they are what `hexlib_tensor.dtype` carries); the keys are host-side names,
# so this spelling fix changed no byte on the wire. It was a live break, not a
# tidy-up: the first spec declaring an int32 input would have been refused by
# `pack_batch` as an unknown dtype AND have KeyError'd genentry's `requires`
# codegen, while `RunnerSpec` accepted it happily.
# `test_runtime_wire.py::test_the_dtype_table_uses_the_same_SPELLING_as_the_
# runner_and_the_generator` binds the three tables so they cannot drift again.
DTYPE_ID = {"fp32": 0, "fp16": 1, "q4_0": 2, "int32": 3, "q8_0": 4}
LAYOUT_ID = {"row_major": 0, "tiled_32x32": 1, "q4_0_repacked": 2}

_HDR = "<10I"
_BUF = "<QQII"
_TENSOR = "<11I"
_OP = f"<II{MAX_PARAMS}i{MAX_SRC}H{MAX_DST}H"
_RSP_HDR = "<IIIIQII"
_RESULT = "<IIQ"

HDR_SIZE = struct.calcsize(_HDR)
BUF_SIZE = struct.calcsize(_BUF)
TENSOR_SIZE = struct.calcsize(_TENSOR)
OP_SIZE = struct.calcsize(_OP)
RSP_HDR_SIZE = struct.calcsize(_RSP_HDR)
RESULT_SIZE = struct.calcsize(_RESULT)


class WireError(Exception):
    pass


@dataclass(frozen=True)
class BufDesc:
    fd: int
    size: int
    flags: int = 0
    # NO `base`. See the module docstring.


@dataclass(frozen=True)
class TensorDesc:
    bi: int
    offset: int
    nbytes: int
    dtype: str
    layout: str
    ne: tuple[int, int, int, int]
    # NO `data`. See the module docstring.


@dataclass(frozen=True)
class OpDesc:
    kind: int
    params: tuple[int, ...] = ()
    src: tuple[int, ...] = ()
    dst: tuple[int, ...] = ()
    flags: int = 0


@dataclass(frozen=True)
class OpResult:
    kind: int
    status: int
    cycles: int

    @property
    def ok(self) -> bool:
        return self.status == STATUS["OK"]


@dataclass(frozen=True)
class BatchResponse:
    status: int
    n_ops: int
    cycles_total: int
    arch: int
    results: tuple[OpResult, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return self.status == STATUS["OK"] and all(r.ok for r in self.results)


def pack_batch(bufs, tensors, ops) -> bytes:
    """Serialize a batch. Refuses malformed input HERE, on the host, where the
    error message can be read -- rather than shipping it to a DSP that can only
    answer with a status code."""
    if len(bufs) > MAX_BUFS:
        raise WireError(f"{len(bufs)} buffers exceeds HEXLIB_MAX_BUFS ({MAX_BUFS})")
    if len(tensors) > MAX_TENSORS:
        raise WireError(f"{len(tensors)} tensors exceeds {MAX_TENSORS}")

    for i, t in enumerate(tensors):
        if not 0 <= t.bi < len(bufs):
            raise WireError(
                f"tensor {i} names buffer index {t.bi}, but the batch declares "
                f"{len(bufs)} buffers"
            )
        end = t.offset + t.nbytes
        if end > bufs[t.bi].size:
            raise WireError(
                f"tensor {i} runs past the end of buffer {t.bi}: "
                f"offset {t.offset} + {t.nbytes} bytes = {end} > {bufs[t.bi].size}"
            )
        if t.dtype not in DTYPE_ID:
            raise WireError(f"tensor {i} has unknown dtype {t.dtype!r}")
        if t.layout not in LAYOUT_ID:
            raise WireError(f"tensor {i} has unknown layout {t.layout!r}")

    for i, op in enumerate(ops):
        if len(op.params) > MAX_PARAMS:
            raise WireError(f"op {i} has {len(op.params)} params, max {MAX_PARAMS}")
        if len(op.src) > MAX_SRC or len(op.dst) > MAX_DST:
            raise WireError(f"op {i} exceeds MAX_SRC/MAX_DST")
        # THE SUM, WHICH THE TWO CHECKS ABOVE DO NOT COVER. MAX_SRC + MAX_DST is
        # 10 and MAX_BUFS is 8: the DSP walks src then dst into ONE `a->buf[]`
        # array (`skel_dispatch.c`) and abandons the op at
        # `nb >= HEXLIB_MAX_BUFS`, which surfaces as a bare batch status
        # INVAL_PARAMS with no way to tell it from any other cause. Named here
        # with the actual counts instead.
        if len(op.src) + len(op.dst) > MAX_BUFS:
            raise WireError(
                f"op {i} names {len(op.src)} sources + {len(op.dst)} "
                f"destinations = {len(op.src) + len(op.dst)} buffers, but "
                f"hexlib_args.buf[] holds HEXLIB_MAX_BUFS ({MAX_BUFS}); the DSP "
                f"would abandon the op and report only INVAL_PARAMS"
            )
        for j in tuple(op.src) + tuple(op.dst):
            if not 0 <= j < len(tensors):
                raise WireError(f"op {i} names tensor {j}, out of range")

    off_bufs = HDR_SIZE
    off_tensors = off_bufs + BUF_SIZE * len(bufs)
    off_ops = off_tensors + TENSOR_SIZE * len(tensors)
    total = off_ops + OP_SIZE * len(ops)

    out = bytearray()
    out += struct.pack(
        _HDR, BATCH_MAGIC, BATCH_VERSION, total, len(bufs), len(tensors),
        len(ops), off_bufs, off_tensors, off_ops, 0,
    )
    for b in bufs:
        # base = 0: the DSP resolves it from its own mmap table, by fd.
        out += struct.pack(_BUF, 0, b.size, b.fd, b.flags)
    for t in tensors:
        out += struct.pack(
            _TENSOR, t.bi, t.offset, t.nbytes, DTYPE_ID[t.dtype],
            LAYOUT_ID[t.layout], *t.ne, 0, 0,  # data = 0, pad
        )
    for op in ops:
        params = tuple(op.params) + (0,) * (MAX_PARAMS - len(op.params))
        src = tuple(op.src) + (0xFFFF,) * (MAX_SRC - len(op.src))
        dst = tuple(op.dst) + (0xFFFF,) * (MAX_DST - len(op.dst))
        out += struct.pack(_OP, op.kind, op.flags, *params, *src, *dst)

    assert len(out) == total, "header's total_size must equal the blob length"
    return bytes(out)


def unpack_response(raw: bytes) -> BatchResponse:
    if len(raw) < RSP_HDR_SIZE:
        raise WireError(f"response truncated: {len(raw)} < {RSP_HDR_SIZE} bytes")
    magic, version, status, n_ops, cycles, arch, _ = struct.unpack_from(_RSP_HDR, raw)
    if magic != BATCH_MAGIC:
        raise WireError(
            f"response magic 0x{magic:08x} != 0x{BATCH_MAGIC:08x} — the DSP wrote "
            "nothing, or wrote something else"
        )
    if version != BATCH_VERSION:
        raise WireError(f"response version {version} != {BATCH_VERSION}")
    if status not in STATUS_NAME:
        raise WireError(f"response status {status} is not a known status")
    need = RSP_HDR_SIZE + RESULT_SIZE * n_ops
    if len(raw) < need:
        raise WireError(
            f"response truncated: claims {n_ops} results ({need} bytes), "
            f"carries {len(raw)}"
        )
    results = tuple(
        OpResult(*struct.unpack_from(_RESULT, raw, RSP_HDR_SIZE + i * RESULT_SIZE))
        for i in range(n_ops)
    )
    return BatchResponse(status, n_ops, cycles, arch, results)

"""Layout as an enumerated value, never a stride pattern.

Strides cannot express the central object of this design -- a VTCM-resident
tile of a DDR tensor is not a stride view; it is a different buffer, at a
different address, possibly in a different physical arrangement, reached by a
DMA. And for q4_0, `nb[0]` is 18 bytes per BLOCK of 32, not a per-element
stride, so a quantized tensor cannot be transposed or arbitrarily sliced. The
generality of strides evaporates exactly where our data lives.

The payoff is that layout mismatch becomes a plan-time error: each op kind
declares which layouts its kernel accepts, and a pass handing it the wrong one
fails here instead of producing plausible, wrong numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from hexlib.graph.ir import Q4_0_BLOCK, Q4_0_BLOCK_BYTES, nbytes


class Layout(Enum):
    DENSE = "dense"
    """Row-major contiguous in logical order. What numpy gives you."""

    Q4_0_BLOCKED = "q4_0_blocked"
    """32-element blocks, each 16 bytes of nibbles + one fp16 scale = 18 bytes.
    This is what a GGUF file holds and what the quantizer emits."""

    Q4_0_REPACKED = "q4_0_repacked"
    """The same bytes, reordered so a 128-byte HVX load lands on data the
    kernel wants together. ggml-hexagon requires this and rejects unrepacked
    weights (`ggml_backend_buffer_is_hexagon_repack`); it is not an
    optimization, it is admission."""

    HMX_TILE_F16 = "hmx_tile_f16"
    """32x32 = 1024 elements = 2048 bytes. In fp16 mode BOTH operands use this
    same shape -- matmul-ops.h:16-19."""

    HMX_ACT_TILE_I8 = "hmx_act_tile_i8"
    """int8 mode activation tile: 2048 B, two bytes per element. UNUSED."""

    HMX_WGT_TILE_I8 = "hmx_wgt_tile_i8"
    """int8 mode weight tile: 1024 B, one byte per element. UNUSED.

    The asymmetry with HMX_ACT_TILE_I8 is not guessable -- four probe rounds in
    forge2 concluded the opposite before measurement settled it. See
    docs/hardware/hmx-int8.md. hexlib uses HMX fp16 mode, which is symmetric;
    these two are retained because the recorded geometry is worth keeping if
    activation quantization is ever taken up."""


class Buffer(Enum):
    DDR = "ddr"
    """Large, slow, across a bus with a modelled penalty."""

    VTCM = "vtcm"
    """Fast, explicitly managed, and small. Its size is a RUNTIME value: VTCM
    is acquired at session start, so the part total is not the usable budget."""

    DDR_L2PREFETCH = "ddr_l2fetch"
    """Stays in DDR; the plan emits an l2fetch ahead of use. Present in the
    enum and DELIBERATELY NOT IMPLEMENTED in M1 -- spec 4.3.1. This encoder's
    matmuls run at roughly 450 MACs per byte and are compute-bound, and its one
    bandwidth-bound op (GELU on [n, 3072]) is better served by fusing it into
    the matmul epilogue, which removes the memory pass rather than speeding it
    up. If it is ever implemented: an l2fetch that is issued and moves nothing
    is a measured, real failure (L2FETCH_COMMAND_KILLED=1 with
    L2FETCH_ACCESS=0), so effectiveness must be read from PMU counters and
    never inferred from the disassembly."""


@dataclass(frozen=True)
class HmxTile:
    rows: int
    cols: int

    @property
    def elements(self) -> int:
        return self.rows * self.cols

    @property
    def nbytes_f16(self) -> int:
        return self.elements * 2


# htp/matmul-ops.h:16-19. Symmetric across both operands in fp16 mode.
HMX_TILE = HmxTile(rows=32, cols=32)

# The number of consecutive q4_0 blocks a repack groups together. Provisional:
# it must be confirmed against a concrete kernel before any weight is repacked
# for real (spec 5.1, "repacking remains to be specified against a concrete
# kernel"). It does not affect byte counts -- repacking reorders, never resizes
# -- so M1's plans are correct regardless.
Q4_0_REPACK_GROUP = 8


@dataclass(frozen=True)
class Placement:
    layout: Layout
    perm: tuple[int, ...]
    buffer: Buffer
    offset: int
    """Bytes into that buffer. A number, never a pointer -- that is what makes
    a plan serializable, and it is right independent of everything else."""

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError(f"placement offset must be non-negative, got {self.offset}")
        if sorted(self.perm) != list(range(len(self.perm))):
            raise ValueError(f"perm {self.perm} is not a permutation")


def layout_nbytes(shape: tuple[int, ...], dtype: str, layout: Layout) -> int:
    """Bytes this tensor occupies when arranged in this layout."""
    if layout in (Layout.DENSE, Layout.Q4_0_BLOCKED, Layout.Q4_0_REPACKED):
        # Repacking reorders bytes; it never changes how many there are.
        return nbytes(shape, dtype)

    if layout is Layout.HMX_TILE_F16:
        if dtype != "fp16":
            raise ValueError(
                f"HMX_TILE_F16 requires an fp16 tensor, got {dtype!r}. hexlib uses "
                "HMX's fp16 mode; the int8 mode needs int8 activations too, which "
                "is out of scope."
            )
        return _tiled_bytes(shape, bytes_per_element=2)

    if layout is Layout.HMX_ACT_TILE_I8:
        return _tiled_bytes(shape, bytes_per_element=2)
    if layout is Layout.HMX_WGT_TILE_I8:
        return _tiled_bytes(shape, bytes_per_element=1)

    raise ValueError(f"no size rule for layout {layout!r}")


def _tiled_bytes(shape: tuple[int, ...], bytes_per_element: int) -> int:
    if len(shape) < 2:
        raise ValueError(f"a tiled layout needs a 2-D or higher shape, got {shape}")
    rows = math.ceil(shape[-2] / HMX_TILE.rows)
    cols = math.ceil(shape[-1] / HMX_TILE.cols)
    batch = math.prod(shape[:-2]) if len(shape) > 2 else 1
    return batch * rows * cols * HMX_TILE.elements * bytes_per_element


# Per op kind, per input, the layouts that kind's kernel accepts.
# An entry here is a claim about a kernel. When a kernel lands, this table is
# what the kernel's spec.json must agree with.
ACCEPTED_LAYOUTS: Mapping[str, tuple[tuple[Layout, ...], ...]] = {
    "matmul": (
        (Layout.DENSE, Layout.HMX_TILE_F16),
        (Layout.Q4_0_REPACKED, Layout.DENSE, Layout.HMX_TILE_F16),
    ),
    "matmul_epilogue": (
        (Layout.DENSE, Layout.HMX_TILE_F16),
        (Layout.Q4_0_REPACKED, Layout.DENSE, Layout.HMX_TILE_F16),
        (Layout.DENSE,),
    ),
    "add": ((Layout.DENSE,), (Layout.DENSE,)),
    "scale": ((Layout.DENSE,),),
    "layernorm": ((Layout.DENSE,), (Layout.DENSE,), (Layout.DENSE,)),
    "gelu_tanh": ((Layout.DENSE,),),
    "gelu_erf": ((Layout.DENSE,),),
    "softmax": ((Layout.DENSE,),),
    "transpose": ((Layout.DENSE,),),
    "reshape": ((Layout.DENSE,),),
    "patchify": ((Layout.DENSE,),),
    "rope_2d": ((Layout.DENSE,), (Layout.DENSE,), (Layout.DENSE,)),
}


def check_layouts(kind: str, placements: Sequence[Placement]) -> list[str]:
    """Problems with handing these placements to this op kind. Empty means fine.

    An unknown kind is a problem, not a pass. A kernel whose accepted layouts
    nobody declared must not be handed arbitrary data.
    """
    accepted = ACCEPTED_LAYOUTS.get(kind)
    if accepted is None:
        return [
            f"op kind {kind!r} declares no accepted layouts; add an entry to "
            "ACCEPTED_LAYOUTS rather than letting it accept anything"
        ]
    if len(placements) != len(accepted):
        return [
            f"op kind {kind!r} accepts {len(accepted)} inputs but was given "
            f"{len(placements)} placements"
        ]
    problems: list[str] = []
    for i, (placement, choices) in enumerate(zip(placements, accepted)):
        if placement.layout not in choices:
            problems.append(
                f"op kind {kind!r} input {i} is laid out as "
                f"{placement.layout.name} but its kernel accepts "
                f"{', '.join(c.name for c in choices)}"
            )
    return problems

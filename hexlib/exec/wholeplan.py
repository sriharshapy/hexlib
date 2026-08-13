"""ONE batch blob carrying EVERY op of a compiled plan -- a single entry point
into the DSP, and a single exit.

WHY THIS EXISTS. `hexlib/exec/dsp.py`'s `DspSimBackend` packs one op per batch
and launches the simulator once per op. That is a correctness path and it works,
but it costs a process start, a QuRT boot and a tear-down per op -- measured at
~7 s of pure overhead on this machine, before any compute. Across the encoder's
259 real-work ops that is half an hour of transport alone, and on a DEVICE the
equivalent per-op round trip is the reason nothing can hold a plan.

The wire was built for this from the start and nothing here extends it:
`wire.pack_batch(bufs, tensors, ops)` already takes a LIST of ops, and
`skel_dispatch.c:171` already loops `for (i = 0; i < hdr.n_ops; i++)` filling a
per-op status and cycle count. What was missing was a host-side builder that
emits the whole plan at once. This is that builder, and it is host-only: no C
changed, no IDL changed, no new kernel.

ONE BUFFER, NOT SEVERAL. Every tensor lands in a single arena at `bi = 0`.
`MAX_BUFS` (8) is NOT an arena limit -- it is the size of the DSP's per-op
`hexlib_args.buf[]` array, which `pack_batch` already checks as
`len(src) + len(dst)`. A `BufDesc` is a mapped fd, so one fd for the whole plan
is both legal and what a device wants: one `rpcmem_alloc`, one `fastrpc_mmap`.

THE ARENA'S THREE REGIONS, and the reason they are not one:

  consts   -- every weight and bias, each at its own offset. Written once,
              never aliased, read throughout. At 256x256 this is 56.9 MB and it
              dominates the allocation.
  activations -- at the PLAN'S OWN VTCM slot offsets, which alias: 88 slots at
              the tiny config share far less space than their total size,
              because `Slot.first_use`/`last_use` say when each is dead. Reusing
              the plan's offsets rather than inventing new ones means this path
              executes the allocator's decisions instead of second-guessing
              them -- and an aliasing bug shows up as a wrong answer, which is
              exactly what `test_aliased_slots_actually_corrupt_a_value` proves
              a name-keyed dict cannot catch.
  io       -- the graph's declared inputs and outputs, each at its own offset,
              never aliased. The host writes the input and reads the output, so
              these must survive the whole run whatever liveness says.

RESHAPE EMITS NO OP. 49 of the plan's 308 steps are reshapes, and in row-major a
reshape is a pure reinterpretation: same bytes, different `ne`. So the output
tensor is placed AT ITS INPUT'S OFFSET and no op is emitted -- zero copies, zero
kernels, and `ne` differs between the two descriptors, which is the whole
content of the operation. Doing so BREAKS the plan's VTCM slot offsets --
see `_refuse_unsafe_aliasing` at the bottom of this file, which is why
`alias_activations` defaults to False and refuses when asked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from hexlib.exec.dsp import _align_up, _encode_params, _ne
from hexlib.graph.ir import nbytes as ir_nbytes
from hexlib.exec.runner import WIRE_DTYPE, WIRE_RAW, RawTensor, select
from hexlib.runtime.genentry import KIND_ID
from hexlib.runtime import wire


class WholePlanError(Exception):
    pass


@dataclass(frozen=True)
class Placement:
    """Where one tensor lives in the arena, and how it is described on the wire."""
    name: str
    offset: int
    nbytes: int
    dtype: str
    layout: str
    ne: tuple[int, int, int, int]
    region: str          # "const" | "act" | "io" | "reshape-alias"


@dataclass
class WholePlanBatch:
    blob: bytes
    payload: bytearray
    placements: dict[str, Placement]
    index: dict[str, int]            # tensor name -> wire tensor index
    n_ops: int
    arena_bytes: int
    outputs: tuple[str, ...]
    skipped_reshapes: int
    op_names: tuple[str, ...] = field(default=())


def build_whole_plan_batch(compiled, feeds: Mapping[str, Any],
                           alias_activations: bool = False) -> WholePlanBatch:
    """Pack `compiled`'s entire plan into one batch, with `feeds` staged in.

    `feeds` supplies every graph input and every const, exactly as
    `interpreter.run` takes them. A const declared `q4_0` is quantized here if it
    arrives as a dense array, the same way `dsp.interpreter_backends` does it, so
    the caller passes the same feeds to both paths and the comparison is of
    arithmetic rather than of weight values.
    """
    from hexlib.exec.quant import quantize_q4_0

    graph, plan = compiled.graph, compiled.plan
    tensors = graph.tensors

    # --- 1. Which region does each tensor belong to? -----------------------
    io_names = set(graph.inputs) | set(graph.outputs)
    const_names = {n for n, t in tensors.items() if t.const}

    # A reshape's output is its input reinterpreted. Resolve chains (a reshape
    # feeding a reshape) to the ultimate storage owner before anything is placed.
    reshape_src: dict[str, str] = {}
    for step in plan.steps:
        op = getattr(step, "op", None)
        if op is not None and op.kind == "reshape":
            reshape_src[op.outputs[0]] = op.inputs[0]

    def storage_owner(name: str) -> str:
        seen = {name}
        while name in reshape_src:
            name = reshape_src[name]
            if name in seen:
                raise WholePlanError(f"reshape chain cycles at {name!r}")
            seen.add(name)
        return name

    # An io tensor must never be aliased away by a reshape -- the host has to be
    # able to read the declared output at a stable address.
    aliased_io = {n for n in reshape_src if n in io_names}
    if aliased_io:
        raise WholePlanError(
            f"these declared graph inputs/outputs are reshape outputs and would "
            f"be aliased onto another tensor's storage: {sorted(aliased_io)}. "
            f"Give them their own placement before using this path."
        )

    slot_offset = {s.tensor: s.offset for s in plan.vtcm}

    # --- 1b. LAYOUT IS A PROPERTY OF THE OP'S BUFFER SLOT, NOT OF THE TENSOR.
    # `spec.buf_layouts()` returns one layout per buffer in src-then-dst order,
    # so the same q4_0 weight is `row_major` for a kernel that reads plain
    # blocks and `q4_0_repacked` for one that wants tiles. Deriving it from the
    # DTYPE instead -- which this file did first -- sends `q4_0_repacked` for
    # every quantized weight and the DSP's `_layout_check` rejects the op with
    # ERR_REQUIRES, correctly. A tensor consumed by two ops that disagree is a
    # real defect (it would need repacking between them), so it is refused here
    # rather than resolved by picking one.
    layout_of: dict[str, str] = {}
    dtype_of: dict[str, str] = {}

    def claim_layout(name: str, layout: str, where: str) -> None:
        prev = layout_of.setdefault(name, layout)
        if prev != layout:
            raise WholePlanError(
                f"tensor {name!r} is used as {prev!r} and as {layout!r} (at "
                f"{where}); one arena cannot hold both without a repack step"
            )

    # THE SPEC'S DTYPE, NOT THE GRAPH'S, decides the bytes in the arena.
    # `dsp.py`'s per-op path coerces each array to `spec.inputs[i]` on the way
    # out (`np.ascontiguousarray(a, dtype=WIRE_DTYPE[dt])`) and declares that
    # dtype on the wire. A whole-plan arena has no such per-call moment, so the
    # coercion has to happen once, at staging. `pos_embed` is the case that
    # forced this: the graph declares it fp32, `add` declares both inputs fp16,
    # and staging the graph's dtype makes the DSP reject the op with
    # ERR_REQUIRES -- correctly, because a kernel reading fp32 bytes as fp16
    # would otherwise return a correctly-shaped wrong answer.
    def claim_dtype(name: str, dtype: str, where: str) -> None:
        prev = dtype_of.setdefault(name, dtype)
        if prev != dtype:
            raise WholePlanError(
                f"tensor {name!r} is read as {prev!r} and as {dtype!r} (at "
                f"{where}); one arena slot cannot hold both, and an implicit "
                f"conversion here would be invisible to every downstream check"
            )

    op_specs: dict[int, tuple[str, Any]] = {}
    for step in plan.steps:
        op = getattr(step, "op", None)
        if op is None or op.kind == "reshape":
            continue
        spec_name, spec = select(op.kind, dict(op.attrs))
        spec.check_requires(dict(op.attrs))
        op_specs[op.id] = (spec_name, spec)
        buf_layouts = spec.buf_layouts()
        for i, n in enumerate(op.inputs):
            claim_layout(n, buf_layouts[i], f"{op.id}:{spec_name} src{i}")
            claim_dtype(n, spec.inputs[i], f"{op.id}:{spec_name} src{i}")
        for n in op.outputs:
            claim_layout(n, buf_layouts[-1], f"{op.id}:{spec_name} dst")
            claim_dtype(n, spec.out_dtype, f"{op.id}:{spec_name} dst")

    # --- 2. Lay the arena out ---------------------------------------------
    placements: dict[str, Placement] = {}
    cursor = 0

    def wire_dtype(name: str) -> str:
        return dtype_of.get(name, tensors[name].dtype)

    def wire_nbytes(name: str) -> int:
        return ir_nbytes(tensors[name].shape, wire_dtype(name))

    def place(name: str, offset: int, region: str) -> None:
        t = tensors[name]
        placements[name] = Placement(
            name=name, offset=offset, nbytes=wire_nbytes(name),
            dtype=wire_dtype(name), layout=layout_of.get(name, "row_major"),
            ne=_ne(t.shape), region=region,
        )

    for name in sorted(const_names):
        place(name, cursor, "const")
        cursor = _align_up(cursor + wire_nbytes(name))
    const_end = cursor

    act_names = [
        n for n in tensors
        if n not in const_names and n not in io_names and n not in reshape_src
    ]
    retyped = [n for n in act_names if wire_dtype(n) != tensors[n].dtype]
    if retyped:
        raise WholePlanError(
            f"these activations would be staged in a dtype other than the one "
            f"the plan sized their VTCM slot with, so every offset after them "
            f"is wrong: {sorted(retyped)[:8]}"
        )
    missing_slot = [n for n in act_names if n not in slot_offset]
    if missing_slot:
        raise WholePlanError(
            f"{len(missing_slot)} activation tensors have no VTCM slot in the "
            f"plan, so this builder has no offset for them: "
            f"{sorted(missing_slot)[:8]}"
        )
    if alias_activations:
        # WHY THIS IS GUARDED AND OFF BY DEFAULT. Reusing the plan's VTCM slot
        # offsets as flat-arena addresses is only safe if the allocator's
        # disjointness guarantee still holds, and RESHAPE ELISION BREAKS IT.
        # The plan gives a reshape output its own slot; this builder puts it on
        # its input's storage instead, so the input's slot stays live longer
        # than the allocator was told and whatever legitimately owns that
        # address in the meantime is overwritten. Measured at the tiny config:
        # correlation with the reference fell from 1.000000 to 0.277.
        #
        # The allocator is NOT at fault -- checked directly, zero of its 88
        # slots overlap in live range. Making this mode correct means unifying
        # reshape chains into one tensor BEFORE allocation, which is a plan-pass
        # change, not a change here. Until then the check below refuses rather
        # than silently returning a corrupted arena, because a wrong answer from
        # this path looks exactly like a kernel bug.
        _refuse_unsafe_aliasing(plan, reshape_src, slot_offset, tensors,
                                act_names, wire_nbytes)
        act_span = max(
            (slot_offset[n] + wire_nbytes(n) for n in act_names), default=0
        )
        for name in act_names:
            place(name, const_end + slot_offset[name], "act")
        cursor = _align_up(const_end + act_span)
    else:
        # NO ALIASING: every activation gets its own address. Costs memory and
        # exercises none of the allocator, but it is the control case -- if the
        # answer is right here and wrong with aliasing on, the defect is in the
        # liveness the slots encode and not in the kernels or the wire.
        cursor = const_end
        for name in sorted(act_names):
            place(name, cursor, "act")
            cursor = _align_up(cursor + wire_nbytes(name))

    for name in sorted(io_names):
        place(name, cursor, "io")
        cursor = _align_up(cursor + wire_nbytes(name))

    # Reshape outputs borrow their owner's offset but keep their OWN ne.
    for name in reshape_src:
        owner = storage_owner(name)
        if owner not in placements:
            raise WholePlanError(f"reshape {name!r} resolves to unplaced {owner!r}")
        t = tensors[name]
        base = placements[owner]
        if wire_nbytes(name) != base.nbytes:
            raise WholePlanError(
                f"reshape {name!r} is {wire_nbytes(name)} bytes but its owner "
                f"{owner!r} is {base.nbytes}; a reshape must preserve byte count"
            )
        placements[name] = Placement(
            name=name, offset=base.offset, nbytes=wire_nbytes(name),
            dtype=wire_dtype(name), layout=layout_of.get(name, "row_major"),
            ne=_ne(t.shape), region="reshape-alias",
        )

    arena_bytes = _align_up(cursor)

    # --- 3. Stage the feeds -----------------------------------------------
    payload = bytearray(arena_bytes)
    for name in sorted(const_names) + sorted(graph.inputs):
        if name not in feeds:
            raise WholePlanError(f"no feed supplied for {name!r}")
        p = placements[name]
        value = feeds[name]
        if p.dtype in WIRE_RAW:
            raw = value if isinstance(value, RawTensor) else RawTensor(
                p.dtype, tuple(np.asarray(value).shape),
                quantize_q4_0(np.asarray(value)),
            )
            data = raw.data
        else:
            data = np.ascontiguousarray(
                np.asarray(value), dtype=WIRE_DTYPE[p.dtype]
            ).tobytes()
        if len(data) != p.nbytes:
            raise WholePlanError(
                f"feed {name!r} staged {len(data)} bytes, but the graph declares "
                f"{p.nbytes} -- a dtype or shape disagreement, not a rounding one"
            )
        payload[p.offset:p.offset + p.nbytes] = data

    # --- 4. Wire descriptors ----------------------------------------------
    order = sorted(placements)
    index = {name: i for i, name in enumerate(order)}
    wire_tensors = [
        wire.TensorDesc(
            bi=0, offset=placements[n].offset, nbytes=placements[n].nbytes,
            dtype=placements[n].dtype, layout=placements[n].layout,
            ne=placements[n].ne,
        )
        for n in order
    ]

    ops: list[wire.OpDesc] = []
    op_names: list[str] = []
    skipped = 0
    for step in plan.steps:
        op = getattr(step, "op", None)
        if op is None:
            continue
        if op.kind == "reshape":
            skipped += 1
            continue
        spec_name, spec = op_specs[op.id]
        stand_ins = tuple(_ShapeOnly(tensors[n].shape) for n in op.inputs)
        ops.append(wire.OpDesc(
            kind=KIND_ID[spec_name],
            params=_encode_params(spec, stand_ins, dict(op.attrs)),
            src=tuple(index[n] for n in op.inputs),
            dst=tuple(index[n] for n in op.outputs),
        ))
        op_names.append(f"{op.id}:{spec_name}")

    bufs = [wire.BufDesc(fd=0, size=arena_bytes)]
    blob = wire.pack_batch(bufs, wire_tensors, ops)

    return WholePlanBatch(
        blob=blob, payload=payload, placements=placements, index=index,
        n_ops=len(ops), arena_bytes=arena_bytes,
        outputs=tuple(graph.outputs), skipped_reshapes=skipped,
        op_names=tuple(op_names),
    )


@dataclass(frozen=True)
class _ShapeOnly:
    """`_encode_params` only ever reads `.shape` for attr-sourced scalars, and
    `numel:`/`dim:` are derived on the DSP from `ne[]`. Passing real arrays here
    would mean materializing every intermediate on the host, which is precisely
    what this path exists to avoid."""
    shape: tuple[int, ...]


def _refuse_unsafe_aliasing(plan, reshape_src, slot_offset, tensors,
                            act_names, wire_nbytes) -> None:
    """Raise unless every pair of activations sharing an address is disjoint in
    time ONCE reshape aliasing is accounted for.

    The plan's own slots satisfy this by construction. What this checks is the
    property AFTER this builder has moved reshape outputs onto their inputs,
    which is the step that can violate it.
    """
    live = {s.tensor: [s.first_use, s.last_use] for s in plan.vtcm}

    # A reshape output's storage is its input's, so the input must be treated as
    # live for the union of both ranges.
    for out, inp in reshape_src.items():
        if out in live and inp in live:
            live[inp][0] = min(live[inp][0], live[out][0])
            live[inp][1] = max(live[inp][1], live[out][1])

    placed = [(n, slot_offset[n], wire_nbytes(n)) for n in act_names
              if n in slot_offset and n in live]
    clashes = []
    for i in range(len(placed)):
        ni, oi, si = placed[i]
        for j in range(i + 1, len(placed)):
            nj, oj, sj = placed[j]
            if oi < oj + sj and oj < oi + si:          # byte ranges overlap
                a, b = live[ni], live[nj]
                if a[0] <= b[1] and b[0] <= a[1]:      # and so do live ranges
                    clashes.append((ni, a, nj, b))
    if clashes:
        detail = "; ".join(
            f"{n1}[{a[0]},{a[1]}] vs {n2}[{b[0]},{b[1]}]"
            for n1, a, n2, b in clashes[:5]
        )
        raise WholePlanError(
            f"{len(clashes)} activation pairs share an address while both are "
            f"live, once reshape aliasing is folded in: {detail}. This arena "
            f"would compute a wrong answer that looks like a kernel bug. Use "
            f"alias_activations=False, or unify reshape chains before the VTCM "
            f"pass so the allocator sees one tensor instead of two."
        )

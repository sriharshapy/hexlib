"""Pass: tile the weight-consuming ops and place their DMA.

Tiling is the mechanism from day one, at every resolution -- not a
large-resolution special case that 256x256 lets us defer. The whole encoder's
weights are about 55.5 MB at q4_0 against an 8 MB VTCM, and the reference never
makes a layer resident at any size: it streams 576-byte chunks
(matmul-ops.c:2653-2655), dequantizes per chunk in-kernel, and double-buffers
against compute (matmul-ops.c:2744-2750).

So VTCM is a working buffer, not a layer cache. The resolution choice affects
how much pressure there is, not whether the mechanism is needed.

This pass is also the ONLY place a Plan's DMA is placed at all, so it is
responsible for the whole traffic picture, not just the weights: the graph's
own input (e.g. the image) has to be fetched from DDR before anything can
read it, and the graph's own output has to be written back to DDR after the
op that produces it -- an executor replaying the plan has no other
instruction that would do either.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from hexlib.graph.ir import Graph, Op, Tensor
from hexlib.graph.layout import Buffer, Layout, Placement, check_layouts
from hexlib.graph.plan import Slot, Step, Tiling, Transfer
from hexlib.result import Err

WEIGHT_CHUNK_BYTES = 576
"""The chunk size ggml-hexagon's matmul streams weights in, matmul-ops.c:2653-2655.

Not a round number, and not ours to change without measuring. It is here as a
named constant so a future measurement changes one line."""

WEIGHT_BUFFERS = 2
"""Double buffering. The DMA engine can fetch the next chunk while HVX or HMX
works the current one; that throughput is forfeited unless it is scheduled."""

TILED_KINDS = ("matmul", "matmul_epilogue")

_WEIGHT_SCRATCH_NAME = "__weight_scratch__"
"""Not a graph tensor. This names the reserved, rotating double-buffer region
itself -- the resource many different weight tensors take turns occupying --
so it can carry its own `Slot` in `plan.vtcm` instead of being invisible."""


def matmul_working_set(inputs, outputs, attrs) -> int:
    """Resident bytes for a tiled matmul: activation, output, and two chunks.

    Replaces M0's "all inputs plus all outputs", which for a
    [256, 3072] x [3072, 3072] matmul claimed about 11 MB resident when the
    real figure is the activation, the output, and 1152 bytes of weight.
    """
    if not inputs or not outputs:
        raise ValueError(
            "matmul_working_set needs at least one input and one output; got "
            f"{len(inputs)} inputs and {len(outputs)} outputs"
        )
    activation = inputs[0].nbytes
    out = sum(t.nbytes for t in outputs)
    extra = sum(t.nbytes for t in inputs[2:])  # a fused bias, if present
    return activation + out + extra + WEIGHT_CHUNK_BYTES * WEIGHT_BUFFERS


def insert_transfers(
    graph: Graph,
    slots: Sequence[Slot],
    budget: int,
) -> tuple[tuple[Step, ...], int, tuple[Slot, ...]] | Err:
    """Turn an ordered graph plus an allocation into steps with DMA placed.

    Returns the steps, the predicted DDR<->VTCM bytes moved, and the `Slot`s
    for the const region (the weight double-buffer plus every resident
    const) so the caller can fold them into `plan.vtcm` and recompute the
    plan's true VTCM high water mark over ALL slots, not activations alone.
    """
    problems = graph.problems()
    if problems:
        return Err("graph is structurally invalid", "\n".join(problems))

    if budget < WEIGHT_CHUNK_BYTES * WEIGHT_BUFFERS:
        return Err(
            "VTCM budget too small to double-buffer a weight chunk",
            f"budget {budget} is below {WEIGHT_CHUNK_BYTES * WEIGHT_BUFFERS} bytes, "
            f"which is {WEIGHT_BUFFERS} chunks of {WEIGHT_CHUNK_BYTES}",
        )

    high_water = max((s.end for s in slots), default=0)
    scratch_base = _align(high_water)
    weight_region_bytes = WEIGHT_CHUNK_BYTES * WEIGHT_BUFFERS
    if scratch_base + weight_region_bytes > budget:
        return Err(
            "no VTCM left for weight streaming",
            f"activations reach {high_water} bytes and the budget is {budget}; "
            f"there is no room for {WEIGHT_BUFFERS} chunks of "
            f"{WEIGHT_CHUNK_BYTES} bytes",
        )

    # Every const tensor's VTCM offset, decided once up front. A tiled
    # weight rotates through the double-buffered region at `scratch_base`;
    # every other const is small enough to move once and stay resident for
    # the rest of the plan, so each gets its OWN offset past that region.
    # Sharing an offset between two simultaneously-resident buffers means the
    # second transfer silently overwrites the first before its op reads it.
    offsets, const_region_end = _plan_offsets(graph, scratch_base)
    if const_region_end > budget:
        const_region_start = _align(scratch_base + weight_region_bytes)
        return Err(
            "no VTCM left for the plan's resident consts",
            f"the non-tiled consts need {const_region_end - const_region_start} "
            f"bytes past the weight buffer, reaching {const_region_end}, but the "
            f"budget is {budget}",
        )

    slot_by_tensor = {s.tensor: s for s in slots}
    last_step = len(graph.ops)

    steps: list[Step] = []
    moved = 0
    next_id = 0
    # Tracks which non-weight tensors already got their one-shot transfer, so
    # a tensor read by many ops (e.g. the RoPE cos/sin tables, read by every
    # block) is moved once, not once per reader. Seeded empty: graph inputs
    # are no longer assumed already resident (Critical 1) -- they earn their
    # entry here exactly like a const does.
    transferred: set[str] = set()

    # A graph output produced but never consumed again still needs writing
    # back; track which outputs still need their DMA-out so a defensive pass
    # below can catch one whose producer this loop somehow missed.
    pending_outputs = set(graph.outputs)

    for op in graph.ops:
        try:
            placements = _placements_for(graph, op, offsets)
        except KeyError as e:
            return Err(
                "const tensor missing a planned VTCM offset",
                f"op {op.id} ({op.kind}) reads {e.args[0]!r}, which "
                "_plan_offsets never assigned an offset to",
            )
        problems = check_layouts(op.kind, placements)
        if problems:
            return Err("layout mismatch at plan time", "\n".join(problems))

        weight_name = _weight_name(graph, op)
        tiled_transfer: Transfer | None = None
        tiling: Tiling | None = None

        if weight_name is not None:
            weight = graph.tensor(weight_name)
            total = weight.nbytes
            count = max(1, -(-total // WEIGHT_CHUNK_BYTES))
            per_chunk = min(WEIGHT_CHUNK_BYTES, total)
            tiled_transfer = Transfer(
                id=next_id,
                tensor=weight_name,
                direction="in",
                vtcm_offset=offsets[weight_name],
                nbytes=per_chunk,
                layout=_layout_for(weight),
            )
            next_id += 1
            tiling = Tiling(transfer=tiled_transfer, count=count, buffers=WEIGHT_BUFFERS)
            moved += per_chunk * count

        dma_in: list[Transfer] = []
        for name in op.inputs:
            if name == weight_name:
                continue
            t = graph.tensor(name)
            is_graph_input = name in graph.inputs
            if not t.const and not is_graph_input:
                continue
            if name in transferred:
                continue
            offset = offsets[name] if t.const else slot_by_tensor[name].offset
            dma_in.append(
                Transfer(
                    id=next_id,
                    tensor=name,
                    direction="in",
                    vtcm_offset=offset,
                    nbytes=t.nbytes,
                    layout=_layout_for(t),
                )
            )
            next_id += 1
            moved += t.nbytes
            transferred.add(name)

        dma_out: list[Transfer] = []
        for name in op.outputs:
            if name not in pending_outputs:
                continue
            t = graph.tensor(name)
            offset = slot_by_tensor[name].offset if name in slot_by_tensor else offsets[name]
            dma_out.append(
                Transfer(
                    id=next_id,
                    tensor=name,
                    direction="out",
                    vtcm_offset=offset,
                    nbytes=t.nbytes,
                    layout=_layout_for(t),
                )
            )
            next_id += 1
            moved += t.nbytes
            pending_outputs.discard(name)

        dma_in_tuple = tuple(dma_in)
        wait_ids = tuple(t.id for t in dma_in_tuple) + (
            (tiled_transfer.id,) if tiled_transfer is not None else ()
        )
        steps.append(
            Step(
                op=op,
                dma_in=dma_in_tuple,
                dma_wait=wait_ids,
                dma_out=tuple(dma_out),
                tiling=tiling,
            )
        )

    if not steps:
        return Err(
            "nothing to schedule",
            "the graph produced no steps; a schedule of nothing is not a schedule",
        )

    # Defensive: a graph input with no consumer, or a graph output somehow
    # not reached above, still gets its transfer rather than silently
    # violating "every graph input/output has a transfer" -- attach it to
    # the first/last step respectively.
    extra_in: list[Transfer] = []
    for name in graph.inputs:
        if name in transferred:
            continue
        t = graph.tensor(name)
        # A const graph input never read by any op has no planned offset --
        # an unusual graph (consts are normally baked in, not declared as
        # runtime inputs), and one this pass cannot place safely. Leave it
        # untransferred rather than guessing an offset; plan_problems'
        # "every graph input needs an in-transfer" invariant reports it.
        if t.const and name not in offsets:
            continue
        offset = offsets[name] if t.const else slot_by_tensor[name].offset
        extra_in.append(
            Transfer(
                id=next_id, tensor=name, direction="in", vtcm_offset=offset,
                nbytes=t.nbytes, layout=_layout_for(t),
            )
        )
        next_id += 1
        moved += t.nbytes
        transferred.add(name)
    if extra_in:
        first = steps[0]
        steps[0] = Step(
            op=first.op,
            dma_in=first.dma_in + tuple(extra_in),
            dma_wait=first.dma_wait + tuple(t.id for t in extra_in),
            dma_out=first.dma_out,
            tiling=first.tiling,
        )

    extra_out: list[Transfer] = []
    for name in pending_outputs:
        t = graph.tensor(name)
        # Same defensive reasoning as the const-input case above: a const
        # graph output produced by no op has no planned offset here either.
        if name not in slot_by_tensor and name not in offsets:
            continue
        offset = slot_by_tensor[name].offset if name in slot_by_tensor else offsets[name]
        extra_out.append(
            Transfer(
                id=next_id, tensor=name, direction="out", vtcm_offset=offset,
                nbytes=t.nbytes, layout=_layout_for(t),
            )
        )
        next_id += 1
        moved += t.nbytes
    if extra_out:
        last = steps[-1]
        steps[-1] = Step(
            op=last.op,
            dma_in=last.dma_in,
            dma_wait=last.dma_wait,
            dma_out=last.dma_out + tuple(extra_out),
            tiling=last.tiling,
        )

    const_slots = _const_slots(graph, offsets, scratch_base, weight_region_bytes, last_step)

    return tuple(steps), moved, const_slots


def _const_slots(
    graph: Graph,
    offsets: Mapping[str, int],
    scratch_base: int,
    weight_region_bytes: int,
    last_step: int,
) -> tuple[Slot, ...]:
    """`Slot`s for the const region, so it lands in `plan.vtcm` like any other.

    The weight double-buffer is ONE reserved region shared by many different
    weight tensors over the plan's life, never all at once -- that sharing is
    what `Tiling.buffers` means -- so it gets a single synthetic slot rather
    than one real slot per weight (which would make unrelated weights look
    like they overlap in VTCM when what actually happens is they take turns).
    Every other const gets its own slot at its own offset, resident for the
    plan's whole lifetime once loaded.
    """
    weight_names = {n for n in (_weight_name(graph, op) for op in graph.ops) if n is not None}
    slots = [
        Slot(
            tensor=_WEIGHT_SCRATCH_NAME,
            offset=scratch_base,
            size=weight_region_bytes,
            first_use=0,
            last_use=last_step,
        )
    ]
    for name, offset in offsets.items():
        if name in weight_names:
            continue
        t = graph.tensor(name)
        slots.append(
            Slot(
                tensor=name,
                offset=offset,
                size=_align(t.nbytes),
                first_use=0,
                last_use=last_step,
            )
        )
    return tuple(slots)


def _align(value: int, to: int = 128) -> int:
    return ((value + to - 1) // to) * to


def _layout_for(t: Tensor) -> Layout:
    """The layout a transfer of this tensor is claimed to land in.

    A const q4_0 weight is repacked on the way in -- that is what makes the
    Q4_0_REPACKED layout true here and what ACCEPTED_LAYOUTS demands. Every
    other tensor lands dense.
    """
    if t.const and t.dtype == "q4_0":
        return Layout.Q4_0_REPACKED
    return Layout.DENSE


def _weight_name(graph: Graph, op: Op) -> str | None:
    """The 2D+ const input this op streams in chunks, if it has one."""
    if op.kind not in TILED_KINDS:
        return None
    for name in op.inputs:
        t = graph.tensor(name)
        if t.const and len(t.shape) >= 2:
            return name
    return None


def _plan_offsets(graph: Graph, scratch_base: int) -> tuple[dict[str, int], int]:
    """VTCM offset for every const tensor this pass will transfer.

    Every tiled weight shares `scratch_base`: that region is a rotating
    double buffer, reused chunk after chunk and op after op, which is what
    `Tiling.buffers=2` means. Every other const is resident for the whole
    plan once transferred, so each is bump-allocated its own offset past the
    weight region, in first-use order -- matching the order `insert_transfers`
    itself schedules the one-shot transfers in.

    Returns the offset map and the bump pointer's final position (i.e. one
    past the last const's region), so the caller can check it against budget.
    """
    const_region_start = _align(scratch_base + WEIGHT_CHUNK_BYTES * WEIGHT_BUFFERS)
    offsets: dict[str, int] = {}
    offset = const_region_start
    for op in graph.ops:
        weight_name = _weight_name(graph, op)
        if weight_name is not None:
            offsets.setdefault(weight_name, scratch_base)
        for name in op.inputs:
            if name == weight_name or name in offsets:
                continue
            t = graph.tensor(name)
            if not t.const:
                continue
            offsets[name] = offset
            offset = _align(offset + t.nbytes)
    return offsets, offset


def _placements_for(
    graph: Graph, op: Op, offsets: Mapping[str, int]
) -> tuple[Placement, ...]:
    """What each input's placement will be once this pass has run.

    A const q4_0 weight is repacked on the way in -- that is what makes the
    Q4_0_REPACKED layout true here and what ACCEPTED_LAYOUTS demands.

    Raises `KeyError` naming the tensor if a const input has no planned
    offset -- `_plan_offsets` scans the same ops and inputs this function
    does, so that should never happen, but silently aliasing the miss onto
    offset 0 (the first activation slot) would be worse than raising: the
    caller converts this into a named `Err` rather than ever using offset 0
    as a fallback.
    """
    out: list[Placement] = []
    for name in op.inputs:
        t = graph.tensor(name)
        layout = _layout_for(t)
        out.append(
            Placement(
                layout=layout,
                perm=tuple(range(len(t.shape))),
                buffer=Buffer.VTCM,
                offset=offsets[name] if t.const else 0,
            )
        )
    return tuple(out)


def plan_problems(
    steps: Sequence[Step],
    graph: Graph,
    slots: Sequence[Slot],
    budget: int | None = None,
) -> list[str]:
    """Spec invariants 3 and 5, which need the op list and the DMA schedule.

    3. Every read of a tensor is preceded by a write or a completed DMA-in.
    5. Every op's working_set is satisfied at the point it runs.

    Also checks:
    - no two transfers active in the same step target overlapping VTCM byte
      ranges -- the DMA-side analogue of the overlap check
      `allocation_problems` already does for VTCM slots. A `Slot` has a
      lifetime `allocation_problems` can compare; a `Transfer` does not, so
      this is scoped to "within one step" rather than across the whole plan
      -- still enough to catch two consts landing at the same address
      because neither was given its own offset.
    - every graph input has a `direction="in"` transfer SOMEWHERE in the
      plan, and every graph output a `direction="out"` transfer -- an
      executor replaying the plan has no other instruction that would fetch
      the input or write the output back, so a plan missing either cannot
      produce a result (Critical 1).
    """
    from hexlib.graph.ops import REGISTRY

    problems: list[str] = []
    available: set[str] = set()
    seen_transfers: set[str] = set()
    all_in_transfers: set[str] = set()
    all_out_transfers: set[str] = set()

    for i, step in enumerate(steps):
        tiling_transfer = step.tiling.transfer if step.tiling else None
        tiling_tuple = (tiling_transfer,) if tiling_transfer is not None else ()

        concurrent = step.dma_in + step.dma_out + tiling_tuple
        for a_idx, a in enumerate(concurrent):
            for b in concurrent[a_idx + 1 :]:
                if a.vtcm_offset < b.vtcm_offset + b.nbytes and b.vtcm_offset < (
                    a.vtcm_offset + a.nbytes
                ):
                    problems.append(
                        f"step {i}: transfers {a.id} ({a.tensor!r}) and {b.id} "
                        f"({b.tensor!r}) both target overlapping VTCM ranges "
                        f"[{a.vtcm_offset}, {a.vtcm_offset + a.nbytes}) and "
                        f"[{b.vtcm_offset}, {b.vtcm_offset + b.nbytes})"
                    )

        for t in step.dma_in:
            all_in_transfers.add(t.tensor)
        for t in step.dma_out:
            all_out_transfers.add(t.tensor)
        if tiling_transfer is not None:
            all_in_transfers.add(tiling_transfer.tensor)

        awaited = set(step.dma_wait)
        for transfer in step.dma_in + tiling_tuple:
            if transfer.id in awaited:
                seen_transfers.add(transfer.tensor)
            else:
                problems.append(
                    f"step {i}: transfer {transfer.id} for {transfer.tensor!r} is "
                    "issued but never awaited"
                )

        if step.op is None:
            continue

        for name in step.op.inputs:
            t = graph.tensor(name)
            if name in available or name in seen_transfers:
                continue
            if t.const or name in graph.inputs:
                problems.append(
                    f"step {i}: op {step.op.id} ({step.op.kind}) reads const or "
                    f"input {name!r} with no DMA-in scheduled for it"
                )
            else:
                problems.append(
                    f"step {i}: op {step.op.id} ({step.op.kind}) reads {name!r} "
                    "before it is written"
                )
        available.update(step.op.outputs)

        if budget is not None:
            try:
                opdef = REGISTRY.get(step.op.kind)
            except KeyError as e:
                problems.append(f"step {i}: {e.args[0]}")
                continue
            inputs = tuple(graph.tensor(n) for n in step.op.inputs)
            outputs = tuple(graph.tensor(n) for n in step.op.outputs)
            try:
                need = opdef.working_set(inputs, outputs, step.op.attrs)
            except (IndexError, TypeError, ValueError, AttributeError) as e:
                problems.append(
                    f"step {i}: op {step.op.id} ({step.op.kind}) working_set "
                    f"raised on malformed op data: {e}"
                )
                continue
            if need > budget:
                problems.append(
                    f"step {i}: op {step.op.id} ({step.op.kind}) needs {need} bytes "
                    f"resident but the VTCM budget is {budget}"
                )

    for name in graph.inputs:
        if name not in all_in_transfers:
            problems.append(
                f"graph input {name!r} has no direction='in' transfer anywhere in "
                "the plan; an executor has no instruction that would fetch it"
            )
    for name in graph.outputs:
        if name not in all_out_transfers:
            problems.append(
                f"graph output {name!r} has no direction='out' transfer anywhere "
                "in the plan; an executor has no instruction that would write it "
                "back to DDR"
            )

    return problems

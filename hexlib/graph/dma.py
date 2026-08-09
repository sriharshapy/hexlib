"""Pass: tile the weight-consuming ops and place their DMA.

Tiling is the mechanism from day one, at every resolution -- not a
large-resolution special case that 256x256 lets us defer. The whole encoder's
weights are about 55.5 MB at q4_0 against an 8 MB VTCM, and the reference never
makes a layer resident at any size: it streams 576-byte chunks
(matmul-ops.c:2653-2655), dequantizes per chunk in-kernel, and double-buffers
against compute (matmul-ops.c:2744-2750).

So VTCM is a working buffer, not a layer cache. The resolution choice affects
how much pressure there is, not whether the mechanism is needed.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from hexlib.graph.ir import Graph, Op
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


def matmul_working_set(inputs, outputs, attrs) -> int:
    """Resident bytes for a tiled matmul: activation, output, and two chunks.

    Replaces M0's "all inputs plus all outputs", which for a
    [256, 3072] x [3072, 3072] matmul claimed about 11 MB resident when the
    real figure is the activation, the output, and 1152 bytes of weight.
    """
    activation = inputs[0].nbytes
    out = sum(t.nbytes for t in outputs)
    extra = sum(t.nbytes for t in inputs[2:])  # a fused bias, if present
    return activation + out + extra + WEIGHT_CHUNK_BYTES * WEIGHT_BUFFERS


def insert_transfers(
    graph: Graph,
    slots: Sequence[Slot],
    budget: int,
) -> tuple[tuple[Step, ...], int] | Err:
    """Turn an ordered graph plus an allocation into steps with DMA placed."""
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

    steps: list[Step] = []
    moved = 0
    next_id = 0
    # Tracks which non-weight consts already got their one-shot transfer, so
    # a tensor read by many ops (e.g. the RoPE cos/sin tables, read by every
    # block) is moved once, not once per reader.
    transferred: set[str] = set()

    for op in graph.ops:
        problems = check_layouts(op.kind, _placements_for(graph, op, offsets))
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
            )
            next_id += 1
            # tile_elements is along the weight's output (n) axis: how many
            # columns one chunk covers, at the weight's own bytes-per-element.
            per_element = total / max(1, weight.shape[-1] * weight.shape[-2])
            tile_elements = max(
                1, int(per_chunk / max(per_element * weight.shape[-2], 1e-9))
            )
            tiling = Tiling(
                axis="n", tile_elements=tile_elements, count=count, buffers=WEIGHT_BUFFERS
            )
            moved += per_chunk * count

        extra_transfers: list[Transfer] = []
        for name in op.inputs:
            if name == weight_name:
                continue
            t = graph.tensor(name)
            if not t.const or name in transferred:
                continue
            extra_transfers.append(
                Transfer(
                    id=next_id,
                    tensor=name,
                    direction="in",
                    vtcm_offset=offsets[name],
                    nbytes=t.nbytes,
                )
            )
            next_id += 1
            moved += t.nbytes
            transferred.add(name)

        dma_in = ((tiled_transfer,) if tiled_transfer is not None else ()) + tuple(
            extra_transfers
        )
        steps.append(
            Step(
                op=op,
                dma_in=dma_in,
                dma_wait=tuple(t.id for t in dma_in),
                dma_out=(),
                tiling=tiling,
            )
        )

    if not steps:
        return Err(
            "nothing to schedule",
            "the graph produced no steps; a schedule of nothing is not a schedule",
        )
    return tuple(steps), moved


def _align(value: int, to: int = 128) -> int:
    return ((value + to - 1) // to) * to


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
    """
    out: list[Placement] = []
    for name in op.inputs:
        t = graph.tensor(name)
        if t.const and t.dtype == "q4_0":
            layout = Layout.Q4_0_REPACKED
        else:
            layout = Layout.DENSE
        out.append(
            Placement(
                layout=layout,
                perm=tuple(range(len(t.shape))),
                buffer=Buffer.VTCM,
                offset=offsets.get(name, 0) if t.const else 0,
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

    Also checks that no two transfers active in the same step target
    overlapping VTCM byte ranges -- the DMA-side analogue of the overlap
    check `allocation_problems` already does for VTCM slots. A `Slot` has a
    lifetime `allocation_problems` can compare; a `Transfer` does not, so
    this is scoped to "within one step" rather than across the whole plan --
    still enough to catch two consts landing at the same address because
    neither was given its own offset.
    """
    from hexlib.graph.ops import REGISTRY

    problems: list[str] = []
    available = set(graph.inputs)
    seen_transfers: set[str] = set()

    for i, step in enumerate(steps):
        concurrent = step.dma_in + step.dma_out
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

        awaited = set(step.dma_wait)
        for transfer in step.dma_in:
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
            if t.const:
                problems.append(
                    f"step {i}: op {step.op.id} ({step.op.kind}) reads const "
                    f"{name!r} with no DMA-in scheduled for it"
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
            need = opdef.working_set(inputs, outputs, step.op.attrs)
            if need > budget:
                problems.append(
                    f"step {i}: op {step.op.id} ({step.op.kind}) needs {need} bytes "
                    f"resident but the VTCM budget is {budget}"
                )

    return problems

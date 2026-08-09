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

from typing import Sequence

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
    if scratch_base + WEIGHT_CHUNK_BYTES * WEIGHT_BUFFERS > budget:
        return Err(
            "no VTCM left for weight streaming",
            f"activations reach {high_water} bytes and the budget is {budget}; "
            f"there is no room for {WEIGHT_BUFFERS} chunks of "
            f"{WEIGHT_CHUNK_BYTES} bytes",
        )

    steps: list[Step] = []
    moved = 0
    next_id = 0
    # Every const is DDR-resident and reaches its first reader by DMA exactly
    # once -- after that it stays put, so later ops that read the same const
    # need no further transfer. Only a matmul/matmul_epilogue's 2D+ weight is
    # big enough to need the chunked, double-buffered treatment below; a
    # bias, a LayerNorm affine param, the position embedding and the RoPE
    # tables are all small enough to move in one shot.
    transferred: set[str] = set()

    for op in graph.ops:
        weights = [
            graph.tensor(name)
            for name in op.inputs
            if graph.tensor(name).const and len(graph.tensor(name).shape) >= 2
        ]
        tiled_transfer: Transfer | None = None
        tiling: Tiling | None = None

        if op.kind in TILED_KINDS and weights:
            weight = weights[0]
            problems = check_layouts(op.kind, _placements_for(graph, op, scratch_base))
            if problems:
                return Err("layout mismatch at plan time", "\n".join(problems))

            total = weight.nbytes
            count = max(1, -(-total // WEIGHT_CHUNK_BYTES))
            per_chunk = min(WEIGHT_CHUNK_BYTES, total)
            tiled_transfer = Transfer(
                id=next_id,
                tensor=weight.name,
                direction="in",
                vtcm_offset=scratch_base,
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
            transferred.add(weight.name)

        extra_transfers: list[Transfer] = []
        for name in op.inputs:
            t = graph.tensor(name)
            if not t.const or name in transferred:
                continue
            extra_transfers.append(
                Transfer(
                    id=next_id,
                    tensor=name,
                    direction="in",
                    vtcm_offset=scratch_base,
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


def _placements_for(graph: Graph, op: Op, scratch_base: int) -> tuple[Placement, ...]:
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
                offset=scratch_base if t.const else 0,
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
    """
    from hexlib.graph.ops import REGISTRY

    problems: list[str] = []
    available = set(graph.inputs)
    seen_transfers: set[str] = set()

    for i, step in enumerate(steps):
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

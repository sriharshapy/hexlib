from __future__ import annotations

import hexlib.graph.opdefs  # noqa: F401
from hexlib.graph.dma import (
    WEIGHT_BUFFERS,
    WEIGHT_CHUNK_BYTES,
    insert_transfers,
    matmul_working_set,
    plan_problems,
)
from hexlib.graph.ir import Graph, Op, Tensor
from hexlib.graph.layout import Layout
from hexlib.graph.liveness import live_intervals
from hexlib.graph.order import order
from hexlib.graph.ops import get
from hexlib.graph.vtcm import allocate
from hexlib.models.qwen35 import qwen35_at
from hexlib.models.vit import build_vision_encoder
from hexlib.result import Err

BUDGET = 8388608


def _matmul_graph(k=3072, n=3072, m=256):
    tensors = {
        "x": Tensor("x", "fp16", (m, k)),
        "w": Tensor("w", "q4_0", (k, n), const=True),
        "y": Tensor("y", "fp16", (m, n)),
    }
    return Graph(
        tensors=tensors,
        ops=(Op(id=0, kind="matmul", inputs=("x", "w"), outputs=("y",), attrs={}),),
        inputs=("x",),
        outputs=("y",),
    )


def _matmul_epilogue_graph():
    """A tiled q4_0 weight AND a one-shot fp32 bias, on the same op.

    Used to check that the recorded `Transfer.layout` differs between the
    two: `Q4_0_REPACKED` for the weight, `DENSE` for the bias.
    """
    tensors = {
        "x": Tensor("x", "fp16", (4, 8)),
        "w": Tensor("w", "q4_0", (8, 32), const=True),
        "b": Tensor("b", "fp32", (32,), const=True),
        "y": Tensor("y", "fp16", (4, 32)),
    }
    return Graph(
        tensors=tensors,
        ops=(
            Op(
                id=0,
                kind="matmul_epilogue",
                inputs=("x", "w", "b"),
                outputs=("y",),
                attrs={"act": "none"},
            ),
        ),
        inputs=("x",),
        outputs=("y",),
    )


def _steps_for(graph, budget=BUDGET):
    g = order(graph)
    ivs = live_intervals(g)
    slots, _ = allocate(ivs, budget=budget)
    return g, slots, insert_transfers(g, slots, budget)


def test_the_weight_chunk_is_576_bytes():
    # matmul-ops.c:2653-2655. Not a round number and not ours to change
    # without measuring.
    assert WEIGHT_CHUNK_BYTES == 576


def test_a_quantized_weight_is_streamed_not_made_resident():
    g, slots, result = _steps_for(_matmul_graph())
    assert not isinstance(result, Err), getattr(result, "detail", "")
    steps, moved, const_slots = result
    (step,) = [s for s in steps if s.op is not None]
    assert step.tiling is not None
    assert step.tiling.count > 1
    # The repeating chunk lives on `Tiling.transfer`, not `Step.dma_in` --
    # `dma_in` now holds only one-shot transfers (here: the graph's own
    # input "x", moved once).
    per_iteration = step.tiling.transfer.nbytes
    assert per_iteration <= WEIGHT_CHUNK_BYTES * 2
    assert moved >= g.tensor("w").nbytes


def test_weight_transfers_are_double_buffered():
    _, _, (steps, _, _) = _steps_for(_matmul_graph())
    (step,) = [s for s in steps if s.op is not None]
    assert step.tiling.buffers == 2, "no double buffering means no DMA/compute overlap"


def test_every_transfer_is_awaited_before_the_op_runs():
    _, _, (steps, _, _) = _steps_for(_matmul_graph())
    for step in steps:
        issued = {t.id for t in step.dma_in}
        if step.tiling:
            issued.add(step.tiling.transfer.id)
        assert issued <= set(step.dma_wait), "a transfer is issued but never awaited"


def test_transfer_ids_are_unique_across_the_whole_plan():
    _, _, (steps, _, _) = _steps_for(build_vision_encoder(qwen35_at(256)))
    ids = [
        t.id
        for s in steps
        for t in (*s.dma_in, *s.dma_out, *((s.tiling.transfer,) if s.tiling else ()))
    ]
    assert len(ids) == len(set(ids))


def test_matmul_working_set_is_far_below_the_whole_weight():
    x = Tensor("x", "fp16", (256, 3072))
    w = Tensor("w", "q4_0", (3072, 3072), const=True)
    y = Tensor("y", "fp16", (256, 3072))
    ws = matmul_working_set((x, w), (y,), {})
    # At these shapes activation+output already exceed a quarter of the
    # weight, so "below w.nbytes // 4" is unsatisfiable by any
    # implementation (3,146,880 >= 1,327,104). Pin the actual property
    # instead: the weight contributes exactly two chunks, nothing more --
    # a regression that folds in even a third of the real weight
    # (1.77 MB) would still slip past a loose "< w.nbytes" bound.
    assert ws - (x.nbytes + y.nbytes) == WEIGHT_CHUNK_BYTES * WEIGHT_BUFFERS


def test_matmul_working_set_on_empty_inputs_raises_a_clear_value_error():
    # Previously this was `inputs[0].nbytes` on an empty tuple -- an IndexError
    # with no context. plan_problems calls working_set with whatever a
    # (possibly malformed) op declares, and wraps this in a try/except of its
    # own, but the function itself should still fail with a message that
    # names the actual problem rather than a bare IndexError.
    import pytest

    with pytest.raises(ValueError):
        matmul_working_set((), (Tensor("y", "fp16", (4,)),), {})


def test_the_registered_matmul_working_set_uses_the_tiled_estimate():
    x = Tensor("x", "fp16", (256, 3072))
    w = Tensor("w", "q4_0", (3072, 3072), const=True)
    y = Tensor("y", "fp16", (256, 3072))
    assert get("matmul").working_set((x, w), (y,), {}) == matmul_working_set(
        (x, w), (y,), {}
    )


def test_a_dense_fp16_weight_is_also_tiled():
    g, slots, (steps, moved, const_slots) = _steps_for(
        Graph(
            tensors={
                "x": Tensor("x", "fp16", (256, 768)),
                "w": Tensor("w", "fp16", (768, 768), const=True),
                "y": Tensor("y", "fp16", (256, 768)),
            },
            ops=(Op(id=0, kind="matmul", inputs=("x", "w"), outputs=("y",), attrs={}),),
            inputs=("x",),
            outputs=("y",),
        )
    )
    (step,) = [s for s in steps if s.op is not None]
    assert step.tiling is not None


def test_an_elementwise_op_gets_no_tiling_and_no_weight_transfer():
    g = Graph(
        tensors={
            "x": Tensor("x", "fp16", (256, 768)),
            "y": Tensor("y", "fp16", (256, 768)),
        },
        ops=(Op(id=0, kind="gelu_tanh", inputs=("x",), outputs=("y",), attrs={}),),
        inputs=("x",),
        outputs=("y",),
    )
    _, _, (steps, moved, const_slots) = _steps_for(g)
    (step,) = [s for s in steps if s.op is not None]
    assert step.tiling is None
    # "x" is the graph's own input, so it still gets exactly one DMA-in
    # transfer (Critical 1) -- there is simply no weight to tile.
    assert [t.tensor for t in step.dma_in] == ["x"]


def test_predicted_bytes_moved_is_at_least_the_total_weight_bytes():
    g = build_vision_encoder(qwen35_at(256))
    _, _, (steps, moved, const_slots) = _steps_for(g)
    weight_bytes = sum(t.nbytes for t in g.tensors.values() if t.const)
    assert moved >= weight_bytes
    # 55.5 MB at q4_0 for the whole encoder (spec 5.2). Sanity-check the order.
    assert 40_000_000 < weight_bytes < 70_000_000


def test_predicted_bytes_moved_also_counts_the_graph_input_and_output():
    # Critical 1: the image coming in and the encoder's own output going
    # back out are DDR<->VTCM traffic too -- an executor has no other
    # instruction that would move either one.
    g = build_vision_encoder(qwen35_at(256))
    _, _, (steps, moved, const_slots) = _steps_for(g)
    image_bytes = g.tensor(g.inputs[0]).nbytes
    output_bytes = g.tensor(g.outputs[0]).nbytes
    weight_and_const_bytes = sum(t.nbytes for t in g.tensors.values() if t.const)
    assert moved >= weight_and_const_bytes + image_bytes + output_bytes


def test_a_layout_mismatch_is_caught_at_plan_time():
    # The structural mitigation for the spec's highest-severity silent
    # corruption risk: an un-repacked q4_0 weight must not reach a matmul.
    from hexlib.graph.layout import Buffer, Placement, check_layouts

    bad = (
        Placement(Layout.DENSE, (0, 1), Buffer.VTCM, 0),
        Placement(Layout.Q4_0_BLOCKED, (0, 1), Buffer.VTCM, 0),
    )
    assert check_layouts("matmul", bad) != []


def test_a_transfer_is_scheduled_for_the_tiled_weight():
    # Renamed from test_the_dma_pass_repacks_quantized_weights_before_use:
    # that name claimed to test repacking, but all it checked was that SOME
    # transfer existed for "w" -- it never looked at the layout the transfer
    # carried. See test_the_recorded_layout_distinguishes_the_weight_from_the_bias
    # below for the actual repack check.
    _, _, (steps, _, _) = _steps_for(_matmul_graph())
    (step,) = [s for s in steps if s.op is not None]
    assert step.tiling is not None
    assert step.tiling.transfer.tensor == "w"


def test_the_recorded_layout_distinguishes_the_weight_from_the_bias():
    # The structural repack check (Important 5): `check_layouts` accepting a
    # `Q4_0_REPACKED` placement proves nothing on its own if that layout is
    # never written down anywhere the rest of the plan can see. This checks
    # the claim actually lands on the `Transfer` -- Q4_0_REPACKED for the
    # quantized weight, DENSE for the fp32 bias riding along on the same op.
    _, _, (steps, _, _) = _steps_for(_matmul_epilogue_graph())
    (step,) = [s for s in steps if s.op is not None]
    assert step.tiling.transfer.layout is Layout.Q4_0_REPACKED
    (bias_transfer,) = [t for t in step.dma_in if t.tensor == "b"]
    assert bias_transfer.layout is Layout.DENSE


def test_plan_problems_catches_a_read_with_no_preceding_write():
    # Invariant 3 from the spec.
    g, slots, (steps, _, _) = _steps_for(_matmul_graph())
    broken = tuple(
        type(s)(op=s.op, dma_in=(), dma_wait=(), dma_out=s.dma_out, tiling=None)
        for s in steps
    )
    problems = plan_problems(broken, g, slots)
    # Stripping dma_in and tiling removes the DMA for BOTH "x" (the graph
    # input) and "w" (the const weight); either is a valid catch of this
    # invariant, so check the union rather than pinning problems[0] to one.
    assert problems
    assert any("w" in p for p in problems)


def test_plan_problems_catches_a_working_set_over_budget():
    # Invariant 5 from the spec.
    g, slots, (steps, _, _) = _steps_for(_matmul_graph())
    assert plan_problems(steps, g, slots, budget=1024) != []


def test_plan_problems_is_empty_on_a_well_formed_plan():
    g, slots, (steps, _, _) = _steps_for(build_vision_encoder(qwen35_at(256)))
    # Invariant 5 needs a real budget to actually run; a plan certified
    # "well-formed" with no budget passed would never exercise it.
    assert plan_problems(steps, g, slots, budget=BUDGET) == []


def test_plan_problems_catches_overlapping_transfers():
    # Invariant 1's DMA-side analogue: two transfers resident at once must
    # not share a VTCM address, or the second silently overwrites the
    # first before its op reads it. The weight's repeating transfer now
    # lives on `Tiling.transfer`, not `dma_in`.
    g, slots, (steps, _, _) = _steps_for(_matmul_graph())
    (step,) = [s for s in steps if s.op is not None]
    weight_transfer = step.tiling.transfer
    collider = type(weight_transfer)(
        id=weight_transfer.id + 1000,
        tensor="collider",
        direction="in",
        vtcm_offset=weight_transfer.vtcm_offset,
        nbytes=weight_transfer.nbytes,
        layout=weight_transfer.layout,
    )
    broken = tuple(
        type(s)(
            op=s.op,
            dma_in=s.dma_in + (collider,),
            dma_wait=s.dma_wait + (collider.id,),
            dma_out=s.dma_out,
            tiling=s.tiling,
        )
        if s is step
        else s
        for s in steps
    )
    problems = plan_problems(broken, g, slots)
    assert problems and "overlapping" in problems[0]


def test_plan_problems_catches_a_missing_input_transfer():
    # Critical 1's new invariant: every graph input needs a direction="in"
    # transfer somewhere in the plan, or an executor has no instruction that
    # would ever fetch it.
    g, slots, (steps, _, _) = _steps_for(_matmul_graph())
    broken = tuple(
        type(s)(
            op=s.op,
            dma_in=tuple(t for t in s.dma_in if t.tensor != "x"),
            dma_wait=s.dma_wait,
            dma_out=s.dma_out,
            tiling=s.tiling,
        )
        for s in steps
    )
    problems = plan_problems(broken, g, slots)
    assert any("graph input" in p and "'x'" in p for p in problems)


def test_plan_problems_catches_a_missing_output_transfer():
    # The output-side half of the same invariant: every graph output needs a
    # direction="out" transfer, or an executor has no instruction that would
    # ever write it back to DDR.
    g, slots, (steps, _, _) = _steps_for(_matmul_graph())
    broken = tuple(
        type(s)(op=s.op, dma_in=s.dma_in, dma_wait=s.dma_wait, dma_out=(), tiling=s.tiling)
        for s in steps
    )
    problems = plan_problems(broken, g, slots)
    assert any("graph output" in p and "'y'" in p for p in problems)


def test_a_budget_too_small_for_two_chunks_is_an_err():
    g = order(_matmul_graph())
    ivs = live_intervals(g)
    slots, _ = allocate(ivs, budget=BUDGET)
    out = insert_transfers(g, slots, budget=1024)
    assert isinstance(out, Err)


def test_a_too_small_budget_for_resident_consts_is_an_err():
    # Distinct from the "too small to double-buffer" branch above: here the
    # weight scratch region fits exactly, but there is no room left for the
    # plan's OTHER resident const (the bias) past it. This branch previously
    # had no test at all.
    g = order(_matmul_epilogue_graph())
    ivs = live_intervals(g)
    slots, _ = allocate(ivs, budget=BUDGET)
    high_water = max(s.end for s in slots)
    scratch_base = ((high_water + 127) // 128) * 128
    tight_budget = scratch_base + WEIGHT_CHUNK_BYTES * WEIGHT_BUFFERS
    out = insert_transfers(g, slots, budget=tight_budget)
    assert isinstance(out, Err)
    assert str(tight_budget) in out.detail

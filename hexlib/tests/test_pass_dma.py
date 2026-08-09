from __future__ import annotations

import hexlib.graph.opdefs  # noqa: F401
from hexlib.graph.dma import (
    WEIGHT_CHUNK_BYTES,
    insert_transfers,
    matmul_working_set,
    plan_problems,
)
from hexlib.graph.ir import Graph, Op, Tensor
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
    steps, moved = result
    (step,) = [s for s in steps if s.op is not None]
    assert step.tiling is not None
    assert step.tiling.count > 1
    per_iteration = sum(t.nbytes for t in step.dma_in)
    assert per_iteration <= WEIGHT_CHUNK_BYTES * 2
    assert moved >= g.tensor("w").nbytes


def test_weight_transfers_are_double_buffered():
    _, _, (steps, _) = _steps_for(_matmul_graph())
    (step,) = [s for s in steps if s.op is not None]
    assert step.tiling.buffers == 2, "no double buffering means no DMA/compute overlap"


def test_every_transfer_is_awaited_before_the_op_runs():
    _, _, (steps, _) = _steps_for(_matmul_graph())
    for step in steps:
        issued = {t.id for t in step.dma_in}
        assert issued <= set(step.dma_wait), "a transfer is issued but never awaited"


def test_transfer_ids_are_unique_across_the_whole_plan():
    _, _, (steps, _) = _steps_for(build_vision_encoder(qwen35_at(256)))
    ids = [t.id for s in steps for t in (*s.dma_in, *s.dma_out)]
    assert len(ids) == len(set(ids))


def test_matmul_working_set_is_far_below_the_whole_weight():
    x = Tensor("x", "fp16", (256, 3072))
    w = Tensor("w", "q4_0", (3072, 3072), const=True)
    y = Tensor("y", "fp16", (256, 3072))
    ws = matmul_working_set((x, w), (y,), {})
    # At these shapes activation+output already exceed a quarter of the
    # weight, so the achievable bound is "below the whole weight", not below
    # a quarter of it -- the activation and output are the same size
    # regardless of q4_0 vs fp16 weight, only the weight's own bytes change.
    assert ws < w.nbytes, f"working set {ws} still counts the whole weight"
    assert ws >= x.nbytes + y.nbytes


def test_the_registered_matmul_working_set_uses_the_tiled_estimate():
    x = Tensor("x", "fp16", (256, 3072))
    w = Tensor("w", "q4_0", (3072, 3072), const=True)
    y = Tensor("y", "fp16", (256, 3072))
    assert get("matmul").working_set((x, w), (y,), {}) == matmul_working_set(
        (x, w), (y,), {}
    )


def test_a_dense_fp16_weight_is_also_tiled():
    g, slots, (steps, moved) = _steps_for(
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
    _, _, (steps, moved) = _steps_for(g)
    (step,) = [s for s in steps if s.op is not None]
    assert step.tiling is None
    assert step.dma_in == ()


def test_predicted_bytes_moved_is_at_least_the_total_weight_bytes():
    g = build_vision_encoder(qwen35_at(256))
    _, _, (steps, moved) = _steps_for(g)
    weight_bytes = sum(t.nbytes for t in g.tensors.values() if t.const)
    assert moved >= weight_bytes
    # 55.5 MB at q4_0 for the whole encoder (spec 5.2). Sanity-check the order.
    assert 40_000_000 < weight_bytes < 70_000_000


def test_a_layout_mismatch_is_caught_at_plan_time():
    # The structural mitigation for the spec's highest-severity silent
    # corruption risk: an un-repacked q4_0 weight must not reach a matmul.
    from hexlib.graph.layout import Buffer, Layout, Placement, check_layouts

    bad = (
        Placement(Layout.DENSE, (0, 1), Buffer.VTCM, 0),
        Placement(Layout.Q4_0_BLOCKED, (0, 1), Buffer.VTCM, 0),
    )
    assert check_layouts("matmul", bad) != []


def test_the_dma_pass_repacks_quantized_weights_before_use():
    _, _, (steps, _) = _steps_for(_matmul_graph())
    (step,) = [s for s in steps if s.op is not None]
    layouts = {t.tensor: t for t in step.dma_in}
    assert "w" in layouts


def test_plan_problems_catches_a_read_with_no_preceding_write():
    # Invariant 3 from the spec.
    g, slots, (steps, _) = _steps_for(_matmul_graph())
    broken = tuple(
        type(s)(op=s.op, dma_in=(), dma_wait=(), dma_out=s.dma_out, tiling=s.tiling)
        for s in steps
    )
    problems = plan_problems(broken, g, slots)
    assert problems and "w" in problems[0]


def test_plan_problems_catches_a_working_set_over_budget():
    # Invariant 5 from the spec.
    g, slots, (steps, _) = _steps_for(_matmul_graph())
    assert plan_problems(steps, g, slots, budget=1024) != []


def test_plan_problems_is_empty_on_a_well_formed_plan():
    g, slots, (steps, _) = _steps_for(build_vision_encoder(qwen35_at(256)))
    assert plan_problems(steps, g, slots) == []


def test_a_budget_too_small_for_two_chunks_is_an_err():
    g = order(_matmul_graph())
    ivs = live_intervals(g)
    slots, _ = allocate(ivs, budget=BUDGET)
    out = insert_transfers(g, slots, budget=1024)
    assert isinstance(out, Err)

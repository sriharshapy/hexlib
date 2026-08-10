from __future__ import annotations

import hexlib.graph.opdefs  # noqa: F401
from hexlib.graph.ir import Graph, Op, Tensor
from hexlib.graph.pipeline import PASSES, compile_graph
from hexlib.graph.plan import Plan, from_json, to_json
from hexlib.models.qwen35 import qwen35_at
from hexlib.models.vit import build_vision_encoder
from hexlib.result import Err

BUDGET = 8388608


def test_the_whole_encoder_compiles_to_a_plan():
    plan = compile_graph(build_vision_encoder(qwen35_at(256)), budget=BUDGET)
    assert isinstance(plan, Plan), getattr(plan, "detail", "")
    assert plan.steps
    assert plan.vtcm_high_water > 0
    assert plan.predicted_bytes_moved > 0
    assert plan.vtcm_budget == BUDGET


def test_the_plan_round_trips_through_json():
    plan = compile_graph(build_vision_encoder(qwen35_at(256)), budget=BUDGET)
    back = from_json(to_json(plan))
    assert not isinstance(back, Err)
    assert back == plan


def test_unimplemented_kinds_are_reported_not_silently_dropped():
    # No kernels exist yet, so every kind should be listed.
    plan = compile_graph(build_vision_encoder(qwen35_at(256)), budget=BUDGET)
    assert plan.unimplemented, "every op kind has kernel=None; the list cannot be empty"
    assert "matmul_epilogue" in plan.unimplemented or "matmul" in plan.unimplemented


def test_a_tiny_budget_is_an_err_naming_the_budget():
    out = compile_graph(build_vision_encoder(qwen35_at(256)), budget=4096)
    assert isinstance(out, Err)
    assert "4096" in out.detail


def test_a_zero_budget_is_an_err_and_never_a_default():
    # The budget is a parameter, never a constant: a caller passing 0 must not
    # silently get 8 MB.
    assert isinstance(compile_graph(build_vision_encoder(qwen35_at(256)), budget=0), Err)


def test_an_invalid_graph_is_an_err_naming_the_first_failing_pass():
    g = Graph(
        tensors={"x": Tensor("x", "fp32", (2,)), "y": Tensor("y", "fp32", (99,))},
        ops=(Op(id=0, kind="scale", inputs=("x",), outputs=("y",), attrs={"factor": 1.0}),),
        inputs=("x",),
        outputs=("y",),
    )
    out = compile_graph(g, budget=BUDGET)
    assert isinstance(out, Err)
    assert "shape" in out.reason.lower() or "shape" in out.detail.lower()


def test_passes_tuple_names_the_six_stages():
    # Just the tuple's contents. This alone proves nothing about whether
    # compile_graph actually CALLS each one -- see the spy-based test below,
    # which is what used to be claimed (wrongly) under this test's old name,
    # test_every_named_pass_actually_runs.
    assert PASSES == ("shapes", "fuse", "order", "liveness", "vtcm", "dma")


def test_compile_graph_actually_invokes_every_named_pass():
    # A pipeline that skipped a pass would produce a plan that looked fine;
    # nothing else guards against a pass being silently dropped. Spy on each
    # pass's entry point as bound in pipeline.py's own namespace (that is
    # what compile_graph actually calls) and confirm a real compile touches
    # all six exactly once.
    from unittest.mock import Mock, patch

    import hexlib.graph.pipeline as pipeline_mod

    names = ("infer_shapes", "fuse", "order", "live_intervals", "allocate", "insert_transfers")
    with patch.multiple(
        pipeline_mod,
        **{name: Mock(wraps=getattr(pipeline_mod, name)) for name in names},
    ):
        plan = compile_graph(build_vision_encoder(qwen35_at(256)), budget=BUDGET)
        assert isinstance(plan, Plan), getattr(plan, "detail", "")
        for name in names:
            getattr(pipeline_mod, name).assert_called_once()


def test_both_order_policies_and_both_alloc_policies_compile():
    from hexlib.graph.order import ORDER_POLICIES
    from hexlib.graph.vtcm import ALLOC_POLICIES

    g = build_vision_encoder(qwen35_at(256))
    for op in ORDER_POLICIES:
        for ap in ALLOC_POLICIES:
            plan = compile_graph(g, budget=BUDGET, order_policy=op, alloc_policy=ap)
            assert isinstance(plan, Plan), f"{op}/{ap}: {getattr(plan, 'detail', '')}"


def test_the_plan_fits_in_vtcm_at_256():
    # `Plan.__post_init__` already makes vtcm_high_water > vtcm_budget
    # unconstructible, so asserting that inequality again here (as this test
    # used to) is checking something that can never fail once `plan` exists
    # at all. What actually needs proving is that compile_graph SUCCEEDS
    # (returns a Plan, not an Err) at this resolution and budget.
    plan = compile_graph(build_vision_encoder(qwen35_at(256)), budget=BUDGET)
    assert isinstance(plan, Plan), getattr(plan, "detail", "")


def test_a_larger_resolution_moves_more_bytes():
    small = compile_graph(build_vision_encoder(qwen35_at(256)), budget=BUDGET)
    assert isinstance(small, Plan), getattr(small, "detail", "")
    large = compile_graph(build_vision_encoder(qwen35_at(512)), budget=BUDGET)
    if isinstance(large, Err):
        # Legitimate: 512^2 may not fit. It must fail because the
        # ALLOCATION does not fit -- not some unrelated malformed-input Err,
        # which `"budget" in large.detail` was too weak to rule out (nearly
        # every allocator error mentions "budget" somewhere in its detail).
        assert large.reason.startswith(("vtcm:", "dma:")), large.reason
        assert "does not fit" in large.reason or "no VTCM left" in large.reason
    else:
        assert large.predicted_bytes_moved > small.predicted_bytes_moved

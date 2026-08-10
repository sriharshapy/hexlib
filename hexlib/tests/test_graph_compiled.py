"""A Compiled pair is executable and round-trips; a mismatched pair cannot exist.

The defect this type fixes was invisible to every existing test because they all
held the post-fusion graph in a local variable already. The tests here are
written from the position an EXECUTOR is in: holding only what was serialized.
"""
import pytest

import hexlib.graph.opdefs  # noqa: F401  -- registers the op definitions
from hexlib.graph import compiled as C
from hexlib.graph.ir import Graph, Op, Tensor
from hexlib.graph.pipeline import compile_graph, compile_model
from hexlib.graph.plan import V75_VTCM_TOTAL_BYTES, Plan, Step
from hexlib.models.qwen35 import qwen35_at
from hexlib.models.vit import build_vision_encoder
from hexlib.result import Err


@pytest.fixture(scope="module")
def model():
    graph = build_vision_encoder(qwen35_at(256))
    assert not isinstance(graph, Err), graph
    m = compile_model(graph, budget=V75_VTCM_TOTAL_BYTES)
    assert not isinstance(m, Err), m
    return m


def test_every_tensor_a_step_names_is_described(model):
    """The point of the type: an executor can resolve every name it sees.

    A plan alone carries names without shapes or dtypes, so this is the
    property that makes replay possible at all.
    """
    for step in model.plan.steps:
        if step.op is None:
            continue
        for name in tuple(step.op.inputs) + tuple(step.op.outputs):
            t = model.tensor(name)
            assert t.shape, f"{name} has no shape"
            assert t.dtype


def test_carries_the_post_fusion_ops_not_the_source_graph(model):
    """Fusion CREATES ops. They exist in no graph the caller passed in."""
    kinds = {op.kind for op in model.graph.ops}
    assert "matmul_epilogue" in kinds, (
        "the fused op kind is absent, so this is the pre-fusion graph"
    )
    assert "gelu_tanh" not in kinds, "an absorbed activation survived fusion"
    n_epilogue = sum(1 for op in model.graph.ops if op.kind == "matmul_epilogue")
    assert n_epilogue == 75


def test_ops_correspond_one_to_one_with_steps_that_have_an_op(model):
    with_op = [s for s in model.plan.steps if s.op is not None]
    assert len(with_op) == len(model.graph.ops)
    for step, op in zip(with_op, model.graph.ops):
        assert step.op.id == op.id
        assert step.op.kind == op.kind


def test_round_trips_through_json(model):
    rt = C.from_json(C.to_json(model))
    assert not isinstance(rt, Err), rt
    assert len(rt.graph.ops) == len(model.graph.ops)
    assert len(rt.graph.tensors) == len(model.graph.tensors)
    assert len(rt.plan.steps) == len(model.plan.steps)
    assert rt.plan.vtcm_high_water == model.plan.vtcm_high_water
    assert rt.plan.predicted_bytes_moved == model.plan.predicted_bytes_moved
    assert rt.plan.target == model.plan.target


def test_round_trip_preserves_tensor_dtypes_and_shapes(model):
    """The whole payload. If shapes came back wrong an executor would allocate
    wrong and only notice as a numerical failure much later."""
    rt = C.from_json(C.to_json(model))
    for name, t in model.graph.tensors.items():
        got = rt.graph.tensor(name)
        assert got.shape == t.shape, name
        assert got.dtype == t.dtype, name
        assert got.const == t.const, name


def test_round_trip_preserves_tuple_valued_attrs(model):
    """JSON has no tuple type. `perm` and `shape` attrs are tuples, and an
    executor that got lists would pass them to numpy and get a different
    answer or an error."""
    rt = C.from_json(C.to_json(model))
    by_id = {op.id: op for op in rt.graph.ops}
    checked = 0
    for op in model.graph.ops:
        for key, value in op.attrs.items():
            if isinstance(value, tuple):
                assert by_id[op.id].attrs[key] == value, f"op {op.id} attr {key}"
                checked += 1
    assert checked > 0, "no tuple attrs found, so this test proved nothing"


def test_mismatched_pair_is_rejected_at_construction():
    """A plan whose steps name tensors the graph lacks is not executable, and
    saying so at construction beats a KeyError inside an executor."""
    tensor = Tensor(name="a", dtype="fp32", shape=(4,))
    graph = Graph(
        tensors={"a": tensor},
        ops=(Op(id=0, kind="cast", inputs=("a",), outputs=("a",), attrs={}),),
        inputs=("a",),
        outputs=("a",),
    )
    plan = Plan(
        steps=(
            Step(
                op=Op(id=0, kind="cast", inputs=("nope",), outputs=("a",), attrs={}),
                dma_in=(),
                dma_wait=(),
                dma_out=(),
            ),
        ),
        vtcm=(),
        vtcm_high_water=0,
        predicted_bytes_moved=0,
        vtcm_budget=1024,
        unimplemented=(),
    )
    with pytest.raises(ValueError, match="does not declare"):
        C.Compiled(graph=graph, plan=plan)


def test_compile_graph_still_returns_a_bare_plan(model):
    """The existing contract is unchanged; compile_model is additive."""
    graph = build_vision_encoder(qwen35_at(256))
    plan = compile_graph(graph, budget=V75_VTCM_TOTAL_BYTES)
    assert isinstance(plan, Plan)
    assert len(plan.steps) == len(model.plan.steps)


def test_malformed_json_is_an_err_not_a_raise():
    for text in ("{", "[]", '{"graph": {}}', '{"plan": {}, "graph": {}}'):
        assert isinstance(C.from_json(text), Err), text

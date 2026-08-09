from __future__ import annotations

import hexlib.graph.opdefs  # noqa: F401
from hexlib.graph.ir import Graph, Op, Tensor
from hexlib.graph.order import ORDER_POLICIES, order, peak_live_bytes
from hexlib.models.qwen35 import qwen35_at
from hexlib.models.vit import build_vision_encoder
from hexlib.result import Err


def _diamond():
    """x -> a, x -> b, (a, b) -> y. Two orders are legal; one frees sooner."""
    tensors = {
        "x": Tensor("x", "fp32", (16,)),
        "a": Tensor("a", "fp32", (16,)),
        "b": Tensor("b", "fp32", (1024,)),
        "y": Tensor("y", "fp32", (16,)),
        "bs": Tensor("bs", "fp32", (16,)),
    }
    return Graph(
        tensors=tensors,
        ops=(
            Op(id=0, kind="scale", inputs=("x",), outputs=("a",), attrs={"factor": 1.0}),
            Op(id=1, kind="reshape", inputs=("x",), outputs=("b",), attrs={"shape": (1024,)}),
            Op(id=2, kind="reshape", inputs=("b",), outputs=("bs",), attrs={"shape": (16,)}),
            Op(id=3, kind="add", inputs=("a", "bs"), outputs=("y",), attrs={}),
        ),
        inputs=("x",),
        outputs=("y",),
    )


def test_order_produces_a_topological_order():
    g = order(_diamond())
    assert not isinstance(g, Err)
    written = set(g.inputs) | {t.name for t in g.tensors.values() if t.const}
    for op in g.ops:
        for name in op.inputs:
            assert name in written, f"{op.kind} reads {name} before it is written"
        written.update(op.outputs)


def test_order_preserves_the_op_set_exactly():
    g = _diamond()
    out = order(g)
    assert sorted(op.id for op in out.ops) == sorted(op.id for op in g.ops)
    assert len(out.ops) == len(g.ops)


def test_order_does_not_mutate_its_input():
    g = _diamond()
    before = [op.id for op in g.ops]
    order(g)
    assert [op.id for op in g.ops] == before


def test_both_policies_produce_valid_orders():
    for name in ORDER_POLICIES:
        out = order(_diamond(), policy=name)
        assert not isinstance(out, Err), name
        assert out.problems() == [], name


def test_min_peak_is_no_worse_than_asap_on_the_diamond():
    asap = peak_live_bytes(order(_diamond(), policy="asap"))
    min_peak = peak_live_bytes(order(_diamond(), policy="min_peak"))
    assert min_peak <= asap


def test_an_unknown_policy_is_an_err_naming_the_known_ones():
    out = order(_diamond(), policy="wishful")
    assert isinstance(out, Err)
    assert "wishful" in out.detail
    assert "asap" in out.detail


def test_a_cycle_is_an_err_naming_the_stuck_ops():
    tensors = {
        "x": Tensor("x", "fp32", (4,)),
        "a": Tensor("a", "fp32", (4,)),
        "b": Tensor("b", "fp32", (4,)),
    }
    g = Graph(
        tensors=tensors,
        ops=(
            Op(id=0, kind="add", inputs=("x", "b"), outputs=("a",), attrs={}),
            Op(id=1, kind="add", inputs=("x", "a"), outputs=("b",), attrs={}),
        ),
        inputs=("x",),
        outputs=("b",),
    )
    out = order(g, policy="asap")
    assert isinstance(out, Err)
    assert "cycle" in out.reason or "cycle" in out.detail


def test_peak_live_bytes_is_positive_and_at_least_the_largest_tensor():
    g = order(_diamond())
    assert peak_live_bytes(g) >= 1024 * 4


def test_the_whole_encoder_orders_under_both_policies():
    g = build_vision_encoder(qwen35_at(256))
    for name in ORDER_POLICIES:
        out = order(g, policy=name)
        assert not isinstance(out, Err), name
        assert out.problems() == [], name

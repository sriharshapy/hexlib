from __future__ import annotations

import hexlib.graph.opdefs  # noqa: F401
from hexlib.graph.ir import Graph, Op, Tensor
from hexlib.graph.liveness import Interval, live_at, live_intervals
from hexlib.graph.order import order
from hexlib.models.qwen35 import qwen35_at
from hexlib.models.vit import build_vision_encoder
from hexlib.result import Err


def _chain():
    tensors = {
        "x": Tensor("x", "fp32", (16,)),
        "a": Tensor("a", "fp32", (16,)),
        "b": Tensor("b", "fp32", (16,)),
        "y": Tensor("y", "fp32", (16,)),
    }
    return Graph(
        tensors=tensors,
        ops=(
            Op(id=0, kind="scale", inputs=("x",), outputs=("a",), attrs={"factor": 2.0}),
            Op(id=1, kind="scale", inputs=("a",), outputs=("b",), attrs={"factor": 3.0}),
            Op(id=2, kind="scale", inputs=("b",), outputs=("y",), attrs={"factor": 4.0}),
        ),
        inputs=("x",),
        outputs=("y",),
    )


def _by_name(intervals):
    return {iv.tensor: iv for iv in intervals}


def test_an_intermediate_dies_at_its_only_consumer():
    ivs = _by_name(live_intervals(_chain()))
    assert ivs["a"].first_use == 0
    assert ivs["a"].last_use == 1


def test_a_graph_output_stays_live_past_the_last_op():
    g = _chain()
    ivs = _by_name(live_intervals(g))
    assert ivs["y"].last_use == len(g.ops)


def test_a_graph_input_is_live_from_step_zero():
    ivs = _by_name(live_intervals(_chain()))
    assert ivs["x"].first_use == 0


def test_consts_are_excluded():
    tensors = {
        "x": Tensor("x", "fp32", (16,)),
        "w": Tensor("w", "fp32", (16,), const=True),
        "y": Tensor("y", "fp32", (16,)),
    }
    g = Graph(
        tensors=tensors,
        ops=(Op(id=0, kind="add", inputs=("x", "w"), outputs=("y",), attrs={}),),
        inputs=("x",),
        outputs=("y",),
    )
    names = {iv.tensor for iv in live_intervals(g)}
    assert "w" not in names, "consts are DDR-resident and streamed, never allocated"


def test_intervals_carry_byte_sizes():
    ivs = _by_name(live_intervals(_chain()))
    assert ivs["a"].nbytes == 64


def test_live_at_returns_only_overlapping_intervals():
    ivs = live_intervals(_chain())
    at_one = {iv.tensor for iv in live_at(ivs, 1)}
    assert "a" in at_one and "b" in at_one
    assert "y" not in at_one


def test_intervals_are_sorted_by_first_use():
    ivs = live_intervals(_chain())
    assert [iv.first_use for iv in ivs] == sorted(iv.first_use for iv in ivs)


def test_a_tensor_that_is_never_read_is_still_given_an_interval():
    # A dead store still occupies memory when it is written. Dropping it here
    # would understate the high-water mark.
    tensors = {
        "x": Tensor("x", "fp32", (16,)),
        "dead": Tensor("dead", "fp32", (16,)),
        "y": Tensor("y", "fp32", (16,)),
    }
    g = Graph(
        tensors=tensors,
        ops=(
            Op(id=0, kind="scale", inputs=("x",), outputs=("dead",), attrs={"factor": 1.0}),
            Op(id=1, kind="scale", inputs=("x",), outputs=("y",), attrs={"factor": 1.0}),
        ),
        inputs=("x",),
        outputs=("y",),
    )
    ivs = _by_name(live_intervals(g))
    assert "dead" in ivs
    assert ivs["dead"].first_use == ivs["dead"].last_use == 0


def test_invalid_graph_is_an_err():
    assert isinstance(live_intervals(Graph(tensors={}, ops=(), inputs=(), outputs=())), Err)


def test_the_whole_encoder_produces_one_interval_per_activation():
    g = order(build_vision_encoder(qwen35_at(256)))
    ivs = live_intervals(g)
    activations = {t.name for t in g.tensors.values() if not t.const}
    assert {iv.tensor for iv in ivs} == activations

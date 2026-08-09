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


def test_overlaps_adjacent_intervals_that_share_a_boundary():
    # Adjacent intervals [0,1] and [1,2] ARE simultaneously live at step 1,
    # where op 1 reads the first and writes the second. Overlaps must report
    # True to prevent the allocator from aliasing their storage.
    a = Interval(tensor="a", first_use=0, last_use=1, nbytes=64)
    b = Interval(tensor="b", first_use=1, last_use=2, nbytes=64)
    assert a.overlaps(b), "adjacent intervals must overlap to prevent data corruption"
    # Verify overlaps() and live_at() agree on which intervals are live at step 1.
    assert a.covers(1) and b.covers(1), "both must cover step 1"
    assert a.overlaps(b), "overlaps() must agree that both are simultaneously live"


def test_overlaps_disjoint_intervals():
    a = Interval(tensor="a", first_use=0, last_use=1, nbytes=64)
    c = Interval(tensor="c", first_use=2, last_use=3, nbytes=64)
    assert not a.overlaps(c), "disjoint intervals must not overlap"


def test_overlaps_nested_intervals():
    outer = Interval(tensor="outer", first_use=0, last_use=3, nbytes=64)
    inner = Interval(tensor="inner", first_use=1, last_use=2, nbytes=64)
    assert outer.overlaps(inner), "nested intervals must overlap"


def test_the_whole_encoder_produces_one_interval_per_activation():
    g = order(build_vision_encoder(qwen35_at(256)))
    ivs = live_intervals(g)
    activations = {t.name for t in g.tensors.values() if not t.const}
    # Names must match activations.
    assert {iv.tensor for iv in ivs} == activations
    # Every interval must be valid (first_use <= last_use).
    assert all(iv.first_use <= iv.last_use for iv in ivs), "all intervals must have first_use <= last_use"
    # Every graph output must stay live to the end.
    for name in g.outputs:
        if not g.tensor(name).const:
            output_iv = next(iv for iv in ivs if iv.tensor == name)
            assert output_iv.last_use == len(g.ops), f"output {name} must stay live to len(ops)"
    # Pick a known first op output and verify its interval starts at 0.
    if g.ops:
        first_op = g.ops[0]
        for out_name in first_op.outputs:
            if not g.tensor(out_name).const:
                out_iv = next(iv for iv in ivs if iv.tensor == out_name)
                assert out_iv.first_use == 0, f"output {out_name} of first op must have first_use == 0"

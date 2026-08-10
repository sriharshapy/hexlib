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


def _two_chains():
    """Two independent matmul chains with large intermediates.

    asap declares both large ops (0 and 1) first, forcing both intermediates
    to be live simultaneously. min_peak should prefer to complete the first
    chain (op 2 after op 0) before starting the second, keeping only one
    intermediate live at a time.

    Declaration order:
      op0: p = matmul(x, w1)    -> (4,256) ~4 KB
      op1: q = matmul(x, w2)    -> (4,256) ~4 KB
      op2: pa = matmul(p, u1)   -> (4,4)   ~64 B  (p's last use)
      op3: qb = matmul(q, u2)   -> (4,4)   ~64 B  (q's last use)
      op4: y = add(pa, qb)

    asap would hold both p and q live together.
    min_peak should prefer op2 immediately after op0.
    """
    tensors = {
        "x": Tensor("x", "fp32", (4, 4)),
        "w1": Tensor("w1", "fp32", (4, 256), const=True),
        "w2": Tensor("w2", "fp32", (4, 256), const=True),
        "u1": Tensor("u1", "fp32", (256, 4), const=True),
        "u2": Tensor("u2", "fp32", (256, 4), const=True),
        "p": Tensor("p", "fp32", (4, 256)),
        "q": Tensor("q", "fp32", (4, 256)),
        "pa": Tensor("pa", "fp32", (4, 4)),
        "qb": Tensor("qb", "fp32", (4, 4)),
        "y": Tensor("y", "fp32", (4, 4)),
    }
    return Graph(
        tensors=tensors,
        ops=(
            Op(id=0, kind="matmul", inputs=("x", "w1"), outputs=("p",), attrs={}),
            Op(id=1, kind="matmul", inputs=("x", "w2"), outputs=("q",), attrs={}),
            Op(id=2, kind="matmul", inputs=("p", "u1"), outputs=("pa",), attrs={}),
            Op(id=3, kind="matmul", inputs=("q", "u2"), outputs=("qb",), attrs={}),
            Op(id=4, kind="add", inputs=("pa", "qb"), outputs=("y",), attrs={}),
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


def test_two_ops_with_a_circular_data_dependency_are_caught_by_structural_validation():
    # Renamed from test_a_cycle_is_an_err_naming_the_stuck_ops: that name
    # claimed to exercise order()'s own ready-set "cycle or unreachable ops"
    # branch, but it never does. Graph.problems() checks read-before-write
    # against a running "written" set built by walking graph.ops in
    # DECLARATION order; a genuine cycle (op 0 needs op 1's output, op 1
    # needs op 0's) can never pass that check, because whichever op is
    # declared first is missing its dependency right there. So
    # graph.problems() -- which order() runs BEFORE ever reaching its own
    # ready-set loop -- always catches a true cycle first, and the
    # ready-set's own "stuck" branch is unreachable from here. This test
    # documents what actually gets exercised: the pre-check, not the
    # scheduler's cycle detection.
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
    assert "structurally invalid" in out.reason
    assert "op 0" in out.detail


def test_peak_live_bytes_is_positive_and_at_least_the_largest_tensor():
    g = order(_diamond())
    assert peak_live_bytes(g) >= 1024 * 4


def test_the_whole_encoder_orders_under_both_policies():
    g = build_vision_encoder(qwen35_at(256))
    for name in ORDER_POLICIES:
        out = order(g, policy=name)
        assert not isinstance(out, Err), name
        assert out.problems() == [], name


def test_undeclared_output_tensor_returns_err():
    """Verify that malformed input (undeclared output) returns Err, not raises."""
    tensors = {
        "x": Tensor("x", "fp32", (4,)),
        "y": Tensor("y", "fp32", (4,)),
    }
    g = Graph(
        tensors=tensors,
        ops=(Op(id=0, kind="scale", inputs=("x",), outputs=("z",), attrs={"factor": 1.0}),),
        inputs=("x",),
        outputs=("y",),
    )
    out = order(g)
    assert isinstance(out, Err)
    assert "structurally invalid" in out.reason or "structurally invalid" in out.detail


def test_a_policy_that_invents_an_op_outside_the_ready_set_is_an_err(monkeypatch):
    # Was `assert chosen in ready` -- a bare assert inside a pass contracted
    # never to raise, and one that silently vanishes under `-O`. A
    # misbehaving policy (a bug in a future custom one, not either shipped
    # policy) must come back as an Err, not an AssertionError.
    rogue_op = Op(id=999, kind="scale", inputs=(), outputs=(), attrs={"factor": 1.0})
    monkeypatch.setitem(ORDER_POLICIES, "rogue", lambda ready, live_bytes, graph: rogue_op)
    out = order(_diamond(), policy="rogue")
    assert isinstance(out, Err)
    assert "ready set" in out.reason


def test_min_peak_beats_asap_on_two_independent_chains():
    """Verify min_peak actually beats asap when declaration order is adversarial.

    Two independent chains with large intermediates; asap holds both live,
    min_peak completes the first chain before starting the second.
    """
    asap = peak_live_bytes(order(_two_chains(), policy="asap"))
    min_peak = peak_live_bytes(order(_two_chains(), policy="min_peak"))
    assert min_peak < asap, f"min_peak ({min_peak}) should beat asap ({asap})"

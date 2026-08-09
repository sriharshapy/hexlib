from __future__ import annotations

import numpy as np

import hexlib.graph.opdefs  # noqa: F401
from hexlib.graph import eager
from hexlib.graph.ir import Graph, Op, Tensor
from hexlib.result import Err


def _linear_graph():
    tensors = {
        "x": Tensor("x", "fp32", (2, 3)),
        "w": Tensor("w", "fp32", (3, 4), const=True),
        "b": Tensor("b", "fp32", (4,), const=True),
        "h": Tensor("h", "fp32", (2, 4)),
        "y": Tensor("y", "fp32", (2, 4)),
    }
    ops = (
        Op(id=0, kind="matmul", inputs=("x", "w"), outputs=("h",), attrs={}),
        Op(id=1, kind="add", inputs=("h", "b"), outputs=("y",), attrs={}),
    )
    return Graph(tensors=tensors, ops=ops, inputs=("x",), outputs=("y",))


def test_runs_a_two_op_graph():
    g = _linear_graph()
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    w = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.arange(4, dtype=np.float32)
    out = eager.run(g, {"x": x, "w": w, "b": b})
    assert not isinstance(out, Err)
    np.testing.assert_allclose(out["y"], x @ w + b)


def test_returns_only_the_declared_outputs():
    out = eager.run(
        _linear_graph(),
        {
            "x": np.zeros((2, 3), np.float32),
            "w": np.zeros((3, 4), np.float32),
            "b": np.zeros(4, np.float32),
        },
    )
    assert set(out) == {"y"}


def test_missing_feed_is_an_err_naming_the_tensor():
    out = eager.run(_linear_graph(), {"x": np.zeros((2, 3), np.float32)})
    assert isinstance(out, Err)
    assert "w" in out.reason or "w" in out.detail


def test_wrong_feed_shape_is_an_err_naming_both_shapes():
    out = eager.run(
        _linear_graph(),
        {
            "x": np.zeros((5, 3), np.float32),
            "w": np.zeros((3, 4), np.float32),
            "b": np.zeros(4, np.float32),
        },
    )
    assert isinstance(out, Err)
    assert "(5, 3)" in out.detail and "(2, 3)" in out.detail


def test_unregistered_op_kind_is_an_err_not_a_skip():
    # Spec 8: "A graph containing an op with no registry entry is an error at
    # pass time, not a silent skip."
    g = Graph(
        tensors={"x": Tensor("x", "fp32", (2,)), "y": Tensor("y", "fp32", (2,))},
        ops=(Op(id=0, kind="does_not_exist", inputs=("x",), outputs=("y",), attrs={}),),
        inputs=("x",),
        outputs=("y",),
    )
    out = eager.run(g, {"x": np.zeros(2, np.float32)})
    assert isinstance(out, Err)
    assert "does_not_exist" in out.detail


def test_structurally_invalid_graph_is_an_err_before_anything_runs():
    g = Graph(
        tensors={"x": Tensor("x", "fp32", (2,))},
        ops=(Op(id=0, kind="scale", inputs=("x",), outputs=("gone",), attrs={"factor": 1.0}),),
        inputs=("x",),
        outputs=("gone",),
    )
    out = eager.run(g, {"x": np.zeros(2, np.float32)})
    assert isinstance(out, Err)
    assert "gone" in out.detail


def test_wrong_output_count_from_a_reference_is_an_err():
    from hexlib.graph.ops import OpDef, Registry

    reg = Registry()
    reg.register(
        OpDef(
            kind="two_out",
            infer=lambda inputs, attrs: (((2,), "fp32"),),
            working_set=lambda i, o, a: 1,
            reference=lambda arrays, attrs: (arrays[0], arrays[0]),  # two, not one
        )
    )
    g = Graph(
        tensors={"x": Tensor("x", "fp32", (2,)), "y": Tensor("y", "fp32", (2,))},
        ops=(Op(id=0, kind="two_out", inputs=("x",), outputs=("y",), attrs={}),),
        inputs=("x",),
        outputs=("y",),
    )
    out = eager.run(g, {"x": np.zeros(2, np.float32)}, registry=reg)
    assert isinstance(out, Err)
    assert "1" in out.detail and "2" in out.detail


def test_result_shape_disagreeing_with_the_declared_tensor_is_an_err():
    from hexlib.graph.ops import OpDef, Registry

    reg = Registry()
    reg.register(
        OpDef(
            kind="wrong_shape",
            infer=lambda inputs, attrs: (((2,), "fp32"),),
            working_set=lambda i, o, a: 1,
            reference=lambda arrays, attrs: (np.zeros(99, np.float32),),
        )
    )
    g = Graph(
        tensors={"x": Tensor("x", "fp32", (2,)), "y": Tensor("y", "fp32", (2,))},
        ops=(Op(id=0, kind="wrong_shape", inputs=("x",), outputs=("y",), attrs={}),),
        inputs=("x",),
        outputs=("y",),
    )
    out = eager.run(g, {"x": np.zeros(2, np.float32)}, registry=reg)
    assert isinstance(out, Err)
    assert "99" in out.detail


def test_q4_0_tensors_are_fed_as_fp32_in_the_oracle():
    # The oracle validates MATH, not precision. A q4_0 weight is fed as the
    # fp32 values it represents; quantization error is measured separately.
    tensors = {
        "x": Tensor("x", "fp16", (2, 32)),
        "w": Tensor("w", "q4_0", (32, 4), const=True),
        "y": Tensor("y", "fp16", (2, 4)),
    }
    g = Graph(
        tensors=tensors,
        ops=(Op(id=0, kind="matmul", inputs=("x", "w"), outputs=("y",), attrs={}),),
        inputs=("x",),
        outputs=("y",),
    )
    out = eager.run(
        g, {"x": np.ones((2, 32), np.float32), "w": np.ones((32, 4), np.float32)}
    )
    assert not isinstance(out, Err)
    np.testing.assert_allclose(out["y"], np.full((2, 4), 32.0))

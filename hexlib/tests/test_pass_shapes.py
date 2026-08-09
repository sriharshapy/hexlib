from __future__ import annotations

import numpy as np

import hexlib.graph.opdefs  # noqa: F401
from hexlib.graph.ir import Graph, Op, Tensor
from hexlib.graph.shapes import infer_shapes
from hexlib.models.qwen35 import qwen35_at
from hexlib.models.vit import build_vision_encoder
from hexlib.result import Err


def _g(out_shape=(2, 4), out_dtype="fp32"):
    return Graph(
        tensors={
            "x": Tensor("x", "fp32", (2, 3)),
            "w": Tensor("w", "fp32", (3, 4), const=True),
            "y": Tensor("y", out_dtype, out_shape),
        },
        ops=(Op(id=0, kind="matmul", inputs=("x", "w"), outputs=("y",), attrs={}),),
        inputs=("x",),
        outputs=("y",),
    )


def test_a_consistent_graph_passes_unchanged():
    g = _g()
    out = infer_shapes(g)
    assert not isinstance(out, Err)
    assert out.tensors["y"].shape == (2, 4)


def test_the_pass_does_not_mutate_its_input():
    g = _g()
    before = {n: (t.shape, t.dtype) for n, t in g.tensors.items()}
    infer_shapes(g)
    after = {n: (t.shape, t.dtype) for n, t in g.tensors.items()}
    assert before == after


def test_declared_shape_disagreeing_with_infer_is_an_err():
    out = infer_shapes(_g(out_shape=(2, 99)))
    assert isinstance(out, Err)
    assert "(2, 99)" in out.detail and "(2, 4)" in out.detail
    assert "y" in out.detail


def test_declared_dtype_disagreeing_with_infer_is_an_err():
    out = infer_shapes(_g(out_dtype="int32"))
    assert isinstance(out, Err)
    assert "int32" in out.detail and "fp32" in out.detail


def test_infer_raising_is_an_err_naming_the_op():
    g = Graph(
        tensors={
            "x": Tensor("x", "fp32", (2, 3)),
            "w": Tensor("w", "fp32", (5, 4), const=True),  # inner dim disagrees
            "y": Tensor("y", "fp32", (2, 4)),
        },
        ops=(Op(id=0, kind="matmul", inputs=("x", "w"), outputs=("y",), attrs={}),),
        inputs=("x",),
        outputs=("y",),
    )
    out = infer_shapes(g)
    assert isinstance(out, Err)
    assert "op 0" in out.detail and "matmul" in out.detail


def test_unregistered_kind_is_an_err_not_a_skip():
    g = Graph(
        tensors={"x": Tensor("x", "fp32", (2,)), "y": Tensor("y", "fp32", (2,))},
        ops=(Op(id=0, kind="nope", inputs=("x",), outputs=("y",), attrs={}),),
        inputs=("x",),
        outputs=("y",),
    )
    out = infer_shapes(g)
    assert isinstance(out, Err)
    assert "nope" in out.detail


def test_output_count_mismatch_is_an_err():
    from hexlib.graph.ops import OpDef, Registry

    reg = Registry()
    reg.register(
        OpDef(
            kind="two",
            infer=lambda inputs, attrs: (((2,), "fp32"), ((2,), "fp32")),
            working_set=lambda i, o, a: 1,
            reference=lambda arrays, attrs: (arrays[0], arrays[0]),
        )
    )
    g = Graph(
        tensors={"x": Tensor("x", "fp32", (2,)), "y": Tensor("y", "fp32", (2,))},
        ops=(Op(id=0, kind="two", inputs=("x",), outputs=("y",), attrs={}),),
        inputs=("x",),
        outputs=("y",),
    )
    out = infer_shapes(g, registry=reg)
    assert isinstance(out, Err)
    assert "1" in out.detail and "2" in out.detail


def test_structurally_invalid_graph_is_an_err_before_inference():
    g = Graph(tensors={}, ops=(), inputs=(), outputs=())
    assert isinstance(infer_shapes(g), Err)


def test_the_whole_qwen35_encoder_at_256_infers_clean():
    g = build_vision_encoder(qwen35_at(256))
    out = infer_shapes(g)
    assert not isinstance(out, Err), getattr(out, "detail", "")

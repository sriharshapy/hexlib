from __future__ import annotations

import numpy as np

import hexlib.graph.opdefs  # noqa: F401
from hexlib.graph import eager
from hexlib.graph.fuse import fuse
from hexlib.graph.ir import Graph, Op, Tensor
from hexlib.graph.shapes import infer_shapes
from hexlib.models.qwen35 import qwen35_at, qwen35_oracle
from hexlib.models.vit import VitConfig, build_vision_encoder
from hexlib.result import Err


def _mlp_graph(act="gelu_tanh"):
    tensors = {
        "x": Tensor("x", "fp32", (4, 8)),
        "w": Tensor("w", "fp32", (8, 16), const=True),
        "b": Tensor("b", "fp32", (16,), const=True),
        "mm": Tensor("mm", "fp32", (4, 16)),
        "bias": Tensor("bias", "fp32", (4, 16)),
        "y": Tensor("y", "fp32", (4, 16)),
    }
    ops = (
        Op(id=0, kind="matmul", inputs=("x", "w"), outputs=("mm",), attrs={}),
        Op(id=1, kind="add", inputs=("mm", "b"), outputs=("bias",), attrs={}),
        Op(id=2, kind=act, inputs=("bias",), outputs=("y",), attrs={}),
    )
    return Graph(tensors=tensors, ops=ops, inputs=("x",), outputs=("y",))


def test_matmul_bias_act_becomes_one_op():
    g = fuse(_mlp_graph())
    assert not isinstance(g, Err)
    kinds = [op.kind for op in g.ops]
    assert kinds == ["matmul_epilogue"]
    assert g.ops[0].attrs["act"] == "gelu_tanh"


def test_matmul_bias_without_an_activation_fuses_with_act_none():
    tensors = {
        "x": Tensor("x", "fp32", (4, 8)),
        "w": Tensor("w", "fp32", (8, 16), const=True),
        "b": Tensor("b", "fp32", (16,), const=True),
        "mm": Tensor("mm", "fp32", (4, 16)),
        "y": Tensor("y", "fp32", (4, 16)),
    }
    g = Graph(
        tensors=tensors,
        ops=(
            Op(id=0, kind="matmul", inputs=("x", "w"), outputs=("mm",), attrs={}),
            Op(id=1, kind="add", inputs=("mm", "b"), outputs=("y",), attrs={}),
        ),
        inputs=("x",),
        outputs=("y",),
    )
    out = fuse(g)
    assert [op.kind for op in out.ops] == ["matmul_epilogue"]
    assert out.ops[0].attrs["act"] == "none"


def test_gelu_erf_fuses_too_and_keeps_its_identity():
    g = fuse(_mlp_graph(act="gelu_erf"))
    assert g.ops[0].attrs["act"] == "gelu_erf"


def test_the_fused_graph_computes_the_same_answer():
    # This is the point of rewriting rather than annotating: the M0 oracle can
    # run the fused graph, so fusion can never quietly change the answer.
    plain = _mlp_graph()
    fused = fuse(plain)
    rs = np.random.RandomState(11)
    feeds = {
        "x": rs.standard_normal((4, 8)).astype(np.float32),
        "w": rs.standard_normal((8, 16)).astype(np.float32),
        "b": rs.standard_normal(16).astype(np.float32),
    }
    a = eager.run(plain, feeds)
    b = eager.run(fused, feeds)
    assert not isinstance(a, Err) and not isinstance(b, Err)
    np.testing.assert_allclose(a["y"], b["y"], rtol=1e-6, atol=1e-6)


def test_a_residual_add_is_not_mistaken_for_a_bias_add():
    # add(x, proj) where both are activations must NOT fuse into the matmul.
    tensors = {
        "x": Tensor("x", "fp32", (4, 8)),
        "w": Tensor("w", "fp32", (8, 8), const=True),
        "mm": Tensor("mm", "fp32", (4, 8)),
        "y": Tensor("y", "fp32", (4, 8)),
    }
    g = Graph(
        tensors=tensors,
        ops=(
            Op(id=0, kind="matmul", inputs=("x", "w"), outputs=("mm",), attrs={}),
            Op(id=1, kind="add", inputs=("mm", "x"), outputs=("y",), attrs={}),
        ),
        inputs=("x",),
        outputs=("y",),
    )
    out = fuse(g)
    assert [op.kind for op in out.ops] == ["matmul", "add"]


def test_an_intermediate_with_two_consumers_does_not_fuse():
    tensors = {
        "x": Tensor("x", "fp32", (4, 8)),
        "w": Tensor("w", "fp32", (8, 16), const=True),
        "b": Tensor("b", "fp32", (16,), const=True),
        "mm": Tensor("mm", "fp32", (4, 16)),
        "y": Tensor("y", "fp32", (4, 16)),
        "z": Tensor("z", "fp32", (4, 16)),
    }
    g = Graph(
        tensors=tensors,
        ops=(
            Op(id=0, kind="matmul", inputs=("x", "w"), outputs=("mm",), attrs={}),
            Op(id=1, kind="add", inputs=("mm", "b"), outputs=("y",), attrs={}),
            Op(id=2, kind="scale", inputs=("mm",), outputs=("z",), attrs={"factor": 2.0}),
        ),
        inputs=("x",),
        outputs=("y", "z"),
    )
    out = fuse(g)
    assert "matmul" in [op.kind for op in out.ops]


def test_an_intermediate_that_is_a_graph_output_does_not_fuse():
    g = _mlp_graph()
    g = Graph(tensors=g.tensors, ops=g.ops, inputs=g.inputs, outputs=("mm", "y"))
    out = fuse(g)
    assert "matmul" in [op.kind for op in out.ops]


def test_fusion_is_idempotent():
    once = fuse(_mlp_graph())
    twice = fuse(once)
    assert [op.kind for op in once.ops] == [op.kind for op in twice.ops]


def test_fuse_returns_the_same_object_when_nothing_fuses():
    # Identity, not equality: Graph is frozen with tuple containers, so a
    # before/after comparison cannot fail regardless of what the pass does.
    g = Graph(
        tensors={"x": Tensor("x", "fp32", (4,)), "y": Tensor("y", "fp32", (4,))},
        ops=(Op(id=0, kind="scale", inputs=("x",), outputs=("y",), attrs={"factor": 2.0}),),
        inputs=("x",),
        outputs=("y",),
    )
    assert fuse(g) is g


def test_fuse_returns_a_new_object_when_it_fuses():
    g = _mlp_graph()
    out = fuse(g)
    assert out is not g
    assert [op.kind for op in g.ops] == ["matmul", "add", "gelu_tanh"]


def test_the_fused_graph_still_validates_and_infers():
    g = fuse(build_vision_encoder(qwen35_at(256)))
    assert not isinstance(g, Err), getattr(g, "detail", "")
    assert g.problems() == []
    assert not isinstance(infer_shapes(g), Err)


def test_fusion_removes_exactly_the_fusable_sites_from_the_encoder():
    cfg = qwen35_at(256)
    plain = build_vision_encoder(cfg)
    fused = fuse(plain)

    # Counted against the real 396-op graph, not estimated:
    #   6 biased matmuls per block (q, k, v, o, fc1, fc2) = 72
    #   + patch_embed + merger fc1 + merger fc2                = 75 biased matmuls
    #   2 unbiased matmuls per block (scores, ctx)             = 24, which do NOT fuse
    #   12 gelu_tanh + 1 gelu_erf                              = 13 activations folded
    n_biased = 6 * cfg.depth + 3
    n_unbiased = 2 * cfg.depth
    n_acts = cfg.depth + 1

    kinds = [op.kind for op in fused.ops]
    assert kinds.count("matmul_epilogue") == n_biased
    assert kinds.count("matmul") == n_unbiased, (
        "the attention score and context matmuls carry no bias and must not fuse"
    )

    # Each fused site removes its bias add; each folded activation removes an op.
    saved = len(plain.ops) - len(fused.ops)
    assert saved == n_biased + n_acts, f"expected {n_biased + n_acts} fused away, got {saved}"


def test_the_whole_tiny_encoder_gives_the_same_answer_fused():
    cfg = VitConfig(
        depth=2, hidden_size=64, num_heads=4, intermediate_size=128, patch_size=4,
        temporal_patch_size=2, in_channels=3, spatial_merge_size=2, out_hidden_size=32,
        layernorm_eps=1e-6, rope_theta=10000.0, image_size=32,
        act_dtype="fp32", weight_dtype="fp32",
    )
    plain = build_vision_encoder(cfg)
    fused = fuse(plain)
    rs = np.random.RandomState(5)
    feeds = {
        t.name: rs.standard_normal(t.shape).astype(np.float32) * 0.05
        for t in plain.tensors.values()
        if t.const
    }
    (in_name,) = plain.inputs
    feeds[in_name] = rs.standard_normal(plain.tensor(in_name).shape).astype(np.float32)
    a = eager.run(plain, feeds)
    b = eager.run(fused, feeds)
    assert not isinstance(a, Err) and not isinstance(b, Err)
    (out_name,) = plain.outputs
    np.testing.assert_allclose(a[out_name], b[out_name], rtol=1e-5, atol=1e-6)

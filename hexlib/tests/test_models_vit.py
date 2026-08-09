from __future__ import annotations

import numpy as np
import pytest
from dataclasses import fields
from typing import get_type_hints

import hexlib.graph.opdefs  # noqa: F401
from hexlib.graph import eager
from hexlib.models.qwen35 import QWEN35_08B_VISION, qwen35_at
from hexlib.models.vit import VitConfig, build_vision_encoder, weight_names, _get_positive_int_field_names
from hexlib.result import Err


def _tiny() -> VitConfig:
    return VitConfig(
        depth=2,
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        patch_size=4,
        temporal_patch_size=2,
        in_channels=3,
        spatial_merge_size=2,
        out_hidden_size=32,
        layernorm_eps=1e-6,
        rope_theta=10000.0,
        image_size=32,
        act_dtype="fp32",
        weight_dtype="fp32",
    )


def test_tiny_graph_builds_and_validates():
    g = build_vision_encoder(_tiny())
    assert not isinstance(g, Err), getattr(g, "detail", "")
    assert g.problems() == []


def test_graph_output_shape_is_tokens_over_merge_squared_by_out_hidden():
    cfg = _tiny()
    g = build_vision_encoder(cfg)
    (out_name,) = g.outputs
    n = (cfg.image_size // cfg.patch_size) ** 2
    assert g.tensor(out_name).shape == (n // cfg.spatial_merge_size**2, cfg.out_hidden_size)


def test_graph_input_is_the_image():
    cfg = _tiny()
    g = build_vision_encoder(cfg)
    (in_name,) = g.inputs
    assert g.tensor(in_name).shape == (
        cfg.in_channels, cfg.temporal_patch_size, cfg.image_size, cfg.image_size
    )


def test_every_op_kind_used_is_registered():
    from hexlib.graph.ops import all_kinds

    g = build_vision_encoder(_tiny())
    known = set(all_kinds())
    used = {op.kind for op in g.ops}
    assert used <= known, f"unregistered kinds in graph: {sorted(used - known)}"


def test_both_gelus_appear_exactly_where_expected():
    cfg = _tiny()
    g = build_vision_encoder(cfg)
    kinds = [op.kind for op in g.ops]
    assert kinds.count("gelu_tanh") == cfg.depth   # one per block MLP
    assert kinds.count("gelu_erf") == 1            # the merger, once


def test_layernorm_count_is_two_per_block_plus_the_merger():
    cfg = _tiny()
    g = build_vision_encoder(cfg)
    kinds = [op.kind for op in g.ops]
    assert kinds.count("layernorm") == 2 * cfg.depth + 1


def test_no_standalone_prenorm_or_postnorm():
    # The first op after patch embedding must not be a layernorm, and the op
    # feeding the merger's layernorm must be a residual add, not another norm.
    cfg = _tiny()
    g = build_vision_encoder(cfg)
    kinds = [op.kind for op in g.ops]
    first_ln = kinds.index("layernorm")
    assert kinds[first_ln - 1] == "add", "a standalone pre-norm crept in before block 0"
    last_ln = len(kinds) - 1 - kinds[::-1].index("layernorm")
    assert kinds[last_ln - 1] == "add", "a standalone post-norm crept in after the blocks"


def test_residual_adds_read_the_block_input_not_the_norm_output():
    # Both operands of a residual add have the same shape, so add(a, proj)
    # (the normalized tensor) instead of add(x, proj) (the block's actual
    # input) passes every shape/structural check and the finite-output
    # end-to-end test, and would only surface later as an unexplained
    # numerical mismatch. Pin the wiring directly, on the op's inputs.
    cfg = _tiny()
    g = build_vision_encoder(cfg)
    ops_by_output = {op.outputs[0]: op for op in g.ops}

    attn_resid = ops_by_output["blk0.resid1"]
    assert attn_resid.kind == "add"
    assert "x0" in attn_resid.inputs, "attention residual must add block 0's own input"
    assert "blk0.ln1.out" not in attn_resid.inputs, (
        "attention residual reads the normalized tensor instead of the block input"
    )

    mlp_resid = ops_by_output["blk0.resid2"]
    assert mlp_resid.kind == "add"
    assert "blk0.resid1" in mlp_resid.inputs, "MLP residual must add the post-attention stream"
    assert "blk0.ln2.out" not in mlp_resid.inputs, (
        "MLP residual reads the normalized tensor instead of the post-attention stream"
    )


def test_qkv_is_three_separate_matmuls_not_one():
    cfg = _tiny()
    g = build_vision_encoder(cfg)
    names = set(weight_names(cfg))
    for layer in range(cfg.depth):
        for p in ("q", "k", "v"):
            assert f"blk{layer}.w{p}" in names
            assert f"blk{layer}.b{p}" in names
    assert not any(".wqkv" in n for n in names)


def test_weights_are_stored_k_by_n_not_out_by_in():
    cfg = _tiny()
    g = build_vision_encoder(cfg)
    assert g.tensor("blk0.w_fc1").shape == (cfg.hidden_size, cfg.intermediate_size)
    assert g.tensor("blk0.w_fc2").shape == (cfg.intermediate_size, cfg.hidden_size)


def test_rope_is_applied_to_q_and_k_but_not_v():
    cfg = _tiny()
    g = build_vision_encoder(cfg)
    rope_ops = [op for op in g.ops if op.kind == "rope_2d"]
    assert len(rope_ops) == 2 * cfg.depth  # exactly q and k, every block


def test_pos_embed_and_rope_tables_are_const_and_folded():
    cfg = _tiny()
    g = build_vision_encoder(cfg)
    n = (cfg.image_size // cfg.patch_size) ** 2
    d = cfg.hidden_size // cfg.num_heads
    assert g.tensor("pos_embed").const and g.tensor("pos_embed").shape == (n, cfg.hidden_size)
    assert g.tensor("rope_cos").const and g.tensor("rope_cos").shape == (n, d)
    assert g.tensor("rope_sin").const and g.tensor("rope_sin").shape == (n, d)


def test_weight_names_matches_the_graph_consts_exactly():
    cfg = _tiny()
    g = build_vision_encoder(cfg)
    graph_consts = {t.name for t in g.tensors.values() if t.const}
    assert set(weight_names(cfg)) == graph_consts


def test_graph_runs_end_to_end_on_random_weights():
    cfg = _tiny()
    g = build_vision_encoder(cfg)
    rs = np.random.RandomState(0)
    feeds = {
        t.name: rs.standard_normal(t.shape).astype(np.float32) * 0.05
        for t in g.tensors.values()
        if t.const
    }
    (in_name,) = g.inputs
    feeds[in_name] = rs.standard_normal(g.tensor(in_name).shape).astype(np.float32)
    out = eager.run(g, feeds)
    assert not isinstance(out, Err), getattr(out, "detail", "")
    (out_name,) = g.outputs
    assert np.all(np.isfinite(out[out_name]))


def test_non_divisible_grid_is_an_err_not_a_crash():
    cfg = VitConfig(**{**_tiny().__dict__, "image_size": 12})  # grid 3, merge 2
    result = build_vision_encoder(cfg)
    assert isinstance(result, Err)
    assert "merge" in result.detail or "divisible" in result.detail


def test_zero_patch_size_is_an_err_not_a_zerodivisionerror():
    cfg = VitConfig(**{**_tiny().__dict__, "patch_size": 0})
    result = build_vision_encoder(cfg)
    assert isinstance(result, Err)
    assert "patch_size" in result.detail
    assert "0" in result.detail


def test_zero_num_heads_is_an_err_not_a_zerodivisionerror():
    cfg = VitConfig(**{**_tiny().__dict__, "num_heads": 0})
    result = build_vision_encoder(cfg)
    assert isinstance(result, Err)
    assert "num_heads" in result.detail
    assert "0" in result.detail


def test_zero_spatial_merge_size_is_an_err_not_a_zerodivisionerror():
    cfg = VitConfig(**{**_tiny().__dict__, "spatial_merge_size": 0})
    result = build_vision_encoder(cfg)
    assert isinstance(result, Err)
    assert "spatial_merge_size" in result.detail
    assert "0" in result.detail


def test_negative_image_size_is_an_err_not_a_slip_through_divisibility():
    # -32 % 16 == 0, so a divisibility-only guard on image_size lets a
    # negative value slip through and reach Tensor.__post_init__, which
    # raises. `_config_problems` must reject it on its own, before any %.
    cfg = VitConfig(**{**_tiny().__dict__, "image_size": -32})
    result = build_vision_encoder(cfg)
    assert isinstance(result, Err)
    assert "image_size" in result.detail
    assert "-32" in result.detail


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_size", 0),
        ("hidden_size", -64),
        ("in_channels", 0),
        ("temporal_patch_size", 0),
        ("intermediate_size", 0),
        ("out_hidden_size", 0),
        ("depth", 0),
        ("depth", -1),
    ],
)
def test_every_dimension_bearing_field_is_guarded_not_just_the_three_divisors(field, value):
    # _config_problems previously only guarded the three fields used directly
    # as divisors (patch_size, spatial_merge_size, num_heads). Every other
    # dimension-bearing field reached Tensor.__post_init__ unguarded and
    # raised ValueError instead of returning Err.
    cfg = VitConfig(**{**_tiny().__dict__, field: value})
    result = build_vision_encoder(cfg)
    assert isinstance(result, Err), f"{field}={value} should be Err, was {result!r}"
    assert field in result.detail
    assert str(value) in result.detail


def test_guard_tracks_vit_config_int_fields_not_a_constant():
    # The guard derives int field names from VitConfig's type annotations
    # rather than maintaining a hand-written list. This test proves the
    # derivation is structural: if someone adds a new int field to VitConfig,
    # the guard immediately catches it without modifying this function.

    # Get the set of int-typed fields directly from the dataclass.
    hints = get_type_hints(VitConfig)
    declared_int_fields = {
        field.name for field in fields(VitConfig)
        if hints.get(field.name) is int
    }

    # Get the set of fields the guard actually checks.
    guarded_fields = _get_positive_int_field_names()

    # They must be identical.
    assert guarded_fields == declared_int_fields, (
        f"guard checks {sorted(guarded_fields)} but VitConfig has int fields "
        f"{sorted(declared_int_fields)}; the guard is out of sync"
    )


def test_weight_names_on_a_bad_config_is_an_err_not_an_empty_tuple():
    # A caller doing `for name in weight_names(cfg): load(name)` on a bad
    # config must not silently load zero weights and proceed as if it had
    # succeeded -- () and Err must not be interchangeable here.
    cfg = VitConfig(**{**_tiny().__dict__, "image_size": 0})
    result = weight_names(cfg)
    assert isinstance(result, Err)


def test_qwen35_config_matches_the_checkpoint():
    cfg = QWEN35_08B_VISION
    assert cfg.depth == 12
    assert cfg.hidden_size == 768
    assert cfg.num_heads == 12
    assert cfg.intermediate_size == 3072
    assert cfg.patch_size == 16
    assert cfg.temporal_patch_size == 2
    assert cfg.in_channels == 3
    assert cfg.spatial_merge_size == 2
    assert cfg.out_hidden_size == 1024
    assert cfg.layernorm_eps == 1e-6


def test_qwen35_at_256_builds_the_whole_encoder():
    cfg = qwen35_at(256)
    g = build_vision_encoder(cfg)
    assert not isinstance(g, Err), getattr(g, "detail", "")
    assert g.problems() == []
    (out_name,) = g.outputs
    assert g.tensor(out_name).shape == (64, 1024)   # 256 patches -> 64 merged tokens
    assert len(g.ops) > 250                          # spec 2: "several hundred ops"


def test_qwen35_at_256_uses_quantized_weights_and_fp16_activations():
    cfg = qwen35_at(256)
    g = build_vision_encoder(cfg)
    assert g.tensor("blk0.w_fc1").dtype == "q4_0"
    assert g.tensor("blk0.b_fc1").dtype == "fp32"   # biases stay fp32, spec 5.1
    (out_name,) = g.outputs
    assert g.tensor(out_name).dtype == "fp16"


def test_every_inner_dim_is_divisible_by_32():
    # Spec 5.1: MUL_MAT requires ne[0] % 32 == 0 for the block-quantized path,
    # and HMX fp16 tiles are 32x32 (spec 4.3).
    cfg = qwen35_at(256)
    g = build_vision_encoder(cfg)
    for t in g.tensors.values():
        if t.dtype == "q4_0":
            assert t.shape[-2] % 32 == 0, f"{t.name} inner dim {t.shape[-2]}"
            assert t.shape[-1] % 32 == 0, f"{t.name} outer dim {t.shape[-1]}"


def test_every_op_infer_agrees_with_the_builders_own_declaration():
    """M1's `infer_shapes` pass (docs/superpowers/plans/
    2026-08-09-vlm-encoder-m1-pass-pipeline.md, Task 1) trusts that every op's
    own `infer` agrees with what the builder declared for its outputs. Prove
    that on the REAL shipped graph -- qwen35_at(256), fp16 activations over
    q4_0 weights -- not on a synthetic fixture where every dtype happens to be
    the same and a disagreement like this one cannot show up.

    This caught a real bug: patchify emits `patches` as fp32 (the host always
    hands in an fp32 image), but the patch-embed matmul's builder-declared
    output was `cfg.act_dtype` (fp16 for the shipped config) while
    `matmul.infer` takes the output dtype from its first input (patches,
    fp32) -- disagreement, undetected until this test walked infer over the
    real graph.
    """
    from hexlib.graph.ops import get

    g = build_vision_encoder(qwen35_at(256))
    assert not isinstance(g, Err), getattr(g, "detail", "")
    assert g.ops, "graph has no ops; this test would pass vacuously"

    mismatches = []
    for op in g.ops:
        inputs = tuple(g.tensor(name) for name in op.inputs)
        got = get(op.kind).infer(inputs, op.attrs)
        declared = tuple((g.tensor(name).shape, g.tensor(name).dtype) for name in op.outputs)
        if got != declared:
            mismatches.append(f"op {op.id} {op.kind} {op.outputs}: declared {declared}, infer says {got}")
    assert not mismatches, "\n".join(mismatches)

"""The M0 gate: hexlib's eager encoder must match the upstream implementation.

Reads committed golden vectors. Imports neither torch nor transformers, so it
runs everywhere and can never be skipped into a false pass.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import hexlib.graph.opdefs  # noqa: F401
from hexlib.graph import eager
from hexlib.models.vit import VitConfig, build_vision_encoder
from hexlib.result import Err

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
VISION_NPZ = os.path.join(DATA, "qwen35_vision_tiny.npz")

TINY_CFG = VitConfig(
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
    image_size=32,          # 8x8 patch grid -> 64 tokens
    act_dtype="fp32",
    weight_dtype="fp32",
)


def test_golden_file_exists_and_is_not_empty():
    # If the golden is missing, every test below would skip and the M0 gate
    # would report nothing while looking green. Fail loudly instead.
    assert os.path.isfile(VISION_NPZ), (
        f"{VISION_NPZ} is missing. Regenerate with: "
        'pip install -e ".[oracle]" && python scripts/gen_vision_oracle.py'
    )
    assert os.path.getsize(VISION_NPZ) > 1024, "golden is suspiciously small"


def test_golden_carries_every_array_the_test_needs():
    z = np.load(VISION_NPZ)
    for key in ("image", "patches", "patches_upstream", "expected_merged", "expected_last_hidden"):
        assert key in z, f"golden is missing {key!r}"
    assert any(k.startswith("param::") for k in z.files), "golden carries no weights"


def test_pos_embed_grid_is_mismatched_with_the_patch_grid_on_purpose():
    """The learned position-embedding grid must NOT equal the patch grid.

    If they matched (as an earlier version of this golden did, at 8x8 learned
    vs 8x8 patches), `np.linspace(0, side - 1, grid)` lands on exact integers,
    so `h_frac`/`w_frac` are zero for every one of the 64 tokens and the
    four-corner bilinear weighted sum in `_pos_embed_table` degenerates to a
    single-index lookup (`[1, 0, 0, 0]` corner weights). A bug in the weight
    formula itself -- swapped `h_frac`/`w_frac`, wrong corner pairing, a sign
    error, wrong summation order -- would then be multiplied by zero and
    vanish, passing the encoder differential at full precision while proving
    nothing about the interpolation. This asserts the non-degenerate case
    directly, with the actual fractional values, rather than assuming it.
    """
    z = np.load(VISION_NPZ)
    side = int(z["param::pos_embed.weight"].shape[0] ** 0.5)
    grid = TINY_CFG.grid
    assert side != grid, (
        f"learned grid side {side} equals patch grid {grid}; the bilinear "
        "interpolation is only exercised at its degenerate, zero-fractional "
        "-weight corner -- see the docstring"
    )
    h_grid = np.linspace(0, side - 1, grid)
    h_frac = h_grid - h_grid.astype(np.int64)
    assert np.count_nonzero(h_frac) > 0, (
        f"h_frac is all zero for side={side}, grid={grid}: {h_frac.tolist()} -- "
        "the interpolation math cannot be wrong in this configuration"
    )


def test_patchify_matches_upstreams_own_literal_permute_chain():
    """Compares against `patches_upstream`, not `patches`.

    `z["patches"]` is produced by `gen_vision_oracle.py::_patches_from_image`,
    which is -- deliberately -- the SAME `reshape`/`transpose`/`reshape`
    expression as hexlib's own `structural.py::_patchify_reference`, just
    spelled in numpy instead of being hexlib's op. Comparing hexlib against
    it would be a tautology: a wrong ordering shared by both sides (because
    both are the one expression, copied) cancels and this test would pass
    regardless. It would also not be rescued by the encoder differential in
    `test_encoder_matches_upstream_within_tolerance`, for the same reason --
    that test feeds `patches` (not `patches_upstream`) into both hexlib's
    graph and the upstream model that produced `expected_merged`.

    `z["patches_upstream"]` is different: it is produced by
    `_patches_from_image_via_literal_upstream_chain`, which executes
    `Qwen2VLImageProcessor._preprocess`'s own torch `reshape` -> `permute(0,
    2, 5, 3, 6, 1, 4, 7)` chain (`image_processing_qwen2_vl.py:196-218`)
    verbatim, with upstream's own literal permute indices -- not hexlib's
    transpose indices, hand-translated to drop the batch axis. Comparing
    hexlib's `patchify` op against THAT is a real, independent check on
    merge-block token ordering.
    """
    z = np.load(VISION_NPZ)
    from hexlib.graph.ops import get

    (out,) = get("patchify").reference(
        (z["image"],),
        {
            "patch": TINY_CFG.patch_size,
            "temporal_patch": TINY_CFG.temporal_patch_size,
            "merge": TINY_CFG.spatial_merge_size,
            "grid_h": TINY_CFG.grid,
            "grid_w": TINY_CFG.grid,
        },
    )
    np.testing.assert_allclose(out, z["patches_upstream"], rtol=1e-6, atol=1e-6)


def test_encoder_matches_upstream_within_tolerance():
    z = np.load(VISION_NPZ)
    graph = build_vision_encoder(TINY_CFG)
    assert not isinstance(graph, Err), getattr(graph, "detail", "")

    feeds = _feeds_from_golden(graph, z)
    out = eager.run(graph, feeds)
    assert not isinstance(out, Err), getattr(out, "detail", "")

    (out_name,) = graph.outputs
    got = out[out_name]
    want = z["expected_merged"]
    assert got.shape == want.shape, f"got {got.shape}, upstream produced {want.shape}"
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def test_a_wrong_epsilon_would_be_caught():
    # The near-miss discipline, scaled from a kernel to a graph: the golden must
    # actually discriminate. If this passes with the wrong eps, the tolerance
    # above is too loose to be evidence of anything.
    z = np.load(VISION_NPZ)
    bad = VitConfig(**{**TINY_CFG.__dict__, "layernorm_eps": 1e-1})
    graph = build_vision_encoder(bad)
    out = eager.run(graph, _feeds_from_golden(graph, z))
    (out_name,) = graph.outputs
    assert not np.allclose(out[out_name], z["expected_merged"], rtol=1e-4, atol=1e-5)


def _vision_bilinear_indices_and_weights(
    grid: int, num_grid_per_side: int, merge: int
) -> tuple[np.ndarray, np.ndarray]:
    """Numpy port of `get_vision_bilinear_indices_and_weights`.

    transformers/vision_utils.py:147-217 (single image, grid_thw = [[1, grid,
    grid]], so the `t` loop there runs once and `.repeat(t)` is a no-op).
    Read, not reinvented: the four-corner bilinear indices/weights over the
    learned `side x side` grid, then reordered into merge-block token order
    (the same order `patchify` and `get_vision_position_ids` use) by
    `vision_utils.py:207-209`.
    """
    side = num_grid_per_side
    h = w = grid

    h_grid = np.linspace(0, side - 1, h)
    w_grid = np.linspace(0, side - 1, w)
    h_floor = h_grid.astype(np.int64)
    w_floor = w_grid.astype(np.int64)
    h_ceil = np.minimum(h_floor + 1, side - 1)
    w_ceil = np.minimum(w_floor + 1, side - 1)
    h_frac = h_grid - h_floor
    w_frac = w_grid - w_floor

    h_floor_offset = h_floor * side
    h_ceil_offset = h_ceil * side

    corner_indices = [
        (h_floor_offset[:, None] + w_floor[None, :]).flatten(),
        (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),
        (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),
        (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),
    ]
    corner_weights = [
        ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
        ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
        (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
        (h_frac[:, None] * w_frac[None, :]).flatten(),
    ]

    h_idx = np.arange(h).reshape(h // merge, merge)
    w_idx = np.arange(w).reshape(w // merge, merge)
    base = h_idx[:, :, None, None] * w + w_idx[None, None, :, :]
    reorder = base.transpose(0, 2, 1, 3).flatten()

    indices = np.stack([c[reorder] for c in corner_indices])
    weights = np.stack([c[reorder] for c in corner_weights])
    return indices, weights


def _vision_position_ids(grid: int, merge: int) -> np.ndarray:
    """Numpy port of `get_vision_position_ids`, transformers/vision_utils.py:76-81.

    Single image, grid_thw = [[1, grid, grid]] -- the `t` loop runs once and
    `.repeat(t)` is a no-op.
    """
    h = w = grid
    hpos_ids = np.broadcast_to(np.arange(h)[:, None], (h, w))
    hpos_ids = hpos_ids.reshape(h // merge, merge, w // merge, merge).transpose(0, 2, 1, 3).flatten()

    wpos_ids = np.broadcast_to(np.arange(w)[None, :], (h, w))
    wpos_ids = wpos_ids.reshape(h // merge, merge, w // merge, merge).transpose(0, 2, 1, 3).flatten()

    return np.stack([hpos_ids, wpos_ids], axis=-1)


def _rope_tables(cfg: VitConfig) -> tuple[np.ndarray, np.ndarray]:
    """rope_cos, rope_sin -- modeling_qwen3_5.py:84-92 (inv_freq, rotary forward)
    and :1096-1108 (position ids -> freqs -> cos/sin), reproduced on the host.

    `Qwen3_5VisionRotaryEmbedding(head_dim // 2)`: inv_freq has
    `dim = head_dim // 2` elements halved again by the `arange(0, dim, 2)`
    step, so `inv_freq` has `head_dim // 4` entries. Each token's (row, col)
    position id is multiplied by inv_freq and flattened row-major (row block,
    then col block) to `[n, head_dim // 2]`, then duplicated by
    `cat((freqs, freqs), dim=-1)` to `[n, head_dim]` before cos/sin.
    """
    d = cfg.head_dim
    dim = d // 2
    inv_freq = 1.0 / (cfg.rope_theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))

    position_ids = _vision_position_ids(cfg.grid, cfg.spatial_merge_size).astype(np.float64)
    freqs = position_ids[:, :, None] * inv_freq[None, None, :]     # [n, 2, dim//2]
    n = position_ids.shape[0]
    flat = freqs.reshape(n, -1)                                    # [n, head_dim//2]
    emb = np.concatenate([flat, flat], axis=-1)                    # [n, head_dim]
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def _pos_embed_table(cfg: VitConfig, table: np.ndarray) -> np.ndarray:
    """pos_embed -- modeling_qwen3_5.py:1090, :1100 and
    `get_vision_bilinear_indices_and_weights` (transformers/vision_utils.py:147-217),
    reproduced on the host: bilinearly resample the learned
    `num_grid_per_side x num_grid_per_side` grid onto this config's patch grid,
    in merge-block token order.

    `num_grid_per_side` is derived from `table.shape[0]` (the golden's own
    `pos_embed.weight` row count), not hardcoded, exactly as
    `self.num_grid_per_side = int(config.num_position_embeddings**0.5)`
    (modeling_qwen3_5.py:1039) computes it from the config. A hardcoded
    constant here would let a future regeneration with a different learned
    grid size leave `side` silently wrong while `table`'s shape still passes
    every shape check -- shape-correct but numerically wrong, the exact
    failure class this oracle otherwise eliminates.
    """
    side = int(table.shape[0] ** 0.5)
    if side * side != table.shape[0]:
        raise ValueError(
            f"pos_embed table has {table.shape[0]} rows, which is not a perfect "
            "square; num_grid_per_side is undefined"
        )
    indices, weights = _vision_bilinear_indices_and_weights(cfg.grid, side, cfg.spatial_merge_size)
    out = np.zeros((indices.shape[1], table.shape[1]), dtype=np.float64)
    for i in range(4):
        out += table[indices[i]].astype(np.float64) * weights[i][:, None]
    return out.astype(np.float32)


def _feeds_from_golden(graph, z) -> dict[str, np.ndarray]:
    """Map the upstream parameter names onto hexlib's tensor names.

    Returns a feed for the graph's image input and for every const tensor.
    hexlib's graph starts from the IMAGE and patchifies it itself, so the feed
    is z["image"] -- not z["patches"]. That is deliberate: it puts patchify
    inside the differential rather than beside it.

    Every Linear weight is TRANSPOSED (torch stores [out, in]; hexlib stores
    [k, n]). The fused qkv weight [3H, H] is split into three [H, H] blocks in
    q, k, v order and transposed -- modeling_qwen3_5.py:929 reshapes the
    linear's output to (seq, 3, heads, -1) and unbinds along the `3` axis, so
    contiguous output-row blocks [0:H], [H:2H], [2H:3H] are q, k, v
    respectively; the same split applies to the weight's rows and the bias.
    pos_embed and rope_cos/sin are computed here, on the host, exactly as
    modeling_qwen3_5.py:1090-1108 does -- that is the point of folding them.
    """
    p = lambda name: z[f"param::{name}"]
    H = TINY_CFG.hidden_size

    feeds: dict[str, np.ndarray] = {}
    feeds["image"] = z["image"]

    # Patch embedding: Conv3d weight [embed, C, T, ph, pw] flattens (row-major)
    # to [embed, C*T*ph*pw] -- exactly patchify's feature order -- then
    # transposes to hexlib's [feat, embed].
    conv_w = p("patch_embed.proj.weight")
    feeds["w_patch_embed"] = conv_w.reshape(conv_w.shape[0], -1).T
    feeds["b_patch_embed"] = p("patch_embed.proj.bias")

    feeds["pos_embed"] = _pos_embed_table(TINY_CFG, p("pos_embed.weight"))
    cos, sin = _rope_tables(TINY_CFG)
    feeds["rope_cos"] = cos
    feeds["rope_sin"] = sin

    for layer in range(TINY_CFG.depth):
        b = f"blocks.{layer}"
        blk = f"blk{layer}"

        qkv_w = p(f"{b}.attn.qkv.weight")   # [3H, H]
        qkv_b = p(f"{b}.attn.qkv.bias")     # [3H]
        for i, which in enumerate(("q", "k", "v")):
            feeds[f"{blk}.w{which}"] = qkv_w[i * H:(i + 1) * H, :].T
            feeds[f"{blk}.b{which}"] = qkv_b[i * H:(i + 1) * H]

        feeds[f"{blk}.wo"] = p(f"{b}.attn.proj.weight").T
        feeds[f"{blk}.bo"] = p(f"{b}.attn.proj.bias")

        feeds[f"{blk}.w_fc1"] = p(f"{b}.mlp.linear_fc1.weight").T
        feeds[f"{blk}.b_fc1"] = p(f"{b}.mlp.linear_fc1.bias")
        feeds[f"{blk}.w_fc2"] = p(f"{b}.mlp.linear_fc2.weight").T
        feeds[f"{blk}.b_fc2"] = p(f"{b}.mlp.linear_fc2.bias")

        feeds[f"{blk}.ln1_w"] = p(f"{b}.norm1.weight")
        feeds[f"{blk}.ln1_b"] = p(f"{b}.norm1.bias")
        feeds[f"{blk}.ln2_w"] = p(f"{b}.norm2.weight")
        feeds[f"{blk}.ln2_b"] = p(f"{b}.norm2.bias")

    feeds["m_ln_w"] = p("merger.norm.weight")
    feeds["m_ln_b"] = p("merger.norm.bias")
    feeds["w_m1"] = p("merger.linear_fc1.weight").T
    feeds["b_m1"] = p("merger.linear_fc1.bias")
    feeds["w_m2"] = p("merger.linear_fc2.weight").T
    feeds["b_m2"] = p("merger.linear_fc2.bias")

    const_names = {t.name for t in graph.tensors.values() if t.const}
    missing = const_names - set(feeds)
    assert not missing, f"_feeds_from_golden supplied no feed for: {sorted(missing)}"
    extra = set(feeds) - const_names - set(graph.inputs)
    assert not extra, f"_feeds_from_golden supplied a feed for non-const tensors: {sorted(extra)}"

    for name, array in feeds.items():
        want_shape = graph.tensor(name).shape
        got_shape = tuple(np.asarray(array).shape)
        assert got_shape == want_shape, (
            f"feed {name!r} has shape {got_shape}, but the graph declares {want_shape} "
            "-- a shape mismatch here is almost always a forgotten transpose"
        )

    return feeds

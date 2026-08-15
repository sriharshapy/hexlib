"""Checkpoint weights -> hexlib graph consts, for the Qwen3.5 vision tower.

THIS MAPPING WAS WRITTEN FOR A TEST AND BELONGS IN THE LIBRARY. It lived in
`hexlib/tests/test_vision_oracle.py` as `_feeds_from_golden`, keyed to that
file's TINY_CFG and to an npz of random parameters. Every part of it -- the
Conv3d flatten, the fused-qkv split, each transpose, the two folded tables --
is exactly what a real checkpoint needs too, and a second copy written for the
checkpoint would be a second chance to get a transpose wrong.

WHAT IS NOT A COPY. `pos_embed` and `rope_cos`/`rope_sin` are DERIVED, not read.
Upstream computes the positional embedding at run time as a bilinear resample of
a learned grid, and the rotary tables from position ids; hexlib folds both to
constants because the resolution is fixed at compile time. A loader that matched
parameter names and copied tensors would produce a graph that is complete,
shape-correct, and wrong in three of its 203 consts.

THE TRANSPOSES ARE THE DANGEROUS PART. torch `nn.Linear` stores [out, in] and
hexlib wants [in, out]; the fused qkv is [3H, H] and splits into three [H, H]
blocks in q, k, v order (modeling_qwen3_5.py:929 reshapes to (seq, 3, heads, -1)
and unbinds along the `3` axis). At hidden_size 768 every one of those is
square, so a forgotten transpose passes every shape check this file makes and
produces a correctly-shaped wrong answer.
"""
from __future__ import annotations

import numpy as np

from hexlib.models.vit import VitConfig


def vision_bilinear_indices_and_weights(
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


def vision_position_ids(grid: int, merge: int) -> np.ndarray:
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


def rope_tables(cfg: VitConfig) -> tuple[np.ndarray, np.ndarray]:
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

    position_ids = vision_position_ids(cfg.grid, cfg.spatial_merge_size).astype(np.float64)
    freqs = position_ids[:, :, None] * inv_freq[None, None, :]     # [n, 2, dim//2]
    n = position_ids.shape[0]
    flat = freqs.reshape(n, -1)                                    # [n, head_dim//2]
    emb = np.concatenate([flat, flat], axis=-1)                    # [n, head_dim]
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def pos_embed_table(cfg: VitConfig, table: np.ndarray) -> np.ndarray:
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
    indices, weights = vision_bilinear_indices_and_weights(cfg.grid, side, cfg.spatial_merge_size)
    out = np.zeros((indices.shape[1], table.shape[1]), dtype=np.float64)
    for i in range(4):
        out += table[indices[i]].astype(np.float64) * weights[i][:, None]
    return out.astype(np.float32)


def feeds_from_params(graph, cfg: VitConfig, get) -> dict[str, np.ndarray]:
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
    p = get
    H = cfg.hidden_size

    feeds: dict[str, np.ndarray] = {}

    # Patch embedding: Conv3d weight [embed, C, T, ph, pw] flattens (row-major)
    # to [embed, C*T*ph*pw] -- exactly patchify's feature order -- then
    # transposes to hexlib's [feat, embed].
    conv_w = p("patch_embed.proj.weight")
    feeds["w_patch_embed"] = conv_w.reshape(conv_w.shape[0], -1).T
    feeds["b_patch_embed"] = p("patch_embed.proj.bias")

    feeds["pos_embed"] = pos_embed_table(cfg, p("pos_embed.weight"))
    cos, sin = rope_tables(cfg)
    feeds["rope_cos"] = cos
    feeds["rope_sin"] = sin

    for layer in range(cfg.depth):
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
    assert not missing, f"feeds_from_params supplied no feed for: {sorted(missing)}"
    extra = set(feeds) - const_names
    assert not extra, f"feeds_from_params supplied a feed for non-const tensors: {sorted(extra)}"

    for name, array in feeds.items():
        want_shape = graph.tensor(name).shape
        got_shape = tuple(np.asarray(array).shape)
        assert got_shape == want_shape, (
            f"feed {name!r} has shape {got_shape}, but the graph declares {want_shape} "
            "-- a shape mismatch here is almost always a forgotten transpose"
        )

    return feeds

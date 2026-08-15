"""The M0 gate: hexlib's eager encoder must match the upstream implementation.

Reads committed golden vectors. Imports neither torch nor transformers, so it
runs everywhere and can never be skipped into a false pass.
"""
from __future__ import annotations

import os

import numpy as np

import hexlib.graph.opdefs  # noqa: F401
from hexlib.graph import eager
from hexlib.models import loader
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


# THE MAPPING NOW LIVES IN hexlib/models/loader.py. It was written here, but
# every part of it -- the Conv3d flatten, the fused-qkv split, each transpose,
# the two folded tables -- is exactly what a real checkpoint needs, and a second
# copy written for the checkpoint would be a second chance to get a transpose
# wrong. These aliases keep this file's tests reading as they did; the code they
# call is the same code the checkpoint loader calls, which is the point.
_vision_bilinear_indices_and_weights = loader.vision_bilinear_indices_and_weights
_vision_position_ids = loader.vision_position_ids
_rope_tables = loader.rope_tables
_pos_embed_table = loader.pos_embed_table


def _feeds_from_golden(graph, z) -> dict[str, np.ndarray]:
    """The golden npz, through the library's mapping.

    `image` is a graph INPUT rather than a const, so it is supplied here and not
    by `feeds_from_params`, which knows only about weights."""
    feeds = loader.feeds_from_params(graph, TINY_CFG,
                                     lambda name: z[f"param::{name}"])
    feeds["image"] = z["image"]
    return feeds



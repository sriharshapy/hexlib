"""Generate the committed golden vectors for the vision encoder oracle.

Run once, by hand, with the `oracle` extra installed:

    pip install -e ".[oracle]"
    python scripts/gen_vision_oracle.py

It writes hexlib/tests/data/*.npz, which are committed. No test imports torch
or transformers; they read the committed arrays instead. That keeps the
differential test unconditional rather than skipped-when-absent.

The weights are random and are written into the output file, so the golden is
reproducible from the file alone and does not depend on a checkpoint download,
on torch's initialisation, or on the transformers version that produced it.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "hexlib", "tests", "data")

# Deliberately tiny. This validates the ARCHITECTURE, not the checkpoint.
TINY = dict(
    depth=2,
    hidden_size=64,
    num_heads=4,
    intermediate_size=128,
    patch_size=4,
    temporal_patch_size=2,
    in_channels=3,
    spatial_merge_size=2,
    out_hidden_size=32,
    # Deliberately mismatched with GRID (8x8 patches): a 5x5 learned grid
    # forces the bilinear resample in `pos_embed` to hit non-trivial
    # fractional weights on every corner. An 8x8-vs-8x8 grid (the previous
    # value here) makes np.linspace(0, 7, 8) land on exact integers, so
    # h_frac/w_frac are zero everywhere and the four-corner weighted sum
    # degenerates to a single-index lookup -- a bug in the weight formula
    # itself (swapped h_frac/w_frac, wrong corner pairing, a sign error)
    # would be multiplied by zero and vanish. See docs/research/
    # oracle-provenance.md for the h_frac evidence this was checked against.
    num_position_embeddings=25,     # a 5x5 learned grid, mismatched on purpose
    hidden_act="gelu_pytorch_tanh",
)
GRID = 8                             # 8x8 patches -> 64 tokens -> 16 merged


def main() -> int:
    os.makedirs(DATA, exist_ok=True)
    rng = np.random.RandomState(20260809)
    torch.manual_seed(0)

    config = Qwen3_5VisionConfig(**TINY)
    model = Qwen3_5VisionModel(config).eval()

    # Overwrite every parameter deterministically from numpy, so the golden does
    # not depend on torch's initialisation scheme or version.
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            values = rng.standard_normal(tuple(param.shape)).astype(np.float32) * 0.05
            param.copy_(torch.from_numpy(values))

    # ONE random image is the source of truth for both halves of the gate.
    # hexlib's graph starts from an image and patchifies it; the upstream model
    # starts from already-patchified tokens. Deriving `patches` FROM `image`
    # using upstream's own reshape is what connects them -- without it, the
    # patchify golden and the encoder golden would be two unrelated facts and
    # nothing would prove hexlib's patchify feeds the encoder correctly.
    image = rng.standard_normal(
        (
            TINY["in_channels"],
            TINY["temporal_patch_size"],
            GRID * TINY["patch_size"],
            GRID * TINY["patch_size"],
        )
    ).astype(np.float32)
    patches = _patches_from_image(image)          # see below -- READ, do not guess
    grid_thw = torch.tensor([[1, GRID, GRID]], dtype=torch.long)

    with torch.no_grad():
        result = model(torch.from_numpy(patches), grid_thw=grid_thw)

    arrays = {
        "image": image,
        "patches": patches,
        "expected_merged": result.pooler_output.numpy().astype(np.float32),
        "expected_last_hidden": result.last_hidden_state.numpy().astype(np.float32),
    }
    for name, param in sorted(model.named_parameters()):
        arrays[f"param::{name}"] = param.detach().numpy().astype(np.float32)
    for name, buf in sorted(model.named_buffers()):
        arrays[f"buffer::{name}"] = buf.detach().numpy().astype(np.float32)

    out = os.path.join(DATA, "qwen35_vision_tiny.npz")
    np.savez_compressed(out, **arrays)
    print(f"wrote {out} ({os.path.getsize(out)} bytes, {len(arrays)} arrays)")

    return 0


def _patches_from_image(image: np.ndarray) -> np.ndarray:
    """[C, T, H, W] -> [n_patches, C*T*patch*patch], using UPSTREAM's own ordering.

    `qwen3_5` declares no `image_processing_*.py` of its own -- confirmed by:

        python -c "import transformers, os; d=os.path.dirname(transformers.__file__); print(sorted(os.listdir(os.path.join(d,'models','qwen3_5'))))"
        -> ['__init__.py', '__pycache__', 'configuration_qwen3_5.py',
            'modeling_qwen3_5.py', 'modular_qwen3_5.py', 'tokenization_qwen3_5.py']

    `transformers/models/auto/image_processing_auto.py:128` maps
    `"qwen3_5"` to `Qwen2VLImageProcessor` (the torchvision-backed one, i.e.
    `transformers/models/qwen2_vl/image_processing_qwen2_vl.py`, NOT the
    `_pil` variant) -- so unlike the RoPE/position-id code, which qwen3_5
    reimplements itself in `vision_utils.py`, the patch-flattening ordering
    genuinely *is* inherited from Qwen2-VL here; that is upstream's own
    declared mapping, not a guess from a sibling model.

    The reshape/transpose/flatten chain is
    `image_processing_qwen2_vl.py:196-218` (`Qwen2VLImageProcessor._preprocess`):

        patches = patches.reshape(batch, channel, grid_h//merge, merge, patch,
                                   grid_w//merge, merge, patch)
        patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
        # [batch, grid_h/merge, grid_w/merge, merge, merge, channel, patch, patch]
        flatten_patches = (patches.unsqueeze(6)
                                   .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
                                   .reshape(batch, grid_h*grid_w,
                                            channel*temporal_patch_size*patch*patch))

    Token order after the permute is (grid_h/merge, grid_w/merge, merge_h,
    merge_w) -- merge-block order, matching `get_vision_position_ids`
    (`transformers/vision_utils.py:76-81`) and hexlib's own `patchify`
    (`hexlib/graph/opdefs/structural.py::_patchify_reference`). Feature order
    is (channel, T, patch_h, patch_w), matching the Conv3d weight layout
    `[embed, C, T, ph, pw]` at `modeling_qwen3_5.py:864`.

    The `.unsqueeze(6).expand(..., temporal_patch_size, ...)` step exists only
    because the processor's real input is a single still image with no T axis
    of its own -- it duplicates that one frame across every temporal slot.
    Our synthetic `image` already carries a genuine, independently-random T
    axis of size `temporal_patch_size` (that is the whole point of feeding two
    distinct frames rather than one broadcast frame), so the equivalent step
    here is a direct reshape over the real T axis instead of a broadcast --
    the token/feature ORDERING is identical either way; only the source of
    the T-axis values differs.

    Conclusion: this ordering agrees exactly with hexlib's `patchify` op
    (verified independently here, not by importing it) -- no disagreement to
    report.
    """
    c, t, h, w = image.shape
    patch = TINY["patch_size"]
    merge = TINY["spatial_merge_size"]
    grid_h, grid_w = h // patch, w // patch

    x = image.reshape(c, t, grid_h // merge, merge, patch, grid_w // merge, merge, patch)
    # Token axes in merge-block order: bh, bw, mh, mw. Feature axes: c, t, ph, pw.
    x = x.transpose(2, 5, 3, 6, 0, 1, 4, 7)
    return x.reshape(grid_h * grid_w, c * t * patch * patch)


if __name__ == "__main__":
    sys.exit(main())

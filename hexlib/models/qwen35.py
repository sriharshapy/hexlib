"""Qwen3.5-0.8B vision tower constants.

Every value cites where it was read from. Two of them do NOT appear in
config.json and would be silently wrong if only the config were consulted:
`layernorm_eps` and `rope_theta` are literals in the model code. That is the
"architecture facts that live in code, not config" risk in the spec.

Full record: docs/research/qwen35-vision-architecture.md
"""
from __future__ import annotations

from hexlib.models.vit import VitConfig

# All from Qwen/Qwen3.5-0.8B config.json, `vision_config`, except where noted.
QWEN35_08B_VISION = VitConfig(
    depth=12,                    # config.json vision_config.depth
    hidden_size=768,             # config.json vision_config.hidden_size
    num_heads=12,                # config.json vision_config.num_heads -> head_dim 64
    intermediate_size=3072,      # config.json vision_config.intermediate_size
    patch_size=16,               # config.json vision_config.patch_size
    temporal_patch_size=2,       # config.json vision_config.temporal_patch_size
    in_channels=3,               # config.json vision_config.in_channels
    spatial_merge_size=2,        # config.json vision_config.spatial_merge_size
    out_hidden_size=1024,        # config.json vision_config.out_hidden_size
    # NOT in config.json. nn.LayerNorm(config.hidden_size, eps=1e-6) at
    # modeling_qwen3_5.py:991-992 (blocks) and :880 (merger). The vision tower
    # uses LayerNorm; the TEXT tower uses RMSNorm with rms_norm_eps. Do not
    # conflate them.
    layernorm_eps=1e-6,
    # NOT in config.json. Qwen3_5VisionRotaryEmbedding.__init__ default
    # theta=10000.0, modeling_qwen3_5.py:84.
    rope_theta=10000.0,
    # 256x256 is the model's own minimum (shortest_edge 65536 pixels) and the
    # only common resolution whose patch count is divisible by 32, which both
    # the q4_0 block size and the HMX fp16 tile want. 448^2 gives 784 patches
    # = 24*32 + 16 and needs padding. Spec 4.3, 5.2.
    image_size=256,
    # Spec 5.1: block-quantized repacked weights are the path ggml-hexagon's
    # dedicated HVX/HMX matmul kernels are built for; activations stay fp16.
    act_dtype="fp16",
    weight_dtype="q4_0",
)


def qwen35_at(image_size: int) -> VitConfig:
    """The same tower at a different square resolution.

    Resolution is a config parameter by construction (spec 5.2): the model
    accepts 256^2 up to 4096^2. Anything whose patch grid is not divisible by
    spatial_merge_size is rejected by build_vision_encoder.
    """
    return VitConfig(**{**QWEN35_08B_VISION.__dict__, "image_size": image_size})


def qwen35_oracle(image_size: int = 256) -> VitConfig:
    """The same tower in fp32, for running against the eager oracle."""
    return VitConfig(
        **{
            **QWEN35_08B_VISION.__dict__,
            "image_size": image_size,
            "act_dtype": "fp32",
            "weight_dtype": "fp32",
        }
    )

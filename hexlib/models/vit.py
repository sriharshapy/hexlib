"""Builds a vision-encoder graph from a config.

Everything that depends only on the config -- the interpolated position
embedding, the RoPE cos/sin tables, the QKV split, the weight transposes --
is folded here, at build time. Resolution is fixed by the config, so all of
it is constant, and none of it costs an op at run time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hexlib.graph.ir import Graph, Op, Tensor
from hexlib.result import Err


@dataclass(frozen=True)
class VitConfig:
    depth: int
    hidden_size: int
    num_heads: int
    intermediate_size: int
    patch_size: int
    temporal_patch_size: int
    in_channels: int
    spatial_merge_size: int
    out_hidden_size: int
    layernorm_eps: float
    rope_theta: float
    image_size: int
    act_dtype: str = "fp32"
    weight_dtype: str = "fp32"

    @property
    def grid(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_tokens(self) -> int:
        return self.grid * self.grid

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def merge_unit(self) -> int:
        return self.spatial_merge_size * self.spatial_merge_size


class _Builder:
    """Accumulates tensors and ops. Names are assigned once, never reused."""

    def __init__(self, cfg: VitConfig) -> None:
        self.cfg = cfg
        self.tensors: dict[str, Tensor] = {}
        self.ops: list[Op] = []
        self._next_id = 0

    def declare(self, name: str, shape: tuple[int, ...], dtype: str, const: bool = False) -> str:
        if name in self.tensors:
            raise ValueError(f"tensor {name!r} declared twice")
        self.tensors[name] = Tensor(name=name, dtype=dtype, shape=shape, const=const)
        return name

    def const(self, name: str, shape: tuple[int, ...], dtype: str | None = None) -> str:
        return self.declare(name, shape, dtype or self.cfg.weight_dtype, const=True)

    def emit(
        self,
        kind: str,
        inputs: tuple[str, ...],
        out_name: str,
        out_shape: tuple[int, ...],
        attrs: dict[str, Any] | None = None,
        out_dtype: str | None = None,
    ) -> str:
        self.declare(out_name, out_shape, out_dtype or self.cfg.act_dtype)
        self.ops.append(
            Op(id=self._next_id, kind=kind, inputs=inputs, outputs=(out_name,), attrs=attrs or {})
        )
        self._next_id += 1
        return out_name

    def linear(self, x: str, wname: str, bname: str, k: int, n: int, out: str) -> str:
        """matmul + bias add. Weight is [k, n]; bias is fp32 (spec 5.1).

        `wname`/`bname` are the exact const-tensor names (matching the
        graph-structure pseudocode's own variable names, e.g. wq/bq,
        w_fc1/b_fc1, w_m1/b_m1), not derived from `out`.
        """
        w = self.const(wname, (k, n))
        b = self.const(bname, (n,), dtype="fp32")
        rows = self.tensors[x].shape[:-1]
        h = self.emit("matmul", (x, w), f"{out}.mm", rows + (n,))
        return self.emit("add", (h, b), out, rows + (n,))


def weight_names(cfg: VitConfig) -> tuple[str, ...]:
    """Every const tensor the graph needs, in a stable order."""
    graph = build_vision_encoder(cfg)
    if isinstance(graph, Err):
        return ()
    return tuple(sorted(t.name for t in graph.tensors.values() if t.const))


def build_vision_encoder(cfg: VitConfig) -> Graph | Err:
    problems = _config_problems(cfg)
    if problems:
        return Err("invalid vision config", "\n".join(problems))

    b = _Builder(cfg)
    n = cfg.num_tokens
    H = cfg.hidden_size
    heads, d = cfg.num_heads, cfg.head_dim
    merge = cfg.spatial_merge_size
    feat = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size

    image = b.declare(
        "image",
        (cfg.in_channels, cfg.temporal_patch_size, cfg.image_size, cfg.image_size),
        "fp32",
    )
    patches = b.emit(
        "patchify",
        (image,),
        "patches",
        (n, feat),
        {
            "patch": cfg.patch_size,
            "temporal_patch": cfg.temporal_patch_size,
            "merge": merge,
            "grid_h": cfg.grid,
            "grid_w": cfg.grid,
        },
        out_dtype="fp32",
    )
    embedded = b.linear(patches, "w_patch_embed", "b_patch_embed", feat, H, "patch_embed.out")
    # The learned 48x48 position grid is bilinearly resampled to this image's
    # patch grid (modeling_qwen3_5.py:1100). Resolution is fixed, so the result
    # is a constant and the interpolation costs no ops here.
    pos = b.const("pos_embed", (n, H), dtype="fp32")
    x = b.emit("add", (embedded, pos), "x0", (n, H))

    # 2D RoPE tables, also constant for a fixed grid. Applied to Q and K inside
    # every block (modeling_qwen3_5.py:932), on top of the absolute pos_embed.
    cos = b.const("rope_cos", (n, d), dtype="fp32")
    sin = b.const("rope_sin", (n, d), dtype="fp32")

    for layer in range(cfg.depth):
        p = f"blk{layer}"
        # No standalone pre-norm exists before block 0: each block is internally
        # pre-norm (modeling_qwen3_5.py:1009-1015).
        a = _layernorm(b, x, f"{p}.ln1_w", f"{p}.ln1_b", f"{p}.ln1.out", (n, H),
                       cfg.layernorm_eps)

        heads_qkv = {}
        for which in ("q", "k", "v"):
            proj = b.linear(a, f"{p}.w{which}", f"{p}.b{which}", H, H, f"{p}.{which}.out")
            shaped = b.emit("reshape", (proj,), f"{p}.{which}.h", (n, heads, d),
                            {"shape": (n, heads, d)})
            if which in ("q", "k"):
                shaped = b.emit("rope_2d", (shaped, cos, sin), f"{p}.{which}.rope",
                                (n, heads, d))
            heads_qkv[which] = b.emit("transpose", (shaped,), f"{p}.{which}.t",
                                      (heads, n, d), {"perm": (1, 0, 2)})

        qs = b.emit("scale", (heads_qkv["q"],), f"{p}.q.scaled", (heads, n, d),
                    {"factor": float(d) ** -0.5})
        ktt = b.emit("transpose", (heads_qkv["k"],), f"{p}.k.tt", (heads, d, n),
                     {"perm": (0, 2, 1)})
        scores = b.emit("matmul", (qs, ktt), f"{p}.scores", (heads, n, n))
        # Full, non-causal attention: no mask at all (modeling_qwen3_5.py:917, :973).
        probs = b.emit("softmax", (scores,), f"{p}.probs", (heads, n, n), {"axis": -1})
        ctx = b.emit("matmul", (probs, heads_qkv["v"]), f"{p}.ctx", (heads, n, d))
        ctxt = b.emit("transpose", (ctx,), f"{p}.ctx.t", (n, heads, d), {"perm": (1, 0, 2)})
        ctxf = b.emit("reshape", (ctxt,), f"{p}.ctx.f", (n, H), {"shape": (n, H)})
        proj = b.linear(ctxf, f"{p}.wo", f"{p}.bo", H, H, f"{p}.o.out")
        x_attn = b.emit("add", (x, proj), f"{p}.resid1", (n, H))

        c = _layernorm(b, x_attn, f"{p}.ln2_w", f"{p}.ln2_b", f"{p}.ln2.out", (n, H),
                       cfg.layernorm_eps)
        f1 = b.linear(c, f"{p}.w_fc1", f"{p}.b_fc1", H, cfg.intermediate_size, f"{p}.fc1.out")
        g = b.emit("gelu_tanh", (f1,), f"{p}.act", (n, cfg.intermediate_size))
        f2 = b.linear(g, f"{p}.w_fc2", f"{p}.b_fc2", cfg.intermediate_size, H, f"{p}.fc2.out")
        x = b.emit("add", (x_attn, f2), f"{p}.resid2", (n, H))

    # The merger's own LayerNorm is the de facto final norm; there is no
    # separate post-norm tensor (modeling_qwen3_5.py:880, :886).
    M = H * cfg.merge_unit
    mn = _layernorm(b, x, "m_ln_w", "m_ln_b", "merger.ln.out", (n, H), cfg.layernorm_eps)
    # A pure reshape, correct only because patchify emitted merge-block order.
    mc = b.emit("reshape", (mn,), "merger.concat", (n // cfg.merge_unit, M),
                {"shape": (n // cfg.merge_unit, M)})
    m1 = b.linear(mc, "w_m1", "b_m1", M, M, "merger.fc1.out")
    # Plain nn.GELU() -- the exact erf form, NOT the blocks' tanh approximation
    # (modeling_qwen3_5.py:882).
    mg = b.emit("gelu_erf", (m1,), "merger.act", (n // cfg.merge_unit, M))
    out = b.linear(mg, "w_m2", "b_m2", M, cfg.out_hidden_size, "merger.out")

    return Graph(tensors=b.tensors, ops=tuple(b.ops), inputs=(image,), outputs=(out,))


def _layernorm(
    b: _Builder, x: str, wname: str, bname: str, out_name: str,
    shape: tuple[int, ...], eps: float,
) -> str:
    """LayerNorm with a learned weight AND bias. eps is 1e-6, hardcoded in the
    model code and absent from config.json (modeling_qwen3_5.py:991-992)."""
    w = b.const(wname, (shape[-1],), dtype="fp32")
    bias = b.const(bname, (shape[-1],), dtype="fp32")
    return b.emit("layernorm", (x, w, bias), out_name, shape, {"eps": eps})


def _config_problems(cfg: VitConfig) -> list[str]:
    problems: list[str] = []
    # Guard every divisor before it is used in a %, so a zero or negative
    # field is a reported Err, never a raw ZeroDivisionError. `build_vision_encoder`
    # must never raise on a bad config.
    if cfg.patch_size <= 0:
        problems.append(f"patch_size {cfg.patch_size} must be positive")
        return problems
    if cfg.image_size % cfg.patch_size:
        problems.append(
            f"image_size {cfg.image_size} is not divisible by patch_size {cfg.patch_size}"
        )
        return problems
    if cfg.spatial_merge_size <= 0:
        problems.append(f"spatial_merge_size {cfg.spatial_merge_size} must be positive")
    elif cfg.grid % cfg.spatial_merge_size:
        problems.append(
            f"patch grid {cfg.grid}x{cfg.grid} is not divisible by spatial_merge_size "
            f"{cfg.spatial_merge_size}; the merger is a pure reshape and needs whole "
            "merge blocks"
        )
    if cfg.num_heads <= 0:
        problems.append(f"num_heads {cfg.num_heads} must be positive")
    elif cfg.hidden_size % cfg.num_heads:
        problems.append(
            f"hidden_size {cfg.hidden_size} is not divisible by num_heads {cfg.num_heads}"
        )
    elif cfg.head_dim % 2:
        problems.append(f"head_dim {cfg.head_dim} must be even for 2D RoPE")
    if cfg.depth < 1:
        problems.append(f"depth {cfg.depth} must be at least 1")
    return problems

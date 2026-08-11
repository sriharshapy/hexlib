# Roadmap

The op backlog, with status and tier, so tier-0 work (docs, scalar baselines, test
vectors, `spec.json` definitions — no SDK needed) is always visibly available. See
`CONTRIBUTING.md` for what each tier means and what the six gates are.

Each op below is a checkpoint on a different subsystem: reduction, competitive
comparison against production code, transcendental, elementwise with position state,
HMX, and finally the full composition. They are listed in the order they are meant to
be attempted, not by difficulty.

| op | status | tier needed to advance it | subsystem checkpoint |
|---|---|---|---|
| `rmsnorm_fp16` | **done** — gates 1-5 cleared, gate 6 (silicon) pending the silicon-path plan | 1 (kernel exists; tier-0 work: additional near-misses, wider shape coverage in `spec.json`) | reduction |
| `rmsnorm_f32` | not started | 0 (baseline, spec, test vectors) then 1 (kernel) | competitive comparison |
| `softmax_fp16` | not started | 0 then 1 | transcendental |
| `rope_fp16` | not started | 0 then 1 | elementwise with position state |
| `matmul_i8_hmx` | not started | 0 then 1 (needs `caps: ["hmx"]`) | HMX |
| `flash_attn_fp16` | not started | 0 then 1 | full composition |

## Notes per op

**`rmsnorm_fp16` (done).** The first kernel through all five simulation-path gates.
Bake-off in `kernels/rmsnorm_fp16/BAKEOFF.md`: an adapted v6 `rmsnorm_gain_fp16` won
at **2231** kernel cycles (**31.13x** over a 69443-cycle scalar baseline), beating an
adapted v6 `fp16_rmsnorm` (10498 cycles) by **4.71x**. ggml-hexagon's `hvx-norm.h` was
not evaluated against it — see below.

*(Corrected 2026-08-11: this paragraph read 2021 cycles / 34.36x / 5.19x, which are the
pre-fix numbers. `BAKEOFF.md`'s "Fix round 2" moved the winner 2021 → 2231 when an
out-of-bounds read was fixed, and `kernels/rmsnorm_fp16/RESULT.md` — the tool-generated
record — has said 2231 since. README.md carried the same stale figures.)*

**`rmsnorm_f32` (next).** The first direct head-to-head against production
ggml-hexagon code. `hvx_fast_rms_norm_mul_f32` (`include/hexlib/hvx/hvx-norm.h`) is
an exact semantic match for RMSNorm-with-per-column-gain and is fp32-only, so it could
not be baked off fairly against `rmsnorm_fp16`'s fp16 candidates — comparing an fp32
implementation to fp16 ones would measure the dtype, not the implementation. An
`rmsnorm_f32` kernel on identical fp32 shapes settles that comparison honestly.

**`softmax_fp16`.** First transcendental op on the roadmap — exercises the vendored
`hvx-exp.h` reduction-and-normalize pattern rather than `hvx-norm.h`'s reduce-and-scale
one.

**`rope_fp16`.** First op with per-position state (rotation angle depends on sequence
position, not just on the row/column shape), rather than a pure per-row or per-column
reduction.

**`matmul_i8_hmx`.** First kernel requiring HMX (`caps: ["hmx"]`), and the first
integer kernel on this roadmap, so it is bit-exact rather than tolerance-compared (see
`CONTRIBUTING.md`). Read `docs/hardware/hmx-int8.md` before attempting this one: the
HMX int8 tile engine's readout applies a non-unit fp8-like scale, so an
exact-int32-out matmul is not directly representable through it, and this needs to be
a *quantizing* kernel design, not a literal int8 GEMM.

**`flash_attn_fp16`.** The full composition — reduction, transcendental, and
elementwise state together, on the hardest and highest-value kernel. Deliberately
last: making it first would conflate "does the pipeline work" with "is the hardest
kernel correct."

## The vision encoder's op kinds

`hexlib plan qwen35 --print` compiles the Qwen3.5-0.8B vision encoder (M1) to a plan
and lists every op kind with `kernel: None` in its `unimplemented` field — that list
is this table. Unlike the backlog above, these are graph-IR op *kinds* (one entry
covers every op of that kind across all 12 blocks), not individual promoted kernels;
a kind needs at least one kernel clearing the six gates in `CONTRIBUTING.md` before it
can leave this list. Ordered by `predicted_bytes_moved` from the bake-off
(`python -m pytest hexlib/tests/test_policy_bakeoff.py -q -s`, `qwen35@256`,
`min_peak`/`largest_first`, the shipped default as of the whole-branch review that
also made the plan move the image in and the encoder's output back out -- see the
fix report below) — the highest-value kernel to write is at the top, because it is the
one moving the most DDR traffic and therefore the one most likely to be memory-bound
in practice. Kinds tied at zero bytes moved (their inputs are already VTCM-resident;
they cost compute cycles, not DMA) are broken by step count, descending.

**Status column updated 2026-08-11.** `OpDef.kernel` is still `None` for every kind —
wiring the registry to the kernel directories is a separate change, tracked in
`docs/STATE.md`'s open items, because it moves figures several tests pin. So
`Plan.unimplemented` still lists all eleven. The column below reports what actually
*exists and runs*, which is the more useful fact:

| op kind | steps | predicted bytes moved | status | related backlog kernel |
|---|---|---|---|---|
| `matmul_epilogue` | 75 | 55,999,488 | **no kernel** — next, and highest value | fused matmul+bias+activation; needs HMX and q4_0 |
| `patchify` | 1 | 1,572,864 | **no kernel** | runs once, at the input. Must emit merge-block order, not raster |
| `add` | 25 | 786,432 | ✅ `add_fp16`, gated, dispatchable | residual add, elementwise |
| `layernorm` | 25 | 153,600 | ⚠️ `layernorm_fp16` gated but **not dispatchable** (no `RunnerSpec`) | reduction, adjacent to `rmsnorm_fp16`/`rmsnorm_f32` |
| `rope_2d` | 24 | 131,072 | **no kernel** | 2D variant of `rope_fp16` |
| `transpose` | 60 | 0 | ⚠️ `transpose_th_fp16` covers perm (1,0,2) — 48 of 60 steps. perm (0,2,1) has no kernel | layout op, no DDR traffic once resident |
| `reshape` | 49 | 0 | ✅ needs no kernel | pure metadata once resident |
| `matmul` | 24 | 0 | **no kernel** | unfused QK^T / attn·V, compute-bound not DMA-bound. Needs HMX |
| `scale` | 12 | 0 | ✅ `scale_fp16`, gated, dispatchable | elementwise |
| `softmax` | 12 | 0 | **no kernel** | must use the fp32 exp path — see `docs/hvx/upstream-findings.md` |
| `cast` | 1 | 0 | ✅ `cast_f32_f16`, gated, dispatchable | runs once, at the input |

**86 of the 259 ops that need a kernel are covered and dispatchable today**; 49 of the
308 steps are reshapes needing none. `matmul_epilogue` and `matmul` together are 99 of
the remaining 173, and are the only two requiring HMX — which this codebase has not yet
used at all.

Total across all eleven kinds: `predicted_bytes_moved = 58,643,456` at 256x256, against
the measured `vtcm_high_water = 5,355,648` of an 8,388,608-byte budget (63.8%). Both
figures now include the image's DMA-in and the encoder output's DMA-out (previously
missing entirely -- see the whole-branch fix report), and `vtcm_high_water` now
includes the const/weight-streaming region (previously computed and budget-checked,
but never folded into `plan.vtcm` or the reported high-water number).

Reproduce both numbers yourself with `hexlib plan qwen35 --print`, which needs no SDK.
The reasoning behind each fix is in the commit that made it — `git log --grep=vtcm` and
`git log --grep=traffic` — and the durable summary is in `docs/STATE.md`. (Earlier
revisions of this section cited report files under `.superpowers/`, which is a
git-ignored agent scratch directory and therefore not something a reader can open.)

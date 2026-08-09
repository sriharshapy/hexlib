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
at 2021 kernel cycles (34.36x over a 69443-cycle scalar baseline), beating an adapted
v6 `fp16_rmsnorm` (10498 cycles) by 5.19x. ggml-hexagon's `hvx-norm.h` was not
evaluated against it — see below.

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

# rmsnorm_fp16 — bake-off

Measured on the hexagon simulator, target v75, toolchain 19.0.04, bus model
pinned (buspenalty 75, busratio 2). Identical harness, identical inputs, identical
flags for every candidate — each candidate was placed at `kernel.c` in turn and run
through `hexlib test kernels/rmsnorm_fp16` unmodified otherwise. The simulator is
cycle-approximate; these are reproducible numbers, not silicon measurements.

Shape: R=8, C=128, eps=1e-5 (`kernel_api.h`). Both HVX candidates and the scalar
baseline were built and run under this identical shape — none of the recorded v6
cycle counts (9798, 2311) are reused directly; they were measured on the different
shapes noted below and are cited only as provenance, not as the numbers being
compared.

| candidate | source | correct | kernel_cycles | accel (ELF) | notes |
|---|---|---|---|---|---|
| scalar baseline | this repo, `baseline.c` copied to `kernel.c` for the Step 6 discrimination check | yes | 69443 | none | reference; both near-misses correctly rejected against it, confirming the harness discriminates before any HVX kernel existed |
| v6 `rmsnorm_gain_fp16` (adapted) | HVX-clean v6, handwritten, `solutions/s2.c` (ror-shift butterfly reduce) | yes | 2231 | hvx, hvx-compute | **winner.** Original is R=6,C=80 with a PER-ROW scalar gain and recorded 9798 cycles at that shape/2.741x; adapted to R=8,C=128 with hexlib's PER-COLUMN gain (rewrote the scale epilogue as a real vector×vector `w[block]` multiply instead of a scalar-splat gain, since a per-row scalar cannot express a per-column vector) and generalized the reduction from reading only block 0 (correct only for the original's single-block C=80 shape) to accumulating qf16 sum-of-squares across all `nb=C/64` blocks before the one-time ror-shift reduce. eps threaded as a parameter instead of hardcoded `1e-3f`. See "Fix round 2" below for the 2021 -> 2231 move. |
| v6 `fp16_rmsnorm` (adapted) | HVX-clean v6, handwritten, block-accumulate + unpack-once scalar sum | yes | 10498 | hvx, hvx-compute | Original is n=2048 (single row) and recorded 2311 cycles at that shape/60.53x. **Correction to this task's brief:** the brief describes this candidate as "no gain vector — you must add the w[] multiply", but the actual `expert.c` already multiplies by a per-feature `gamma[]` in the scale epilogue, structurally identical to hexlib's per-column `w[c]`; no gain multiply had to be added. Wrapped the original single-vector body in a `for r in [0,R)` loop (x/y offset by `r*C`, `w[]` reused unchanged every row, exactly like the original's `gamma` not varying by call). eps threaded as a parameter instead of hardcoded `1e-3f`. Loses to candidate A by 5.19x at this shape: its reduction unpacks the qf16 accumulator to memory and finishes with a 64-iteration scalar add loop — negligible when paid once for n=2048, but paid once PER ROW here (8x), while candidate A's ror-shift butterfly never leaves the vector unit. |
| ggml-hexagon `hvx-norm.h` | llama.cpp, MIT | — | — | — | **not evaluated: fp32 only.** `hvx_fast_rms_norm_mul_f32` (`include/hexlib/hvx/hvx-norm.h`) is an exact semantic match — RMSNorm with a per-column gain vector, reduction in `Vqf32`/`Vsf` — but there is no fp16 norm anywhere in the vendored ggml-hexagon set, and v6 has no fp32 norm at all (every v6 norm is fp16 or i8). Comparing an fp32 implementation against fp16 candidates would measure the dtype, not the implementation. Deferred to kernel #2, `rmsnorm_f32`, where it goes head to head with a hexlib implementation on identical fp32 shapes — the first direct measurement against production ggml-hexagon code. |

**Fix round 1 (numbers above are post-fix).** `hexlib/build.py` originally linked
only `[impl.c, harness.c]`, so this kernel's `harness.c` (which calls
`rmsnorm_fp16_baseline` from `baseline.c` for its reference) could only link by
`#include`-ing `baseline.c` directly into `harness.c`'s translation unit. That
workaround was flagged as a blocking follow-up before it calcified into the
convention every later kernel would copy, and `build_kernel` was fixed to link
`baseline.c` as its own third source. Re-measuring after the fix: the scalar
baseline moved 69438 -> 69443 (+5 cycles), the winner moved 2020 -> 2021
(+1 cycle), and the losing HVX candidate did not move at all (10498 -> 10498).
Same source, same compiler flags, same harness logic in every case — only the
final ELF's link-time object layout changed (baseline.c compiles to its own
object now instead of being merged into harness.o). A cycle-approximate
simulator with a pinned bus/cache timing model is sensitive to exactly this kind
of instruction-address/cache-line shift; the reviewer's original conclusion
(the translation-unit merge could not affect the measurement, since there is no
LTO and no cross-TU inlining either way) is consistent with a change this small
and this inconsistent in direction across candidates. The bake-off's conclusion
(candidate A wins, by roughly the same ~34x/~5x margins) is unaffected.

**Fix round 2 (final whole-branch review, ledger #3).** `kernel.c`'s
sum-of-squares reduction had `if (nb > 0) { ... } else { accSq =
Q6_Vqf16_vmpy_VhfVhf(xv[0], xv[0]); }` inside the per-row loop. The `else` was
not merely dead (`nb = C/BLK` with `BLK=64` and `kernel_api.h`'s contract
requiring `C` a multiple of 64, so `nb` is never `<=0` for a conforming call) --
it was unsafe if ever reached: `xv[0]` is a full 128-byte HVX vector load past
the end of a row shorter than one vector, an out-of-bounds read. Removed the
`else` and hoisted the `nb <= 0` guard above the row loop as an early `return`,
so a future caller violating the C-is-a-multiple-of-64 contract gets a no-op
instead of an out-of-bounds read. Re-measuring: the winner moved 2021 -> 2231
(+210 cycles, ~10.4%). That is a real codegen change, not link-layout noise
like round 1's ~0.05%/1-5-cycle shifts: removing a branch and its dead body,
and moving a loop-invariant check out of the row loop, changes the actual
instruction stream the cycle-approximate bus/cache timing model times, not
just where it sits in the binary. Scalar baseline (69443) and the losing HVX
candidate (10498) are unaffected -- neither's source was touched by this fix.
The bake-off's conclusion (candidate A wins) is unaffected; only its margin
narrows slightly (31.13x / 4.71x, from 34.36x / 5.19x).

**Winner:** v6 `rmsnorm_gain_fp16` (adapted) at 2231 cycles, 31.13x over the scalar
baseline (69443 cycles) and 4.71x over the other HVX candidate (10498 cycles).

**Why it wins:** both HVX candidates vectorize the sum-of-squares reduction and the
scale epilogue the same way at the block level (`Q6_Vqf16_vmpy_VhfVhf` +
`Q6_Vqf16_vadd_Vqf16Vqf16` accumulate, one qf16 multiply for `x*inv_rms`, one more
for `*w[block]`). The difference is how the 64-lane qf16 accumulator becomes one
scalar. The winner finishes the horizontal reduction entirely inside the vector
unit — a six-step ror-shift butterfly (`hreduce_qf16`: rotate the accumulator by
64/32/16/8/4/2 lanes with a qf16 add at each step) leaves every lane holding the
full sum, so reading lane 0 is the only vector-to-scalar transition in the whole
reduction. The loser instead unpacks the accumulator to memory and finishes with a
64-iteration scalar float add loop — cheap when paid once for a single n=2048 call
(the original v6 shape), but paid once per row here, so an 8-row batch pays it 8x.
The reciprocal square root is a single scalar `sqrtf` call either way (O(1), never
the bottleneck); the gain multiply is folded into the same per-block vector pass as
the scale, with no extra pass over the data.

**Authorship:** the winning implementation is adapted from HVX-clean's v6 corpus,
`data/v6/tasks/rmsnorm_gain_fp16/expert.c` (handwritten HVX expert, `solutions/s2.c`
variant), same author/repository as hexlib. The reduction and scale mechanism is
carried over verbatim; the gain axis (per-row scalar to per-column vector), the
multi-block reduction (single block to `nb` blocks), and the `eps` parameter
(hardcoded to threaded) were rewritten for this task's contract. See `kernel.c`'s
header comment for the exact adaptation.

To beat it: implement `rmsnorm_fp16` against `kernel_api.h`, run
`hexlib test kernels/rmsnorm_fp16`, and open a PR with your result table. Any
candidate that is correct and faster becomes the champion.

# Contributing to hexlib

**Contributing a kernel requires the Hexagon SDK, and you run the gates yourself.**
`hexagon-clang` compiles it and `hexagon-sim` simulates it; both are SDK-only. hexlib
has no build service, no self-hosted runner, and no credential held on your behalf.
You run `hexlib test` on your own machine and attach the result table it prints to
your pull request. Nobody runs the gates for you, and CI cannot — see "What CI checks
and what it cannot" below.

`hexlib test` writes the table to `_work/<name>.result.md` **and**, on a pass, to
`kernels/<name>/RESULT.md`. Commit the `RESULT.md` — CI has no Hexagon SDK, so that
committed file is the only evidence it can check. Paste the same table into your pull
request description for reviewers.

This is a deliberate trade, not an oversight: building anything for Hexagon requires
`hexagon-clang`, so **the SDK is required to use hexlib at all, not merely to
contribute to it.** Anyone who would deploy a hexlib kernel to a Hexagon NSP needs the
toolchain regardless. Tier 1 below is not "contributors we turned away" — it is
"everyone who could have used this anyway."

## The three tiers

| tier | needs | can contribute | verified |
|---|---|---|---|
| 0 | nothing | docs, scalar reference implementations, test vectors, `spec.json` definitions, host-side code, tooling | GitHub Actions |
| 1 | x86 + Hexagon SDK | kernels — full compile, simulation, correctness, accel proof | locally, table attached to PR |
| 2 | a Snapdragon device or your own QDC account, arranged by you | silicon validation | locally, table attached to PR |

Tier 0 is deliberately large and is the on-ramp for everyone else. A scalar
`baseline.c` plus test vectors plus a `spec.json` is a complete, useful contribution
that unblocks a kernel author, and it needs nothing but a C compiler. `ROADMAP.md` is
written so tier-0 work is always available and always visibly wanted.

Device access (tier 2) is entirely your own affair — a phone on your desk or a QDC
account you arrange yourself. hexlib does not broker either, and no QDC credential is
ever stored in GitHub secrets or passed to CI.

## The six gates

A kernel clears these one at a time, and is unsupported until it clears all six.

1. **Select** — from `ROADMAP.md`, prioritized by LLM inference need, not by ease.
2. **Gather candidates** — any existing handwritten expert, any ggml-hexagon
   implementation that is a genuine semantic match, and any fresh attempt. All
   wrapped to the same signature.
3. **Bake off in simulation** — identical harness, identical inputs, identical flags,
   toolchain pinned to `19.0.04`. Correctness first, then `kernel_cycles`, then ELF
   anti-cheat. The result table is committed as `BAKEOFF.md`.
4. **Generalize** — the winner gets a real name, shape parameters, a public header, a
   documented authorship record, and a comment explaining *why* it is fast.
5. **Adversarial** — the `nearmiss_*.c` variants must fail, and `spec.json` edge cases
   become named tests.
6. **Silicon** — batched QDC or local device, sim-vs-silicon drift recorded. **The
   transport now exists; the per-kernel record does not.** A QDC session with
   `--stage-dir` pushes a batch blob and its arena to an SM8650, runs it, and pulls the
   arena back — the whole encoder has gone through it in a single FastRPC invoke, and
   the one sim-vs-silicon cycle comparison that produced is in `README.md`. What is
   still missing is a *per-kernel* drift record and a place to put it, so a kernel that
   has cleared gates 1-5 has still not cleared gate 6. Do not mark one as gate-6 clear
   on the strength of the whole-encoder run.

   **An HMX kernel cannot clear gates 3 and 5 on the standalone-ELF path at all** — HMX
   needs power, acquisition, a lock and possibly a dedicated thread, none of which exist
   in a program with no protection domain around it. HMX work belongs on the QuRT-hosted
   batch path; see `kernels/hmx_matmul_fp16/README.md`, which is a worked example of a
   kernel committed *without* a `RESULT.md` because it does not gate.

`hexlib new-kernel <name>` scaffolds a directory that satisfies the structural half of
gates 1-5 (required files, a near-miss stub, a matching `spec.json`); `hexlib
validate <kernel>` checks that structure without building anything; `hexlib test
<kernel>` runs gate 3 and reports gates 3 and 5 together in one table.

## The near-miss requirement

Every kernel PR must include at least one `nearmiss_*.c`: a plausible-but-wrong
variant — a skipped `eps`, a wrong axis, a mean instead of an RMS — that the harness
must reject. **A harness that accepts your kernel proves nothing until it is shown to
reject something close by.** `hexlib validate` refuses a kernel directory with zero
near-miss files, and the gate itself refuses to pass a kernel whose near-miss set is
empty or whose near-misses are merely *inconclusive* (failed to compile, failed to
run) rather than actually rejected — an inconclusive near-miss was never offered to
the harness, so it demonstrates nothing about whether the harness discriminates, and
counting it as a rejection would let a typo in a near-miss silently turn the gate
green.

## Integer kernels must be bit-exact

Tolerance comparison exists because HVX float goes through the non-IEEE qf16 path and
float operations reorder, so an fp16 result cannot be compared bit-exactly against a
scalar reference. An integer kernel has neither property: there is exactly one right
answer, byte for byte. A tolerance on an integer path would hide wrong results instead
of accommodating the hardware. `validate_dir` enforces this: any `spec.json` whose
`dtype` contains no floating-point token (`fp16`, `fp32`, `f16`, `f32`, `float`,
`half`, `bf16`) must declare `"tolerance": "exact"`, or validation fails.

## `spec.json`'s `mechanisms` and `params`

`mechanisms` is checked against the ELF: `hexlib test` rejects a declared `hvx` or
`hmx` claim the compiled object does not actually show (the two mechanisms that are
ELF-provable; `dma`, `vtcm`, `l2fetch`, and `scalar` are not, so they are not checked).
`params` is descriptive metadata only — nothing cross-checks it against
`kernel_api.h`'s `#define`s, which are what actually gets compiled and are the
authoritative source of truth for shapes and constants. Keep `params` accurate for
readers, but do not rely on it being enforced.

## The bake-off

To challenge an existing kernel: implement your candidate against the same
`kernel_api.h` contract, run `hexlib test kernels/<name>` unmodified otherwise, and
open a PR with your result table. **Any candidate that is correct and faster becomes
the champion.** The existing `BAKEOFF.md` records every candidate ever measured,
including losers and build failures, with each candidate's source and, where
applicable, authorship or license — add your row to it rather than replacing it. See
`kernels/rmsnorm_fp16/BAKEOFF.md` for the shape a bake-off record takes, including how
a losing candidate is written up with the same care as the winner.

## What CI checks and what it cannot

CI has no Hexagon SDK — it is license-restricted and is never installed on a hosted
runner — so CI **cannot** compile, simulate, or re-verify a kernel's numbers. What CI
*can* do, and does, on every PR:

- run the offline test suite (`pytest -m "not sdk"`): CLI logic, `spec.json` schema
  checks, and everything that needs no toolchain;
- for every directory under `kernels/`, check that it is structurally complete
  (`check_kernel_contract`): required files present, at least one near-miss, a
  `RESULT.md` that exists, decodes as UTF-8, and parses;
- check that the parsed `RESULT.md` records the pinned toolchain (`19.0.04`) and the
  pinned target (`v75`); and
- check that the parsed `RESULT.md`'s gate row says **PASS**.

CI rejects a kernel directory whose `RESULT.md` is missing, fails to parse, records a
different toolchain, or records `FAIL`. That does not make a forged table impossible
— it makes an *accidental* green impossible, and it makes a deliberate one a visible,
dated, attributed claim. **Numbers are reproducible by any maintainer with an SDK**:
the simulation gate is deterministic given a pinned toolchain, so a maintainer can
rerun your bake-off and get the same numbers, and a mismatch is visible rather than
silent.

## Absence is not evidence

During this project's own construction, the same bug shape appeared four separate
times, in four different components:

- a device-farm job that completed having run zero tests was reported as passing
  (QDC job 742504: `completed`, no `TestLogs/results.xml`, an empty parsed test dict,
  and `passed = job_ok` anyway);
- a near-miss that failed to compile was counted as "the harness caught it" (any
  `BuildError`/`SimError` from a near-miss was treated as rejection, so a typo in a
  near-miss silently greened the gate);
- a corrupted `RESULT.md` parsed cleanly and returned zero problems (`errors=
  "replace"` repaired invalid bytes before the parser ever saw them, so damage
  outside the two regex-anchored regions was invisible); and
- a test that scans the vendored HVX headers for forbidden `ggml` references passed
  when there were no headers to scan (`assert offenders == []` is true of an empty
  list, so a missing or empty vendor directory looked identical to a clean one).

The pattern is the same every time: **a check that finds nothing to check reports
success.** An empty result is not evidence of a passing state; it is evidence that
the check never ran. When you write a guard — a test, a contract check, a gate — make
its empty case an explicit failure (`assert results, "nothing to check"`, a
three-state outcome instead of a boolean, a decode error that becomes a problem
instead of a silent repair). And when you add a new guard, verify it by making it
fire once: break the thing it's supposed to catch and confirm the guard catches it,
before trusting it to stay quiet on everything else. This is the most useful thing
this project has learned about itself.

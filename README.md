# hexlib

**A compiler and kernel library for Qualcomm Hexagon NPUs.**

hexlib takes a neural network, compiles it to an explicit VTCM and DMA plan, and
executes that plan against hand-written HVX/HMX kernels on a Hexagon NSP — targeting
v75 (Snapdragon 8 Gen 3 / SM8650) first. Every kernel arrives with a correctness
verdict against a scalar reference, a cycle count, ELF-level proof that the vector or
matrix unit was genuinely used, and near-miss variants that must still fail.

Two things make it unusual. **The scheduling layer is pure host Python** — the graph,
the passes, the allocator and the plan need no SDK, no simulator and no device, which
is where most of the contribution surface lives. And **the verification is adversarial
by construction**: a kernel does not pass because it produced plausible numbers, it
passes because a deliberately-broken variant of it demonstrably fails.

---

## Status

Honest state, because this project's own worst recurring bug is a claim that outruns
its evidence.

| | what works | where |
|---|---|---|
| **Kernel pipeline** | ✅ shipped | write a `.c`, run `hexlib test`, get a gate verdict + cycles + ELF proof |
| **Graph → plan compiler** | ✅ shipped | `hexlib plan qwen35 --print`, no SDK needed |
| **Plan executor** | ✅ shipped | whole encoder runs end to end; validated against PyTorch **on a tiny config only** — no full-size reference exists yet |
| **6 kernels** | ✅ gated | 4 dispatchable from the executor; 86 of 259 real-work ops |
| **Silicon-path runtime** | 🚧 on a branch | FastRPC + DSP skel; simulator green, **never run on hardware** |
| **On-device execution** | ❌ not yet | cross-compiles and stages; no job has been run |

**Nothing here has executed on real silicon.** All cycle counts come from
`hexagon-sim` under a pinned bus model. The simulator is cycle-*approximate* — see
[`docs/hardware/simulator-accuracy.md`](docs/hardware/simulator-accuracy.md) for where
it is most likely to drift.

### The Hexagon SDK is required to build or run a kernel

Not just to contribute — to use hexlib on a kernel at all. `hexagon-clang` and
`hexagon-sim` are discovered through `HEXAGON_SDK_ROOT`. **hexlib never vendors,
bundles, or fetches the SDK**; it is licence-restricted and you obtain it yourself.

Without it you can still do a great deal, and it is the most useful work available:
the entire graph and scheduling layer, the op registry, the numpy reference executor,
scalar baselines, `spec.json` contracts and test vectors are all pure Python. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the tier system.

---

## Quickstart

```bash
pip install -e .

# No SDK required — compile a model to a VTCM/DMA plan and inspect it
hexlib plan qwen35 --print

# SDK required — scaffold, then gate, a kernel
hexlib new-kernel my_kernel
hexlib test my_kernel
```

`hexlib new-kernel` writes a conforming directory: `kernel.c`, `kernel_api.h`,
`baseline.c` (your scalar reference), `harness.c` (which builds its own inputs and so
cannot be handed a passing answer), a `nearmiss_*.c` stub, and `spec.json`.

`hexlib test` compiles it, runs it on the simulator, disassembles the ELF to prove HVX
or HMX was used, confirms every near-miss variant is still rejected, and writes a
`RESULT.md` you attach to a PR.

---

## How it works

```
  model (PyTorch/HF config)
        │
        ▼
  graph IR ── op registry (13 kinds, each with a numpy reference)
        │
        ├── shapes → fuse → order → liveness → VTCM alloc → DMA
        │                                                    │
        ▼                                                    ▼
     Plan  ─────────────────────────────────────────►  serialized, diffable
        │
        ▼
   executor ── replays the plan through a real VTCM byte image
        │
        ├──► numpy reference        (any op without a kernel)
        ├──► standalone ELF path    (one simulator launch per op)
        └──► DSP skel batch path    (one FastRPC invoke per batch)  ◄── the silicon path
```

Every pass is a pure function, so ~80% of the system is testable with no SDK and no
device. The plan is the contract between the two halves: the compiler decides *where
every byte lives and when it moves*, and the executor is deliberately dumb.

**Layout is an enumerated value, not `ne`/`nb` strides.** This makes "the kernel got
un-repacked weights" a plan-time error rather than silent numerical corruption, and
strides cannot express a VTCM-resident tile of a DDR tensor — which is the central
object the compiler manipulates.

Design docs: [encoder](docs/superpowers/specs/2026-08-09-vlm-encoder-design.md) ·
[silicon path](docs/superpowers/specs/2026-08-10-silicon-path-runtime-design.md) ·
[architecture overview](docs/architecture.md)

---

## Kernels

Six kernels through the gates. Cycles are `kernel_cycles` — the DSP-side count for the
kernel call alone, never whole-program `cycles`, which carries 155k–190k of roughly
constant harness and CRT overhead.

| kernel | cycles | accuracy vs numpy | notes |
|---|---|---|---|
| `scale_fp16` | **886** | exact (normal range) | factor 0.125 is a power of two, so no mantissa bit is lost |
| `transpose_th_fp16` | **706** | exact | perm (1,0,2), both directions |
| `add_fp16` | **1139** | 1 ULP | the hardware's fp16 narrowing is not IEEE round-to-nearest-even |
| `cast_f32_f16` | **1176** | bit-exact | needs a lane deal — the widening conversion interleaves |
| `rmsnorm_fp16` | **2231** | — | **31.13×** over a 69443-cycle scalar baseline |
| `layernorm_fp16` | 111088 | — | **a first rung, not a result** — reductions still scalar |

`layernorm_fp16`'s number is deliberately unoptimised: the affine epilogue is
vectorised, both reductions are not. It was left scalar so the reduction has a
*recorded* baseline to beat rather than an assumed one. A rotate-and-add butterfly
already exists in `kernels/rmsnorm_fp16/`.

Full bake-off records, including the candidates that **lost**, live in each kernel's
`BAKEOFF.md`.

### Target model

The Qwen3.5-0.8B vision encoder, at 256×256:

```
VTCM high water   5,355,648 of 8,388,608 bytes (63.8%)
DDR ↔ VTCM        58,643,456 bytes
Plan steps        308   (396 ops before fusion)
```

`matmul_epilogue` alone accounts for 56.0 of those 58.6 MB — 95.5% of all the traffic —
which is why it is next.

**Numerical validation is at a different scale, and the distinction matters.** The plan
figures above are at 256×256. The accuracy figures below are **not**: they are measured
on a *tiny* config — 2 layers, hidden 64, image 32 — against committed golden vectors,
with no torch at test time.

| | |
|---|---|
| tiny config vs upstream `transformers` | **4.47e-08** |
| tiny config through the plan executor, fp32 | 4.470e-08 |
| tiny config through the plan executor, fp16 | 6.747e-05 |

**There is no full-size PyTorch reference yet**, so nothing here says the 0.8B encoder is
validated at 256×256. What the tiny config does establish is that the graph, the pass
pipeline, the plan and the executor agree with upstream to fp32 round-off, and what the
fp16 row costs — which is the part a larger config would not change. Obtaining a
full-size reference is tracked in [`docs/STATE.md`](docs/STATE.md).

*(Corrected 2026-08-11: these three figures previously sat directly under the "at
256×256" heading with no scale caveat, which read as a claim about the full model.)*

---

## How correctness is established

The gates exist because of specific ways this project has been wrong before, each
recorded in [`CONTRIBUTING.md`](CONTRIBUTING.md):

- **The harness builds its own inputs** and never reads a file, so it cannot be handed
  a passing answer. The runner that *does* read files is a separate binary that prints
  no verdict. Both facts are asserted by a test.
- **Acceleration is proven from the compiled ELF**, by disassembly — not from source
  text, and not from a self-reported flag.
- **Near-misses must fail.** A dropped tail, a mean instead of an RMS, a forgotten lane
  deal: each is committed as a variant that the harness has to reject. One of them
  found a real bug in hexlib's own simulator wrapper.
- **A tolerance wide enough for the widest shape can be wider than the bug it is meant
  to catch.** `layernorm`'s unbiased-variance near-miss is a 0.065% error at C=768,
  where fp16's own precision is ~0.05% — indistinguishable. It was *wrongly accepted*
  on the first run. The fix was a shape where the bug is bigger (C=64, 0.79%), not a
  tolerance argued down.
- **Absence is never success.** Status codes start at 1, so a zero-filled response
  buffer that was never written cannot read as OK.

---

## Repository layout

```
hexlib/graph/      IR, op registry, and the pass pipeline (pure Python, no SDK)
hexlib/exec/       the plan executor and its three dispatch backends
hexlib/runtime/    the silicon path: wire format, IDL, DSP skel, host, build recipes
hexlib/device/     device backends (QDC job plumbing)
hexlib/tests/      the offline suite — runs without an SDK, except where marked
include/hexlib/    DSP-side headers a kernel includes, incl. vendored HVX math
kernels/           one self-contained directory per gated kernel
docs/hvx/          learning HVX: a function-by-function tour of the vendored headers
docs/hardware/     measured hardware notes (HMX int8, simulator accuracy)
docs/research/     audit records — what was read directly vs. inferred
```

## Documentation

**Start here**
- [`docs/architecture.md`](docs/architecture.md) — how the pieces fit together
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — the tiers, the six gates, what CI does and does not check
- [`ROADMAP.md`](ROADMAP.md) — the op backlog, so tier-0 work is always visible

**Learning HVX**
- [`docs/hvx/`](docs/hvx/README.md) — a guided tour of all 22 vendored headers: vector
  types and predicates, alignment, horizontal reductions, transcendentals from
  polynomial approximation, division by Newton–Raphson, and the reduce-then-broadcast
  pattern nearly every transformer kernel is a variation on. **They record what they
  could not explain** rather than smoothing it over — the open questions are written
  down deliberately. (No count is given here on purpose: a number in a doc goes stale,
  and this one had.)
- [`docs/hvx/upstream-findings.md`](docs/hvx/upstream-findings.md) — three real defects
  found in upstream llama.cpp while writing that tour, with evidence. hexlib calls none
  of them; the worst is a coefficient off by 234,118× inside an fp16 exponential.

**Hardware reality**
- [`docs/hardware/simulator-accuracy.md`](docs/hardware/simulator-accuracy.md) — what
  "cycle-approximate" means and where it drifts
- [`docs/hardware/hmx-int8.md`](docs/hardware/hmx-int8.md) — the measured HMX int8 MAC sequence
- [`docs/research/oracle-provenance.md`](docs/research/oracle-provenance.md) — what the
  committed golden vectors prove, and what they do not

**Project state**
- [`docs/STATE.md`](docs/STATE.md) — the working handoff record: what is decided, what
  is proven, what is merely claimed, and every open question

## Contributing

Tier-0 work needs no SDK and no hardware: scalar baselines, `spec.json` contracts, test
vectors, near-miss variants, documentation, and anything in the graph or pass pipeline.
[`ROADMAP.md`](ROADMAP.md) keeps that work visible. Read
[`CONTRIBUTING.md`](CONTRIBUTING.md) first — the gates are non-negotiable, and the
reason each one exists is written down.

## License

[MIT](LICENSE).

hexlib **vendors** MIT-licensed HVX math headers from llama.cpp's `ggml-hexagon`
backend (byte-identical, never edited in place) and **adapts** its FastRPC runtime
(rewritten in hexlib's own tree). Both carry attribution requirements, and every source
— with its licence and the upstream commit it came from — is recorded in
[`ATTRIBUTION.md`](ATTRIBUTION.md).

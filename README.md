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
| **Plan executor** | ✅ shipped | whole encoder runs end to end, and now against the **real** Qwen3.5-0.8B checkpoint at 256×256 |
| **12 kernels** | ✅ gated | **every one of the encoder's 259 real-work ops selects a kernel** |
| **Silicon-path runtime** | ✅ runs on hardware | whole plan, **one** FastRPC invoke, on an SM8650 |
| **On-device execution** | ⚠️ tiny config only | the 256×256 encoder has never been run on a device |
| **HMX** | ❌ blocked | a kernel exists and does **not** gate — see below |

Cycle counts, unless a row says silicon, come from `hexagon-sim` under a pinned bus
model. The simulator is cycle-*approximate* — see
[`docs/hardware/simulator-accuracy.md`](docs/hardware/simulator-accuracy.md) for where
it is most likely to drift. On the one workload measured both ways it runs 6.7% high —
see [Results](#results-pytorch--simulator--silicon) below, which chains PyTorch, the
simulator and the device.

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

Twelve kernels through gates 1–5. Cycles are `kernel_cycles` — the DSP-side count for
the kernel call alone, never whole-program `cycles`, which carries 155k–190k of roughly
constant harness and CRT overhead. Each row's number and near-miss set is the
tool-generated `kernels/<name>/RESULT.md`, not a figure retyped here.

| kernel | cycles | max abs error | notes |
|---|---|---|---|
| `transpose_th_fp16` | **706** | 0 | perm (1,0,2), both directions |
| `scale_fp16` | **886** | 0 | factor 0.125 is a power of two, so no mantissa bit is lost |
| `add_fp16` | **1139** | 0 | 1 ULP class: the hardware's fp16 narrowing is not IEEE round-to-nearest-even |
| `cast_f32_f16` | **1176** | 0.5 | needs a lane deal — the widening conversion interleaves |
| `rope_2d_fp16` | **1212** | 6.10e-05 | six near-misses, the largest set here — pairing, table indexing and sign are each independently wrong-able |
| `rmsnorm_fp16` | **2231** | 1.95e-03 | **31.13×** over a 69443-cycle scalar baseline |
| `transpose_hd_fp16` | **5606** | 0 | perm (0,2,1); movement-only, no arithmetic in the ELF |
| `softmax_fp16` | **11292** | 0 | fp32 exp path deliberately — see the upstream findings below |
| `layernorm_fp16` | 111088 | 4.88e-04 | **a first rung, not a result** — reductions still scalar |
| `matmul_fp16` | 995714 | 9.77e-04 | the encoder's 24 unfused attention matmuls; gated at N=200 |
| `patchify_fp32` | 2018331 | 0 | runs once, at the input; emits merge-block order, not raster |
| `matmul_epilogue_fp16` | 14310406 | 2.44e-04 | fused matmul+bias+activation over q4_0 weights — 75 of 308 steps and 95.5% of all DDR traffic |

`layernorm_fp16`'s number is deliberately unoptimised: the affine epilogue is
vectorised, both reductions are not. It was left scalar so the reduction has a
*recorded* baseline to beat rather than an assumed one. A rotate-and-add butterfly
already exists in `kernels/rmsnorm_fp16/`. `matmul_epilogue_fp16` is the same story at
the other end of the scale — it is correct and gated, and nothing has been optimised
about it yet.

Full bake-off records, including the candidates that **lost**, live in each kernel's
`BAKEOFF.md`.

**A thirteenth kernel is committed and does not gate.**
[`kernels/hmx_matmul_fp16/`](kernels/hmx_matmul_fp16/README.md) has no `RESULT.md`,
deliberately, because the simulator faults before the harness prints a verdict — and
the gate reports that as a failure, not a pass. The reason is now known exactly and it
is structural: **HMX cannot be used inside `hexlib test`'s standalone ELF at all.** The
directory's `README.md` records the four things it needs, the two real defects found on
the way, and what to do next.

### Target model

The Qwen3.5-0.8B vision encoder, at 256×256:

```
VTCM high water   5,355,648 of 8,388,608 bytes (63.8%)
DDR ↔ VTCM        58,643,456 bytes
Plan steps        308   (396 ops before fusion)
```

`matmul_epilogue` alone accounts for 56.0 of those 58.6 MB — 95.5% of all the traffic.

Every one of the 259 real-work steps now selects a dispatchable kernel; the remaining 49
are reshapes, which are pure metadata once resident and need none. That claim is
*asked*, not counted — `hexlib/tests/test_encoder_dispatch_coverage.py` puts every step
of the real compiled plan through `select()`, because this project published a wrong
coverage number three times by counting kernel directories instead. (`hexlib plan
--print` still lists all eleven kinds under "no kernel": `OpDef.kernel` is a separate
registry that is still `None` everywhere, and wiring it moves figures several tests pin.)

### Accuracy — measured on the real checkpoint

The shipped `Qwen/Qwen3.5-0.8B` vision weights at 256×256, against `transformers`, with
fp32 arithmetic on **both** sides so nothing but the weight format differs:

| | cosine vs `transformers` |
|---|---|
| hexlib fp32, full-precision weights | **0.9999999999** |
| q8_0 weights | **0.999002** (max_rel 7.98e-02, 1.89× the weight bytes) |
| q4_0 weights | **0.867606** (max_rel 4.32e-01) |
| fp16 *activations*, weights exact | 0.999991 — **15,000× smaller than quantization** |

**q4_0 is not enough for this encoder, and it is not the kernels' fault.** Mixed
precision was measured and rejected: the error is *diffuse*, so keeping the patch
embedding, the merger and all of attention at 8 bits while the MLPs stay q4_0 still only
reaches 0.913. Smaller blocks were measured and rejected: block=8 spends 6 bits/value on
more scales for 0.937, where q8_0 spends 8.5 on mantissa for 0.999. Fusion and HMX
cannot recover it either — fusion's entire budget is the fp16-activation term, 8.8e-06
of cosine, and HMX is a speed lever, not an accuracy one.

**A tiny-config sweep said the opposite of all of this** and nearly sent the work the
wrong way: at 2 layers with random weights it ranked the merger dominant, put `wq`/`wk`
at the noise floor, and made mixed precision look like a 30× win. Two layers is not
enough depth for diffuse error to compound, and random normals have no outliers. **Do
not tune quantization against the tiny config.**

The tiny config (2 layers, hidden 64, image 32) is still what the committed golden
vectors cover, and still what runs with no torch at test time: 4.47e-08 vs upstream,
4.470e-08 through the plan executor in fp32, 6.747e-05 in fp16.

*(Corrected 2026-08-11: the tiny-config figures previously sat directly under the "at
256×256" heading with no scale caveat, which read as a claim about the full model. The
full-size reference that was missing then now exists — it is the first row of the table
above.)*

---

## Results: PyTorch → simulator → silicon

The chain is four links, and each is checked against the one before it rather than
against an assumption. Read down the table: what upstream `transformers` computes, what
hexlib's numpy reference computes from the same weights, what the Hexagon simulator
computes running the real kernels, and what an SM8650 computes running the same blob.

| link | what is compared | scale | result |
|---|---|---|---|
| **PyTorch → hexlib reference** | upstream `transformers` vs the graph + passes + plan executor, fp32 both sides, real `Qwen/Qwen3.5-0.8B` weights | **256×256, 12 layers** | cosine **0.9999999999** |
| **PyTorch → hexlib reference** | same, against committed golden vectors with no torch at test time | tiny (2 layers) | **4.47e-08** max abs |
| **hexlib reference → simulator, per-op** | every op with a kernel routed through `hexagon-sim`, one launch each, vs the numpy registry over the identical plan and feeds | tiny (49 ops) | max rel **1.1319e-03**, corr **1.000000** |
| **hexlib reference → simulator, one invoke** | the whole plan as a *single* batch blob, one simulator entry | tiny (49 ops) | the same figure |
| **hexlib reference → silicon, one invoke** | the same blob, one FastRPC invoke on an SM8650 | tiny (49 ops) | the same figure — cosine **0.99999967** |

**Three transports, one answer.** The per-op simulator path, the single-invoke simulator
path and the device agree to the digits printed above; that agreement is the point, not
the individual number. The residual 1.13e-03 is fp16 activations compounding across the
encoder, not a wrong kernel — every intermediate narrows to fp16 and feeds the next op.

Cycles, on the one workload measured both ways:

| | cycles_total | note |
|---|---|---|
| simulator, single invoke | 302,087,160 | no `--timing --buspenalty 75 --busratio 2` on the batch path |
| SM8650, single invoke | **283,220,278** | 0 ops not OK |

The simulator is **6.7% high** here. That is a bare comparison of two numbers, not a
calibrated drift figure — the flags differ, and one workload is not a model.

**Everything on the device is the tiny config.** The 256×256 encoder compiles to a 46 KB
blob over a 203.7 MB arena and **has never been run on a device**. Nothing above is a
full-model silicon result. The reason is cost, not capability: a full-size *simulator*
run is 259 separate `hexagon-sim` launches, which is hours for a signal a 49-op graph
gives in minutes, and QDC sessions bill for their whole timeout.

Three things silicon settled that no simulator test could:

- **`cycles_total` is non-zero in a user-mode unsigned PD.** `SYSCFG.PCYCLEEN` cannot be
  set there, and a dead counter would have invalidated every cycle figure this project
  has ever reported. It is alive.
- **`arch_ver` is 0x8c75, bit-identical to the simulator**, with `unsigned_pd_support=1`
  and `vtcm_total_bytes=8388608`. An assertion that had never actually been checked.
- **A real defect the simulator structurally could not catch.** FastRPC keeps the two
  caches coherent only for buffers passed as invoke arguments; hexlib's are mapped out
  of band by `fastrpc_mmap` and named by fd, so nothing wrote them back. `--self-test`
  returned 3859/4100 values not bit-exact and the encoder's last op read all zero. It is
  *ordered* corruption — early writes landed, the final op's 1024 bytes never left the
  cache — which is what identified it, because random corruption does not sort itself by
  age. Fixed with `qurt_mem_cache_clean`: **invalidate before and flush after**. Flush
  alone works for exactly one invoke per session and then silently computes on old data.

**Device cycle counts from before that fix are still valid** — PCYCLE is a register read.
Device *data* from before it is not.

Reaching a device at all needs three non-obvious things, none of them in the SDK
signature: the SSH key must be the one **QDC** issued rather than your own,
`session_parameters=[SSHONLY]` is what provisions SSH at all, and what you get back is
an **adb tunnel**, not a shell — nothing runs remotely, everything goes through a local
`adb -P <port>`. Sessions bill for the whole timeout, not for what you use.

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

# hexlib

A kernel library and programming model for Qualcomm Hexagon NPUs. hexlib gives
developers callable, verified, cycle-measured kernels, and gives kernel authors a
documented way to write and run their own, targeting a single NSP (v75) first.

**The Hexagon SDK is required to use hexlib at all** — not just to contribute to it.
Every command below that touches a kernel (`hexlib test`, and anything that compiles
or simulates) needs `hexagon-clang` and `hexagon-sim` from the SDK, discovered through
`HEXAGON_SDK_ROOT`. hexlib never vendors, bundles, or fetches the SDK; it is
license-restricted and you obtain it yourself. If you don't have it, you can still
read the docs, write scalar reference implementations, and design `spec.json` files
(see `CONTRIBUTING.md`), but you cannot build or run a kernel.

## What v1 covers

This is the **simulation path**. Kernels compile with `hexagon-clang` and run on
`hexagon-sim`; correctness, cycle counts, and HVX/HMX use (proven from the compiled
ELF, not from source text or a self-reported flag) all come from the simulator.
**Device backends (`--device local`, `--device qdc`) and silicon validation (gate 6)
are not implemented in this plan** — they arrive with the silicon-path plan. Nothing
here should be read as a working device pipeline; `hexlib test --device local` and
`--device qdc` currently just print that the backend isn't implemented yet.

## Quickstart

```bash
pip install -e .
hexlib new-kernel my_kernel        # scaffolds kernels/my_kernel
hexlib test kernels/my_kernel       # builds it, simulates it, prints a gate table
```

`hexlib new-kernel` writes a conforming, empty kernel directory (`kernel_api.h`,
`baseline.c`, `harness.c`, a `nearmiss_*.c` stub, `spec.json`, `README.md`). Fill in
the contract and the scalar reference, then write the kernel. `hexlib test` builds it
against the SDK, runs it on the simulator, disassembles the ELF to prove HVX/HMX use,
confirms the near-miss variant is still rejected, and writes a result table you attach
to your PR. See `CONTRIBUTING.md` for the full gate sequence.

## Worked example: rmsnorm_fp16

The one kernel shipped so far. RMSNorm with a per-column gain, row-wise, fp16;
shape `R=8, C=128, eps=1e-5`. Full record in
[`kernels/rmsnorm_fp16/BAKEOFF.md`](kernels/rmsnorm_fp16/BAKEOFF.md) and
[`kernels/rmsnorm_fp16/RESULT.md`](kernels/rmsnorm_fp16/RESULT.md).

| gate | result |
|---|---|
| correct | PASS |
| kernel_cycles | 2021 |
| accel (ELF-proven) | hvx, hvx-compute |
| near-miss `nearmiss_mean_not_rms.c` | correctly rejected |
| near-miss `nearmiss_no_eps.c` | correctly rejected |
| **gate** | **PASS** |

target `v75` · toolchain `19.0.04`

The winning candidate (an adapted v6 `rmsnorm_gain_fp16`) measured **2021 kernel
cycles** against a **69443**-cycle scalar baseline — **34.36x** — and beat the other
HVX candidate measured for this kernel (an adapted v6 `fp16_rmsnorm`, 10498 cycles) by
5.19x. These are numbers from `hexagon-sim` under a pinned bus model
(`--timing --buspenalty 75 --busratio 2`), not silicon measurements — see
[`docs/hardware/simulator-accuracy.md`](docs/hardware/simulator-accuracy.md) for what
the simulator does and does not guarantee. `kernel_cycles` is the DSP-side cycle count
for the kernel call alone; never compare whole-program `cycles`, which includes
155k-190k cycles of roughly constant harness/CRT overhead.

## Repository layout

```
hexlib/            the CLI and verification pipeline (new-kernel, validate, test)
include/hexlib/    DSP-side headers a kernel #includes, including vendored HVX math
kernels/           one self-contained directory per promoted kernel
docs/hardware/     measured hardware notes (HMX int8, simulator accuracy)
```

## Documentation

- [`CONTRIBUTING.md`](CONTRIBUTING.md) — the three tiers, the six gates, the
  bake-off, and what CI does and does not check.
- [`ROADMAP.md`](ROADMAP.md) — the op backlog, with status and tier, so tier-0 work
  (no SDK needed) is always visible.
- [`ATTRIBUTION.md`](ATTRIBUTION.md) — every vendored source, its license, and the
  commit it came from.
- [`docs/hardware/hmx-int8.md`](docs/hardware/hmx-int8.md) — the measured HMX int8
  MAC sequence.
- [`docs/hardware/simulator-accuracy.md`](docs/hardware/simulator-accuracy.md) — what
  "cycle-approximate" means and where the simulator is most likely to drift from
  silicon.
- [`docs/research/oracle-provenance.md`](docs/research/oracle-provenance.md) — what the
  committed vision-encoder golden vectors prove, and what they do not.

## License

[MIT](LICENSE). hexlib also vendors MIT-licensed code from llama.cpp's ggml-hexagon
backend, which carries its own attribution requirement — see
[`ATTRIBUTION.md`](ATTRIBUTION.md).

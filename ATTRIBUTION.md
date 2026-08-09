# Attribution

## hexlib is MIT licensed

See [`LICENSE`](LICENSE). MIT was chosen over Apache-2.0 for compatibility with the
code hexlib vendors — llama.cpp's ggml-hexagon headers are MIT, and matching licenses
keeps contributions able to flow back upstream, which is a stated goal of the project.
The trade accepted in doing so is that MIT carries no explicit patent grant.

hexlib **vendors MIT-licensed code, which requires attribution independently of
hexlib's own license.** MIT's attribution requirement travels with the code. The rest
of this document is that attribution.

## Vendored: ggml-hexagon HVX headers (MIT)

**Source:** `ggml/src/ggml-hexagon/htp/` in
[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp).
**License:** MIT. **Copyright:** (c) 2023-2026 The ggml authors.
**Upstream commit:** `6a32c29a746a2e44de463de647f9f6661eb5086b` (2026-08-06).
**Copied into:** `include/hexlib/hvx/` (verbatim, unmodified).

These are the HVX primitive math headers — `hvx-types.h`, `hvx-base.h`,
`hvx-utils.h`, `hvx-arith.h`, `hvx-copy.h`, `hvx-reduce.h`, `hvx-norm.h`,
`hvx-exp.h`, `hvx-log.h`, `hvx-sqrt.h`, `hvx-inverse.h`, `hvx-div.h`,
`hvx-scale.h`, `hvx-sigmoid.h`, `hvx-sin-cos.h`, `hvx-pow.h`, `hvx-floor.h`,
`hvx-repl.h` — plus the supporting `hex-utils.h`, `hex-fastdiv.h`, `hex-dump.h`,
`hex-common.h` needed to satisfy their internal includes. The full record, including
why these specific headers (already ggml-free, proven on real v75 silicon) and the
re-sync procedure, lives in `include/hexlib/hvx/README.md` — this section agrees with
that record and must be kept in sync with it if the vendored commit ever changes.

These headers are copied, not imported: hexlib has no build or runtime dependency on
llama.cpp or ggml, and none of the copied headers reference ggml (verified by
`hexlib/tests/test_vendored_headers.py`, which fails the build if a `ggml` reference
appears in any of them, or if the vendored directory is empty).

**Deferred to plan 2 (the tile DSL / v2 spec):** ggml-hexagon's IDL, host driver
(`htp-drv.cpp`), and CMake toolchain file are also planned to be copied under this
same MIT attribution, once the DSL and device runtime work that needs them begins.
Nothing under those categories has been copied yet in this plan; when it is, this
document must be updated alongside it.

## Adapted, not vendored: hexbench (same author)

hexlib's toolchain discovery (`hexlib/toolchain.py`) and ELF anti-cheat logic
(`hexlib/anticheat.py`) are adapted from the same author's `hexbench` project
(`hexbench/env/toolchain.py` and `hexbench/env/anticheat.py`). Both files say so in
their module docstrings. This is the same author attributing their own prior work to
itself, not a third-party license obligation — there is no separate hexbench license
notice to reproduce here, and no hexbench copyright to attribute beyond the author's
own.

**hexlib has no dependency on hexbench.** Nothing in this repository imports
`hexbench` at build time or run time; the adaptation is source-level (copied,
generalized, and rewritten where hexlib's contract differs), and there is no
`import hexbench` anywhere in the codebase. hexbench itself, and the v6/forge2
kernel corpora it manages, remain outside this repository as a private quarry —
hexlib contains only kernels that have individually cleared the six gates in
`CONTRIBUTING.md`, not a bulk import.

## The measured HMX int8 sequence

`docs/hardware/hmx-int8.md` records a measured HMX int8 multiply-accumulate
instruction sequence. That knowledge — which packet structure and tile masks produce
a genuine accumulate rather than a silently-cleared one — was originally established
as documentation (not code) in this project's own prior work, and is reproduced here
as documentation, the same way this repository already treats hardware manuals and
vendor headers as sources of fact rather than sources of code. No kernel, harness, or
committed source in this repository derives from it; only the knowledge of which
instruction sequence to test does.

## The Hexagon SDK is never vendored

The Hexagon SDK (`hexagon-clang`, `hexagon-sim`, the qurt and FastRPC headers) is
license-restricted and is **never vendored, bundled, or fetched** by this repository,
by CI, or by any script in it. It is discovered at build time through
`HEXAGON_SDK_ROOT`, which every contributor and every user supplies for themselves.
`.gitignore` guards against an accidental copy of SDK directories landing in the
repository. See the SDK requirement stated on the first screen of `README.md` and
`CONTRIBUTING.md`.

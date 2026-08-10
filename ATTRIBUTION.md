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

## Adapted: ggml-hexagon's FastRPC runtime (MIT)

**Source:** `ggml/src/ggml-hexagon/` in
[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp).
**License:** MIT. **Copyright:** (c) 2023-2026 The ggml authors.
**Upstream commit:** `6a32c29a746a2e44de463de647f9f6661eb5086b` (2026-08-06).

hexlib's silicon-path runtime (`hexlib/runtime/`) is **adapted** from this
backend — rewritten in hexlib's own tree, not copied verbatim. What was adapted,
and from where:

| hexlib | upstream | what was taken |
|---|---|---|
| `runtime/idl/hexlib_iface.idl` | `htp/htp_iface.idl` | the session lifecycle: `start`, `stop`, `mmap`, `munmap`, `hwinfo` |
| `runtime/host/driver.c` | `htp-drv.cpp` | dlopen/dlsym of `libcdsprpc`, so a missing driver is a message rather than a loader failure |
| `runtime/skel/skel_bufs.c` | `htp/main.c` `reuse_buf`/`mmap_buf`/`prep_tensor` | fd→base mmap caching, and the **(buffer index, offset)** tensor addressing that keeps host addresses off the wire |
| `runtime/skel/skel_vtcm.c` | `htp/main.c` `vtcm_acquire`/`vtcm_alloc` | `HAP_compute_res_*` acquisition with a release callback |
| `runtime/skel/hexlib_dsp.h` | `htp/htp-ops.h` | the batch descriptor SHAPE, and `htp_status`'s "OK is 1, not 0" |
| `runtime/skel/skel.c` | `htp/main.c` session entry points | the `open`/`close`/`start`/`stop`/`mmap`/`munmap`/`hwinfo` lifecycle qaic's skel dispatches to; `invoke` is hexlib's own (a single opaque batch, not a dspqueue packet per op) |

**Deliberately not adapted:** `dspqueue` dispatch (`htp_main_thread`,
`htp_packet_callback`, `process_opbatch`), because it has no simulator path;
`htp_tensor`'s `ne`/`nb` strides, because hexlib uses an enumerated layout; and
the ggml opcode enum.

**One upstream defect is fixed rather than carried over:** `mmap_buf` returns
silently with `base == 0` when all mmap slots are occupied, after which
`prep_tensor` computes `0 + offset` and the kernel reads or writes a small bogus
address; it also `abort()`s on a failed mapping. hexlib returns
`HEXLIB_DSP_ERR_NO_MMAP_SLOT` / `HEXLIB_DSP_ERR_MMAP_FAILED` and runs no op.

This is **adapted, not vendored** — unlike `include/hexlib/hvx/`, which is
byte-identical upstream and must never be edited in place. hexlib still has no
build or runtime dependency on llama.cpp or ggml.

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

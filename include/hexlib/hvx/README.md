# Vendored HVX primitive headers

Copied verbatim from `ggml/src/ggml-hexagon/htp/` in llama.cpp
(<https://github.com/ggml-org/llama.cpp>), **MIT licensed**, Copyright (c)
2023-2026 The ggml authors.

**Upstream commit:** `6a32c29a746a2e44de463de647f9f6661eb5086b` (2026-08-06)

These are copied, not imported: hexlib has no dependency on llama.cpp or ggml.
They were chosen because they are already ggml-free — every `*-ops.c` file in
that directory references ggml and none of these headers does — and because they
are proven on real v75 silicon, which is not a property that can be re-earned
cheaply.

**Do not edit these in place.** A local fix here silently diverges from upstream
and makes the next re-sync a merge. Fix it upstream, or wrap it in a hexlib
header alongside.

Re-sync procedure: re-copy at a newer commit, update the hash above and in
`ATTRIBUTION.md`, and re-run every kernel's gate — these headers are on the
critical path of every cycle number in the repository.

## Copied files

The following headers were copied:
- hvx-types.h
- hvx-base.h
- hvx-utils.h
- hvx-arith.h
- hvx-copy.h
- hvx-reduce.h
- hvx-norm.h
- hvx-exp.h
- hvx-log.h
- hvx-sqrt.h
- hvx-inverse.h
- hvx-div.h
- hvx-scale.h
- hvx-sigmoid.h
- hvx-sin-cos.h
- hvx-pow.h
- hvx-floor.h
- hvx-repl.h

Additionally, the following supporting hex-*.h headers were copied to satisfy
internal dependencies:
- hex-utils.h
- hex-fastdiv.h
- hex-dump.h
- hex-common.h

# rmsnorm_fp16

RMSNorm with a per-column gain, row-wise, fp16. This is hexlib's first promoted
kernel — see `BAKEOFF.md` for the measured candidates and `RESULT.md` for the
gate output.

## Contract

For each of `R` independent rows of length `C` (`C` a multiple of 64, the fp16
lane count of one 128B HVX vector):

```
ms        = (1/C) * sum_c x[r][c]^2          (accumulated in float)
inv       = 1 / sqrt(ms + eps)
y[r][c]   = (fp16) ( (float) x[r][c] * inv * (float) w[c] )
```

`x`, `w`, `y` are `__fp16`; `w` is length `C` and shared across every row (a
per-column gain, not a per-row one). The reduction and the reciprocal square
root are computed in float; only the stored result is rounded to fp16. HVX
float goes through the non-IEEE qf16 path and float operations reorder, so
results are compared with `hexlib_close_f16`, never bit-exactly.

## Why it's fast

The winning implementation finishes the sum-of-squares horizontal reduction
entirely inside the HVX vector unit with a six-step ror-shift butterfly,
rather than unpacking the accumulator to memory and finishing with a scalar
loop. See the header comment in `kernel.c` and the "Why it wins" section of
`BAKEOFF.md` for the full mechanism and why it beats the alternative at this
kernel's R=8 batch shape.

## Files

- `kernel_api.h` — the contract.
- `baseline.c` — scalar reference (`rmsnorm_fp16_baseline`), compiled and
  linked as a third source alongside `<impl>.c` and `harness.c`
  (`hexlib/build.py`'s `build_kernel`), which calls it for its reference.
- `harness.c` — builds inputs, runs the baseline for reference, times only the
  kernel call, and prints the verdict the driver parses.
- `nearmiss_no_eps.c` — plausible bug: `eps` dropped from the denominator.
- `nearmiss_mean_not_rms.c` — plausible bug: normalizing by the mean of `x`
  instead of the root-mean-square (the LayerNorm/RMSNorm confusion).
- `kernel.c` — the promoted kernel.
- `spec.json` — machine-readable spec.
- `BAKEOFF.md` — every candidate measured, including the ones that lost.
- `RESULT.md` — the gate output for the promoted kernel, copied from
  `_work/rmsnorm_fp16.result.md`.

## Reproduce

```bash
hexlib test kernels/rmsnorm_fp16
```

Expected: `correct PASS`, both near-misses `correctly rejected`,
`accel (ELF-proven) hvx, hvx-compute`, gate `PASS`.

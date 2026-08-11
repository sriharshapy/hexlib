# hexlib/tests/test_kernels.py
"""Source-level regression guards for shipped kernels.

These are not a substitute for the simulator gate (`hexlib test`, run
separately, requires the SDK) -- they guard against specific defects
reintroducing themselves in source review, where a runtime repro is
impractical (the defect below is a static out-of-bounds read that the
standalone Hexagon simulator has no sanitizer to catch, and which is
unreachable under every conforming call anyway).

THE SLICER IS SHARED, NOT COPIED. This file carried a fourth private copy of
the brace-counting slicer `hexlib/tests/csource.py` exists to consolidate, and
the weakest of the four: `src.index("void " + name + "(")` with no comment
handling at all. That mattered here specifically, because the assertion below
is `"else" not in body` -- a NEGATIVE check, which any comment inside
rmsnorm_fp16 containing the word "else" (or "otherwise... else", or a prose
mention of the removed branch) would trip, and which the comment-blanked slice
from `csource.function_body` cannot be tripped by. Migrated.
"""
from __future__ import annotations

import pathlib
import re

from hexlib.tests.csource import function_body as _function_body

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RMSNORM_KERNEL_C = REPO_ROOT / "kernels" / "rmsnorm_fp16" / "kernel.c"


def test_rmsnorm_fp16_has_no_dead_and_unsafe_else_branch():
    """Ledger #3 (final whole-branch review): `if (nb > 0) {...} else {
    accSq = Q6_Vqf16_vmpy_VhfVhf(xv[0], xv[0]); }` inside the row loop was
    dead under every conforming call (kernel_api.h requires C to be a
    multiple of 64, so nb = C/64 is never <= 0) but unsafe if ever reached: a
    full 128-byte HVX vector read past the end of a row shorter than one
    vector. The fix hoists the `nb <= 0` guard above the row loop as an early
    `return` and removes the `else`."""
    body = _function_body(RMSNORM_KERNEL_C.read_text(encoding="utf-8"), "rmsnorm_fp16")
    assert "else" not in body, (
        "the dead-and-unsafe else branch reading xv[0] for nb<=0 is back"
    )
    assert re.search(r"if\s*\(\s*nb\s*<=\s*0\s*\)", body), (
        "the nb<=0 guard (now an early return) is missing"
    )

"""The kernel directory: hexlib's unit of contribution, review, and verification.

A kernel is a directory, not a file. It carries its own scalar baseline, its own
harness, its own near-miss variants, and a spec — so a contribution is one new
directory, CI can check it mechanically, and a reviewer reads a generated table
instead of four hundred lines of intrinsics.

The near-miss requirement is deliberate and unusual: a PR must include at least
one plausible-but-wrong variant that the harness catches. A harness that passes
the kernel proves nothing until it is also shown to FAIL something close by.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any

REQUIRED_FILES: tuple[str, ...] = (
    "kernel.c",
    "kernel_api.h",
    "baseline.c",
    "harness.c",
    "spec.json",
)

KNOWN_CAPS = frozenset({"hmx"})
KNOWN_MECHANISMS = frozenset({"hvx", "hmx", "dma", "vtcm", "l2fetch", "scalar"})

EXACT_TOLERANCE = "exact"

# Any of these appearing anywhere in a dtype string means some part of the
# pipeline is floating point. Substring matching is deliberate: dtypes in the
# wild look like "uint8xint8->int32" and "int8->fp16", not like clean enums.
_FLOAT_DTYPE_TOKENS = ("fp16", "fp32", "f16", "f32", "float", "half", "bf16")


def is_integer_dtype(dtype: str) -> bool:
    """True when no stage of the dtype pipeline is floating point.

    Tolerance comparison exists because HVX float is the non-IEEE qf16 path and
    float operations reorder, so an fp16 result cannot be compared bit-exactly.
    An integer path has neither property -- there is exactly one right answer,
    byte for byte -- so a tolerance there hides wrong results instead of
    accommodating the hardware.
    """
    return not any(tok in dtype.lower() for tok in _FLOAT_DTYPE_TOKENS)


@dataclass(frozen=True)
class KernelSpec:
    task_id: str
    dtype: str
    caps: list[str] = field(default_factory=list)
    mechanisms: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    expert_kernel_cycles: int | None = None
    tolerance: str = "exact"
    tags: list[str] = field(default_factory=list)


def load_spec(kernel_dir: str) -> KernelSpec:
    path = os.path.join(kernel_dir, "spec.json")
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"{path} does not exist")
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON: {e}")
    if "task_id" not in raw or "dtype" not in raw:
        raise ValueError(f"{path} must contain 'task_id' and 'dtype'")
    known = {f.name for f in KernelSpec.__dataclass_fields__.values()}
    return KernelSpec(**{k: v for k, v in raw.items() if k in known})


def nearmiss_files(kernel_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(kernel_dir, "nearmiss_*.c")))


def validate_dir(kernel_dir: str) -> list[str]:
    """Return a list of problems. Empty means valid. Never raises."""
    problems: list[str] = []

    if not os.path.isdir(kernel_dir):
        return [f"{kernel_dir} is not a directory"]

    for name in REQUIRED_FILES:
        if not os.path.isfile(os.path.join(kernel_dir, name)):
            problems.append(f"missing required file: {name}")

    if not nearmiss_files(kernel_dir):
        problems.append(
            "missing near-miss: at least one nearmiss_*.c is required, so the "
            "harness is shown to reject a plausible wrong implementation"
        )

    try:
        spec = load_spec(kernel_dir)
    except ValueError as e:
        problems.append(str(e))
        return problems

    expected_id = os.path.basename(os.path.normpath(kernel_dir))
    if spec.task_id != expected_id:
        problems.append(
            f"spec.json task_id is {spec.task_id!r} but the directory is "
            f"{expected_id!r}; they must match"
        )
    for cap in spec.caps:
        if cap not in KNOWN_CAPS:
            problems.append(f"unknown cap {cap!r}; known caps: {sorted(KNOWN_CAPS)}")
    for mech in spec.mechanisms:
        if mech not in KNOWN_MECHANISMS:
            problems.append(
                f"unknown mechanism {mech!r}; known: {sorted(KNOWN_MECHANISMS)}"
            )
    # A malformed dtype is its own problem, and must be reported rather than
    # skipped: `is_integer_dtype(None)` would raise on `.lower()`, and silently
    # passing over it would let a spec with dtype 123 validate completely clean.
    if not isinstance(spec.dtype, str) or not spec.dtype:
        problems.append(
            f"spec.json dtype must be a non-empty string, got {spec.dtype!r}"
        )
    elif is_integer_dtype(spec.dtype) and spec.tolerance != EXACT_TOLERANCE:
        problems.append(
            f"dtype {spec.dtype!r} is an integer pipeline, so tolerance must be "
            f"{EXACT_TOLERANCE!r} (bit-exact), not {spec.tolerance!r}. Integer "
            "paths have no reordering and no representation error: a tolerance "
            "there hides wrong results rather than accommodating the hardware."
        )
    return problems


_SPEC_TEMPLATE = {
    "task_id": "",
    "dtype": "fp16",
    "caps": [],
    "mechanisms": ["hvx"],
    "params": {},
    "expert_kernel_cycles": None,
    "tolerance": "hexlib_close_f16",
    "tags": [],
}

_STUBS = {
    "kernel.c": (
        '#include "kernel_api.h"\n\n'
        "/* Your kernel. Kernels are GNU C: no name mangling, so the harness\n"
        " * links against exactly the symbol you declare here. */\n"
        "void {name}(void) {{\n}}\n"
    ),
    "kernel_api.h": (
        "#ifndef HEXLIB_KERNEL_API_H\n#define HEXLIB_KERNEL_API_H\n"
        "typedef __fp16 hexlib_hf;\n\n"
        "/* Document the exact mathematical contract here, including shapes,\n"
        " * dtypes, and where the reference rounds. */\n"
        "void {name}(void);\n"
        "#endif\n"
    ),
    "baseline.c": (
        '#include "kernel_api.h"\n\n'
        "/* Scalar reference. Correct and obvious, never fast. */\n"
        "void {name}_baseline(void) {{\n}}\n"
    ),
    "harness.c": (
        '#include "hexlib/hexlib_harness.h"\n#include "kernel_api.h"\n\n'
        "int main(void) {{\n    return 0;\n}}\n"
    ),
    "nearmiss_plausible_bug.c": (
        '#include "kernel_api.h"\n\n'
        "/* A plausible WRONG implementation the harness must reject. Model it on a\n"
        " * real mistake — a skipped rescale, a wrong axis, a missing epsilon. */\n"
        "void {name}(void) {{\n}}\n"
    ),
    "README.md": "# {name}\n\nWhat this kernel computes, and why it is fast.\n",
}


def scaffold(kernels_root: str, name: str) -> str:
    """Create a conforming kernel directory. Refuses to overwrite."""
    target = os.path.join(kernels_root, name)
    if os.path.exists(target):
        raise FileExistsError(f"{target} already exists")
    os.makedirs(target)

    for filename, template in _STUBS.items():
        with open(os.path.join(target, filename), "w", encoding="utf-8") as f:
            f.write(template.format(name=name))

    spec = dict(_SPEC_TEMPLATE)
    spec["task_id"] = name
    with open(os.path.join(target, "spec.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
        f.write("\n")
    return target

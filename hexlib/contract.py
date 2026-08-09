# hexlib/contract.py
"""What CI can enforce without a Hexagon SDK.

CI cannot re-run the gate — the SDK is license-restricted and is never installed
on a hosted runner. What CI CAN do is refuse a pull request that claims a pass it
did not attach: a kernel directory must carry a RESULT.md produced by
`hexlib test`, it must parse, it must record the pinned toolchain, and its gate
line must say PASS.

That does not make a forged table impossible. It makes an ACCIDENTAL green
impossible, and it makes a deliberate one a visible, dated, attributed claim that
any maintainer with an SDK can reproduce and disprove.
"""
from __future__ import annotations

import os
import re

from hexlib import kerneldir as kd
from hexlib import toolchain as tc

RESULT_FILENAME = "RESULT.md"

_CONDITIONS = re.compile(
    r"target\s+`(?P<target>[^`]+)`\s*·\s*toolchain\s+`(?P<toolchain>[^`]+)`\s*·\s*"
    r"SDK\s+`(?P<sdk>[^`]+)`\s*·\s*host\s+`(?P<host>[^`]+)`\s*·\s*`(?P<ts>[^`]+)`"
)
_GATE = re.compile(r"\|\s*\*\*gate\*\*\s*\|\s*\*\*(PASS|FAIL)\*\*\s*\|")


def parse_result_table(text: str) -> dict | None:
    """Extract the conditions and the gate verdict. None if either is absent."""
    cond = _CONDITIONS.search(text)
    gate = _GATE.search(text)
    if not cond or not gate:
        return None
    d = cond.groupdict()
    d["gate"] = gate.group(1)
    return d


def check_kernel_contract(kernel_dir: str) -> list[str]:
    """Everything CI can check about a kernel directory without an SDK."""
    problems = list(kd.validate_dir(kernel_dir))

    result_path = os.path.join(kernel_dir, RESULT_FILENAME)
    if not os.path.isfile(result_path):
        problems.append(
            f"missing {RESULT_FILENAME}: run `hexlib test {kernel_dir}` locally and "
            "commit the result table. CI has no Hexagon SDK and cannot produce it."
        )
        return problems

    # errors="replace", not the default strict decoding: a RESULT.md that is
    # binary garbage must still produce a "could not be parsed" problem, not an
    # unhandled UnicodeDecodeError. check_kernel_contract must never raise.
    with open(result_path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    parsed = parse_result_table(text)
    if parsed is None:
        problems.append(
            f"{RESULT_FILENAME} could not be parsed: it must be the table emitted "
            "by `hexlib test`, including its trailing conditions line and gate row."
        )
        return problems

    if parsed["toolchain"] != tc.TOOLCHAIN_VERSION:
        problems.append(
            f"{RESULT_FILENAME} records toolchain {parsed['toolchain']} but this "
            f"repository pins {tc.TOOLCHAIN_VERSION}. Cycle numbers from a "
            "different toolchain are not comparable."
        )

    if parsed["target"] != tc.DSP_ARCH:
        problems.append(
            f"{RESULT_FILENAME} records target {parsed['target']}, expected "
            f"{tc.DSP_ARCH}"
        )

    if parsed["gate"] != "PASS":
        problems.append(
            f"{RESULT_FILENAME} records gate FAIL; a kernel cannot land until its "
            "gate passes locally"
        )

    return problems

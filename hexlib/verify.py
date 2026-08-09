# hexlib/verify.py
"""The simulation gate: build, simulate, prove acceleration, and confirm the
near-misses still fail.

WHY NEAR-MISSES ARE PART OF THE GATE. A harness that accepts the kernel proves
nothing until it is also shown to REJECT something close by. Every near-miss
must fail; one that passes means the test does not discriminate, and the
kernel's own pass is worthless.

The report is the artifact a contributor pastes into a pull request, so it
states the conditions it was produced under — toolchain, SDK, host, timestamp —
and never describes a simulation number as silicon-validated.
"""
from __future__ import annotations

import getpass
import json
import os
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from hexlib import kerneldir as kd
from hexlib import toolchain as tc
from hexlib.anticheat import AccelProof, prove_accel
from hexlib.build import BuildError, build_kernel
from hexlib.result import Err, Measurements, Ok, Result
from hexlib.sim import SimError, run_sim

# A near-miss outcome. Three states, not two, because "the near-miss did not
# run" is not evidence of anything and must never be counted as rejection.
NEARMISS_REJECTED = "rejected"       # built, ran, harness marked it INCORRECT
NEARMISS_ACCEPTED = "accepted"       # built, ran, harness marked it CORRECT
NEARMISS_INCONCLUSIVE = "inconclusive"  # never got a verdict; prefix of a reason


@dataclass(frozen=True)
class VerifyReport:
    task_id: str
    correct: bool
    kernel_cycles: int
    accel: AccelProof
    nearmiss: dict[str, str]  # filename -> NEARMISS_* (or an inconclusive reason)
    toolchain_version: str
    sdk_version: str
    host: str
    timestamp: str
    expert_kernel_cycles: int | None = None

    def gate_passed(self) -> bool:
        if not self.correct:
            return False
        # Load-only HVX is not acceleration: bytes moved through the vector unit
        # while the arithmetic stayed in scalar registers.
        if not (self.accel.used_hvx_compute or self.accel.used_hmx):
            return False
        # No near-misses means the harness was never shown to discriminate.
        # validate_dir already requires one, but gate_passed must not depend on
        # a caller having run that check -- all({}) is True, which would be a
        # silent pass.
        if not self.nearmiss:
            return False
        # Every near-miss must have BUILT, RUN, and been rejected. An
        # inconclusive one proves nothing: a near-miss that fails to compile
        # because of an unrelated typo was never offered to the harness at all,
        # so counting it as rejection would let a broken near-miss green the
        # gate -- the exact failure this mechanism exists to prevent.
        return all(v == NEARMISS_REJECTED for v in self.nearmiss.values())

    def to_table(self) -> str:
        speedup = ""
        if self.expert_kernel_cycles and self.kernel_cycles:
            ratio = self.expert_kernel_cycles / self.kernel_cycles
            speedup = f" ({ratio:.2f}x vs recorded {self.expert_kernel_cycles})"
        mechs = [
            n
            for n, v in (
                ("hvx", self.accel.used_hvx),
                ("hvx-compute", self.accel.used_hvx_compute),
                ("hmx", self.accel.used_hmx),
            )
            if v
        ]
        lines = [
            f"### hexlib verify — {self.task_id}",
            "",
            f"| gate | result |",
            f"|---|---|",
            f"| correct | {'PASS' if self.correct else 'FAIL'} |",
            f"| kernel_cycles | {self.kernel_cycles}{speedup} |",
            f"| accel (ELF-proven) | {', '.join(mechs) if mechs else 'NONE'} |",
        ]
        for name, state in sorted(self.nearmiss.items()):
            if state == NEARMISS_REJECTED:
                shown = "correctly rejected"
            elif state == NEARMISS_ACCEPTED:
                shown = "WRONGLY ACCEPTED"
            else:
                shown = f"INCONCLUSIVE -- {state}"
            lines.append(f"| near-miss `{name}` | {shown} |")
        lines += [
            f"| **gate** | **{'PASS' if self.gate_passed() else 'FAIL'}** |",
            "",
            f"target `{tc.DSP_ARCH}` · toolchain `{self.toolchain_version}` · "
            f"SDK `{self.sdk_version}` · host `{self.host}` · `{self.timestamp}`",
            "",
            "Measured on the hexagon simulator under the pinned bus model "
            f"(buspenalty {tc.BUS_PENALTY}, busratio {tc.BUS_RATIO}). The "
            "simulator is cycle-approximate; these numbers are reproducible, "
            "not silicon measurements.",
        ]
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _sdk_version(sdk_root: str) -> str:
    return os.path.basename(os.path.normpath(sdk_root)) or "unknown"


def verify(kernel_dir: str, out_dir: str, sdk_root: str | None = None) -> Result:
    """Run the whole simulation gate. Returns Ok(Measurements) or Err(reason)."""
    problems = kd.validate_dir(kernel_dir)
    if problems:
        return Err(
            f"{kernel_dir} is not a valid kernel directory",
            "\n".join(f"  - {p}" for p in problems),
        )

    spec = kd.load_spec(kernel_dir)
    root = sdk_root or tc.default_sdk_root()

    try:
        built = build_kernel(kernel_dir, out_dir, spec.caps, sdk_root=root)
    except BuildError as e:
        return Err(f"{spec.task_id}: build failed", e.compiler_output)

    try:
        outcome = run_sim(built, spec.caps)
    except SimError as e:
        return Err(f"{spec.task_id}: simulation failed — {e}", e.sim_output)

    accel = prove_accel(built.obj, built.bin_dir)

    # Every near-miss must build, run, and be REJECTED by the harness.
    #
    # A near-miss that fails to build or never produces a verdict is
    # INCONCLUSIVE, not rejected. It was never offered to the harness, so it
    # demonstrates nothing about whether the harness discriminates -- and
    # counting it as rejection would mean a single typo in a near-miss silently
    # turns the gate green, which is precisely the failure this mechanism
    # exists to catch.
    nearmiss: dict[str, str] = {}
    for path in kd.nearmiss_files(kernel_dir):
        name = os.path.basename(path)
        try:
            nm_built = build_kernel(kernel_dir, out_dir, spec.caps, impl=name,
                                    sdk_root=root)
        except BuildError as e:
            first = (e.compiler_output or str(e)).strip().splitlines()
            nearmiss[name] = (
                f"{NEARMISS_INCONCLUSIVE}: did not build -- "
                f"{first[0] if first else 'no compiler output'}"
            )
            continue
        try:
            nm_outcome = run_sim(nm_built, spec.caps)
        except SimError as e:
            nearmiss[name] = f"{NEARMISS_INCONCLUSIVE}: did not run -- {e}"
            continue
        nearmiss[name] = (
            NEARMISS_ACCEPTED if nm_outcome.correct else NEARMISS_REJECTED
        )

    report = VerifyReport(
        task_id=spec.task_id,
        correct=outcome.correct,
        kernel_cycles=outcome.kernel_cycles,
        accel=accel,
        nearmiss=nearmiss,
        toolchain_version=built.toolchain_version,
        sdk_version=_sdk_version(root),
        host=f"{getpass.getuser()}@{platform.node()}",
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        expert_kernel_cycles=spec.expert_kernel_cycles,
    )

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{spec.task_id}.result.json"), "w",
              encoding="utf-8") as f:
        f.write(report.to_json())
    with open(os.path.join(out_dir, f"{spec.task_id}.result.md"), "w",
              encoding="utf-8") as f:
        f.write(report.to_table() + "\n")

    if not report.gate_passed():
        return Err(f"{spec.task_id}: gate FAILED", report.to_table())

    return Ok(
        Measurements(
            kernel_cycles=report.kernel_cycles,
            toolchain_version=report.toolchain_version,
            sdk_version=report.sdk_version,
            host=report.host,
            timestamp=report.timestamp,
        )
    )

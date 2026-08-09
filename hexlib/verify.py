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


@dataclass(frozen=True)
class VerifyReport:
    task_id: str
    correct: bool
    kernel_cycles: int
    accel: AccelProof
    nearmiss: dict[str, bool]  # filename -> True if it correctly FAILED
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
        return all(self.nearmiss.values())

    def to_table(self) -> str:
        speedup = ""
        if self.expert_kernel_cycles:
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
        for name, rejected in sorted(self.nearmiss.items()):
            lines.append(
                f"| near-miss `{name}` | "
                f"{'correctly rejected' if rejected else 'WRONGLY ACCEPTED'} |"
            )
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

    # Every near-miss must FAIL. A near-miss that builds and passes means the
    # harness does not discriminate.
    nearmiss: dict[str, bool] = {}
    for path in kd.nearmiss_files(kernel_dir):
        name = os.path.basename(path)
        try:
            nm_built = build_kernel(kernel_dir, out_dir, spec.caps, impl=name,
                                    sdk_root=root)
            nm_outcome = run_sim(nm_built, spec.caps)
            nearmiss[name] = not nm_outcome.correct
        except (BuildError, SimError):
            # A near-miss that does not build or run has been rejected, which is
            # the outcome we require of it.
            nearmiss[name] = True

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

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
    n_wrong: int = 0
    max_err: float = 0.0
    movement_only: bool = False
    """This op performs NO arithmetic, so its acceleration is bytes per
    instruction rather than vector arithmetic.

    Set from `spec.json`. It exists because the encoder contains 49 pure layout
    ops -- two transposes and a patchify -- and the default accel rule below was
    written when every kernel in the project was a compute kernel. A transpose
    that moves 128 bytes per instruction instead of 2 IS accelerated, and it has
    no arithmetic to put in a vector register.

    NARROW ON PURPOSE, and it does not turn the check off. A movement-only
    kernel must still prove it used the vector unit (`used_hvx`), so a purely
    scalar implementation still fails the gate. What it drops is only the
    requirement of vector ARITHMETIC, which for this class of op would be
    satisfiable only by adding arithmetic that does not belong there -- gaming
    the check rather than passing it.

    It is declared in the committed spec and printed in the result table, so a
    reviewer sees the claim. A kernel that does have arithmetic and sets this
    flag is a reviewable lie, not a silent one.
    """

    def gate_passed(self) -> bool:
        if not self.correct:
            return False
        if self.movement_only:
            # No arithmetic exists to vectorise; using the vector unit at all is
            # the whole claim, and a scalar implementation still fails here.
            if not self.accel.used_hvx:
                return False
        # Load-only HVX is not acceleration: bytes moved through the vector unit
        # while the arithmetic stayed in scalar registers.
        elif not (self.accel.used_hvx_compute or self.accel.used_hmx):
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
        # Printed, so the weaker accel requirement is visible to a reviewer
        # rather than buried in spec.json.
        movement = " · movement-only (no arithmetic)" if self.movement_only else ""
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
            f"| max abs error | {self.max_err:.6g} (n_wrong {self.n_wrong}) |",
            f"| kernel_cycles | {self.kernel_cycles}{speedup} |",
            f"| accel (ELF-proven) | "
            f"{', '.join(mechs) if mechs else 'NONE'}{movement} |",
        ]
        for name, state in sorted(self.nearmiss.items()):
            if state == NEARMISS_REJECTED:
                shown = "correctly rejected"
            elif state == NEARMISS_ACCEPTED:
                shown = "WRONGLY ACCEPTED"
            else:
                # `state` already carries the "inconclusive: " prefix from
                # NEARMISS_INCONCLUSIVE (e.g. "inconclusive: did not build --
                # ..."), so prefixing "INCONCLUSIVE -- " again would render
                # "INCONCLUSIVE -- inconclusive: did not build -- ...". Strip
                # it before display.
                reason = state.split(":", 1)[1].strip() if ":" in state else state
                shown = f"INCONCLUSIVE -- {reason}"
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

    # Delete any previous run's artifacts before anything that can fail. A
    # result table that outlives the run that produced it will eventually be
    # attached to a different one: a contributor greens the gate, edits
    # kernel.c, re-runs and gets a build failure, and _work/<task>.result.md
    # still holds the earlier PASS with its earlier timestamp.
    os.makedirs(out_dir, exist_ok=True)
    for stale in (
        os.path.join(out_dir, f"{spec.task_id}.result.json"),
        os.path.join(out_dir, f"{spec.task_id}.result.md"),
        os.path.join(kernel_dir, "RESULT.md"),
    ):
        try:
            os.remove(stale)
        except FileNotFoundError:
            pass

    try:
        built = build_kernel(kernel_dir, out_dir, spec.caps, sdk_root=root)
    except BuildError as e:
        return Err(f"{spec.task_id}: build failed", e.compiler_output)
    except (FileNotFoundError, ValueError) as e:
        # Toolchain discovery: no SDK, an incomplete SDK, or an unreadable
        # version path. These raise directly out of build_kernel (before any
        # subprocess is even invoked), so they are not BuildErrors and must be
        # caught here rather than crashing to a raw traceback.
        return Err(f"{spec.task_id}: Hexagon SDK is not usable", str(e))

    try:
        outcome = run_sim(built, spec.caps)
    except SimError as e:
        return Err(f"{spec.task_id}: simulation failed — {e}", e.sim_output)

    accel = prove_accel(built.obj, built.bin_dir)

    # A declared mechanism the ELF does not show is a claim exceeding the
    # measurement -- spec.json is what a future index, search page, or docs
    # generator would read, and nothing else ever compares it against the
    # AccelProof already computed above. Only hvx and hmx are ELF-provable;
    # dma/vtcm/l2fetch/scalar are not, so they are not checked here. Placed
    # before the near-miss loop so a bad claim fails fast without paying for
    # near-miss builds.
    declared = set(spec.mechanisms or [])
    unproven = sorted(
        m for m in ("hvx", "hmx")
        if m in declared
        and not (accel.used_hvx if m == "hvx" else accel.used_hmx)
    )
    if unproven:
        return Err(
            f"{spec.task_id}: spec.json declares mechanisms the ELF does not "
            f"show: {', '.join(unproven)}. Remove the claim from spec.json, or "
            "use the mechanism.",
        )

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
        n_wrong=outcome.n_wrong,
        max_err=outcome.max_err,
        movement_only=bool(getattr(spec, "movement_only", False)),
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

    # Write the contributor-facing artifact into the kernel directory itself,
    # so the file CI validates (contract.py's RESULT_FILENAME) is the file
    # this tool produced, and nobody has to copy a gitignored _work/ path by
    # hand. Only on a pass: a FAIL belongs in _work, not committed next to the
    # kernel.
    with open(os.path.join(kernel_dir, "RESULT.md"), "w", encoding="utf-8") as f:
        f.write(report.to_table() + "\n")

    return Ok(
        Measurements(
            kernel_cycles=report.kernel_cycles,
            toolchain_version=report.toolchain_version,
            sdk_version=report.sdk_version,
            host=report.host,
            timestamp=report.timestamp,
        )
    )

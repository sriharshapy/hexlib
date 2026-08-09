"""Run a built kernel on hexagon-sim and recover a verdict and a cycle count.

FAIL CLOSED. A simulator process that exits 0 having printed neither verdict nor
cycle count is a FAILURE. This is the same defect as QDC job 742504 — job
`completed`, zero results recovered, reported as a pass — and it is prevented
here by having no code path that produces a success without both lines.

ALWAYS kernel_cycles, NEVER whole-program cycles. Harness and CRT overhead is
roughly constant at 155k-190k cycles, so whole-program ratios scale inversely
with kernel size and manufacture 4x-38x differences out of nothing.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from hexlib import toolchain as tc
from hexlib.build import BuildOutput


class SimError(Exception):
    def __init__(self, message: str, sim_output: str = "") -> None:
        super().__init__(message)
        self.sim_output = sim_output


@dataclass(frozen=True)
class SimOutcome:
    correct: bool
    n_wrong: int
    max_err: float
    kernel_cycles: int


_VERDICT = re.compile(
    r"HEXLIB_VERDICT correct=(\d+) wrong=(\d+) maxerr=([0-9eE.+-]+)"
)
_KCYCLES = re.compile(r"HEXLIB_KCYCLES kernel=(\d+)")


def parse_verdict(text: str) -> tuple[bool, int, float] | None:
    m = _VERDICT.search(text)
    if not m:
        return None
    return bool(int(m.group(1))), int(m.group(2)), float(m.group(3))


def parse_kernel_cycles(text: str) -> int | None:
    m = _KCYCLES.search(text)
    return int(m.group(1)) if m else None


def sim_command(sim_exe: str, elf: str, caps: list[str]) -> list[str]:
    """Assemble the simulator command. Pure — assertable without an SDK.

    The bus knobs are the SDK's representative defaults, pinned explicitly so
    numbers stay reproducible across SDK upgrades. They are NOT device-matched,
    which is why simulator output is reproducible rather than silicon-validated.
    """
    cmd = [sim_exe, f"-m{tc.DSP_ARCH}"]
    cmd += tc.sim_flags_for_caps(caps)
    if tc.TIMING_MODE:
        cmd += [
            "--timing",
            "--buspenalty", str(tc.BUS_PENALTY),
            "--busratio", str(tc.BUS_RATIO),
        ]
    cmd.append(elf)
    return cmd


def run_sim(
    build_out: BuildOutput, caps: list[str], timeout: float | None = None
) -> SimOutcome:
    env = tc.toolchain_env(build_out.bin_dir)
    sim_exe = os.path.join(build_out.bin_dir, tc.exe("hexagon-sim"))
    cmd = sim_command(sim_exe, build_out.elf, caps)

    rc, out, err, timed_out = tc.run(
        cmd, env, timeout=timeout or tc.SIM_TIMEOUT_MAX_S
    )
    combined = out + err

    if timed_out:
        raise SimError(
            f"simulator timed out after {timeout or tc.SIM_TIMEOUT_MAX_S}s — "
            "the kernel may not terminate",
            combined,
        )

    verdict = parse_verdict(combined)
    if verdict is None:
        raise SimError(
            "no verdict recovered from the simulator: the harness never printed "
            "HEXLIB_VERDICT, so nothing was actually checked. This is a failure, "
            "not a pass.",
            combined,
        )

    cycles = parse_kernel_cycles(combined)
    if cycles is None:
        raise SimError(
            "no kernel cycle count recovered: the harness never printed "
            "HEXLIB_KCYCLES. A result without measurements is a failure.",
            combined,
        )

    correct, n_wrong, max_err = verdict
    return SimOutcome(
        correct=correct, n_wrong=n_wrong, max_err=max_err, kernel_cycles=cycles
    )

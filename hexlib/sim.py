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


# maxerr is matched as a non-space run and parsed with float(), NOT as a
# character class of digits and exponent punctuation.
#
# WHY. printf("%.9g") emits `inf` and `nan` for a non-finite error, and a
# digits-only class does not match either. The whole verdict line then fails to
# parse, and a kernel that produced inf was reported as "no verdict recovered:
# the harness never printed HEXLIB_VERDICT, so nothing was actually checked" --
# when the harness had printed a perfectly good verdict saying the kernel was
# WRONG. Found by a near-miss that adds fp16 bit patterns as integers, which
# overflows to inf: it was scored INCONCLUSIVE instead of correctly rejected.
#
# It failed safe rather than dangerously -- inconclusive, never a pass -- but it
# misattributed the cause, and a real kernel that overflowed would have been
# reported as not having run rather than as incorrect.
_VERDICT = re.compile(
    r"HEXLIB_VERDICT correct=(\d+) wrong=(\d+) maxerr=(\S+)"
)
_KCYCLES = re.compile(r"HEXLIB_KCYCLES kernel=(\d+)")


def parse_verdict(text: str) -> tuple[bool, int, float] | None:
    m = _VERDICT.search(text)
    if not m:
        return None
    try:
        max_err = float(m.group(3))
    except ValueError:
        # A verdict whose error field is unparseable has still told us
        # correct=/wrong=, and those are what decide the gate. Losing the whole
        # line over an unreadable magnitude is how an inf became "never ran".
        max_err = float("inf")
    return bool(int(m.group(1))), int(m.group(2)), max_err


def parse_kernel_cycles(text: str) -> int | None:
    m = _KCYCLES.search(text)
    return int(m.group(1)) if m else None


def count_verdicts(text: str) -> int:
    return len(_VERDICT.findall(text))


def count_kernel_cycles(text: str) -> int:
    return len(_KCYCLES.findall(text))


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

    if rc != 0:
        raise SimError(
            f"simulator exited {rc}. A run that faulted has not demonstrated "
            "anything, even if the harness managed to flush its verdict first.",
            combined,
        )

    # EXACTLY ONE of each, never merely "at least one". re.search takes the
    # first match and discards the rest: a harness that reports per shape would
    # have every failure after the first silently dropped, and a kernel that
    # prints these lines itself runs BEFORE the harness reports, so its forged
    # verdict would win -- and would keep winning when a maintainer re-ran the
    # gate to check.
    n_verdicts = count_verdicts(combined)
    if n_verdicts > 1:
        raise SimError(
            f"{n_verdicts} HEXLIB_VERDICT lines recovered; exactly one is "
            "required. Only the harness may print it, and only once. Two "
            "verdicts means either a harness reporting per shape (in which "
            "case every failure after the first would be discarded) or output "
            "from the kernel itself.",
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

    n_cycles = count_kernel_cycles(combined)
    if n_cycles > 1:
        raise SimError(
            f"{n_cycles} HEXLIB_KCYCLES lines recovered; exactly one is "
            "required. A run with two cycle counts has no single measurement.",
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

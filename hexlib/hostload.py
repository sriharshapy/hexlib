"""Tell a starved simulator apart from a kernel that never finishes.

WHY THIS EXISTS. `hexlib test` measures WALL time and kills a stage at
SIM_TIMEOUT_MAX_S. Wall time is not a property of the kernel -- it is a
property of the kernel AND whatever else the host happened to be running --
so a timeout on its own says nothing about which of the two caused it. The
gate used to resolve that ambiguity by guessing, and the guess was wrong
twice in one session on matmul_fp16:

  * nearmiss_fp16_accumulate.c genuinely needed 1195s against a 900s ceiling
    and was reported as "the kernel may not terminate". It terminates.
  * nearmiss_wrong_batch_stride.c had ALREADY passed in 825s, then blew past
    1800s on a re-run because a game launched 5 minutes into that stage. Same
    byte-identical ELF, same deterministic input, so the simulated work was
    identical; only the host had changed.

Both were reported with the same words, and neither set of words was true.
An hour went into re-running a gate to discover the second one.

THE MEASUREMENT. Simulated cycles would be the ideal evidence, but the
harness prints HEXLIB_KCYCLES at the END of a run, so a stage that times out
has no cycle count -- and that is exactly the stage in question. What IS
available while the process is alive is its CPU time. hexagon-sim is
single-threaded, so on an unloaded host it accrues very close to one
CPU-second per wall-second. A process that got 0.4 was starved by something
else on the machine; a process that got 0.98 was given everything it asked
for and still did not finish, which IS evidence about the kernel.

WHY NOT SYSTEM-WIDE LOAD. "Is the machine busy" answers a different question
than "was THIS process starved", and answers it worse: it needs the host's
core count to interpret, and it convicts an idle-but-slow run of contention
that never touched it. Per-process CPU share needs no such correction.

psutil IS OPTIONAL. It is not in this project's dependencies and is not
being added for a diagnostic: an import failure degrades to "unmeasured",
which reports the ambiguity honestly instead of inventing a cause. Same
reasoning as hexlib/device/qdc/job.py's lazy SDK import.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

try:  # optional, see module docstring
    import psutil
except Exception:  # pragma: no cover - depends on the environment
    psutil = None

# The simulator executable, matched case-insensitively as a prefix so
# "hexagon-sim" and "hexagon-sim.exe" both hit.
_SIM_PROC_HINT = "hexagon-sim"

# Below this share of one CPU, a single-threaded process was competing for the
# machine rather than using it. Ordinary scheduling noise on an idle host does
# not push a busy single-threaded process under ~0.9; the contended stage that
# motivated this file would have scored far below 0.75.
STARVED_BELOW = 0.75

_SAMPLE_INTERVAL_S = 2.0


@dataclass(frozen=True)
class LoadStats:
    """What a run cost in wall time, and how much CPU it was actually given.

    `cpu_s` is None when nothing could be measured (psutil missing, or the
    process began and ended between two samples). None means UNKNOWN and must
    never be read as either "starved" or "not starved".
    """

    wall_s: float
    cpu_s: float | None
    samples: int

    @property
    def cpu_share(self) -> float | None:
        """CPU-seconds per wall-second. ~1.0 for an unstarved single thread."""
        if self.cpu_s is None or self.wall_s <= 0:
            return None
        return self.cpu_s / self.wall_s

    @property
    def starved(self) -> bool | None:
        """True/False when measured, None when unknown. Tri-state on purpose."""
        share = self.cpu_share
        return None if share is None else share < STARVED_BELOW


class SimLoadMonitor:
    """Sample the simulator child's CPU time for the life of a `with` block.

    Deliberately samples a CHILD DISCOVERED BY PID rather than taking a pid as
    an argument: that keeps toolchain.run's signature and its carefully
    documented decoding behaviour untouched, and keeps every existing caller
    and the tests that monkeypatch it working unchanged.

    A monitor that measures nothing is not an error. `stats()` reports
    cpu_s=None and the caller says so out loud.
    """

    def __init__(
        self,
        proc_hint: str = _SIM_PROC_HINT,
        interval_s: float = _SAMPLE_INTERVAL_S,
    ) -> None:
        self._hint = proc_hint.lower()
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu_s: float | None = None
        self._samples = 0
        self._t0 = 0.0
        self._t1: float | None = None

    def __enter__(self) -> "SimLoadMonitor":
        self._t0 = time.time()
        if psutil is not None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._t1 = time.time()
        self._stop.set()
        if self._thread is not None:
            # The sampler only reads counters, so it cannot block on anything
            # slow; a short join keeps a wedged thread from holding up a gate.
            self._thread.join(timeout=self._interval * 2)

    def _matching_child(self):
        me = psutil.Process(os.getpid())
        for child in me.children(recursive=True):
            try:
                if child.name().lower().startswith(self._hint):
                    return child
            except psutil.Error:
                continue
        return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                child = self._matching_child()
                if child is not None:
                    t = child.cpu_times()
                    # Cumulative and monotonic, so the last reading before the
                    # process dies is the total. max() guards against reading a
                    # DIFFERENT, younger hexagon-sim after a restart.
                    total = float(t.user) + float(t.system)
                    self._cpu_s = total if self._cpu_s is None else max(self._cpu_s, total)
                    self._samples += 1
            except Exception:
                # Sampling is diagnostic. It must never be the reason a gate
                # fails, so every error here degrades to "unmeasured".
                pass
            self._stop.wait(self._interval)

    def stats(self) -> LoadStats:
        end = self._t1 if self._t1 is not None else time.time()
        return LoadStats(
            wall_s=end - self._t0, cpu_s=self._cpu_s, samples=self._samples
        )


def timeout_diagnosis(timeout_s: float, stats: LoadStats) -> str:
    """Explain a timeout in terms of what was measured, never by guessing.

    Pure: takes the numbers, returns the sentence. The three branches are the
    three things that can actually be true, and the unmeasured branch says so
    rather than defaulting to blaming the kernel.
    """
    share = stats.cpu_share
    if share is None:
        return (
            "the simulator's CPU share could not be measured, so this timeout "
            "does NOT distinguish a kernel that never terminates from a host "
            "too busy to finish one that does. Re-run on an idle host before "
            "treating it as a kernel defect."
        )
    if share < STARVED_BELOW:
        return (
            f"the simulator was given only {share:.0%} of one CPU "
            f"({stats.cpu_s:.0f} CPU-seconds over {stats.wall_s:.0f} wall-"
            f"seconds) — the HOST was contended. This is not evidence that "
            f"the kernel fails to terminate; wall-clock budgets measure the "
            f"machine as much as the code. Re-run on an idle host."
        )
    return (
        f"the simulator was given {share:.0%} of one CPU "
        f"({stats.cpu_s:.0f} CPU-seconds over {stats.wall_s:.0f} wall-seconds), "
        f"so it was not DESCHEDULED. That is weaker than 'the host was idle': a "
        f"process keeps accruing a full CPU-second per wall-second while losing "
        f"badly to memory-bandwidth contention, an SMT sibling, or cache "
        f"pressure. Measured 2026-08-13: a byte-identical near-miss ELF ran "
        f"1218s once and exceeded 1800s twice at ~99% share. So this points at "
        f"the kernel or the shape WITHOUT ruling out a loaded host -- check what "
        f"else was running before concluding the code is at fault."
    )

"""A timeout must be attributed from measurement, never from a guess.

These are the cases that actually occurred on matmul_fp16 in one session:
a stage that was genuinely over budget on an idle host, and a stage that had
already passed and only failed because a game was launched while it ran.
"""
import pytest

from hexlib import hostload
from hexlib.hostload import LoadStats


def test_cpu_share_is_cpu_seconds_per_wall_second():
    assert LoadStats(wall_s=100.0, cpu_s=98.0, samples=50).cpu_share == pytest.approx(0.98)


def test_unmeasured_share_is_none_not_zero():
    """None means UNKNOWN. Zero would mean 'fully starved' and convict the host."""
    stats = LoadStats(wall_s=100.0, cpu_s=None, samples=0)
    assert stats.cpu_share is None
    assert stats.starved is None


def test_zero_wall_time_does_not_divide_by_zero():
    assert LoadStats(wall_s=0.0, cpu_s=1.0, samples=1).cpu_share is None


def test_starved_is_tri_state():
    assert LoadStats(wall_s=100.0, cpu_s=40.0, samples=9).starved is True
    assert LoadStats(wall_s=100.0, cpu_s=99.0, samples=9).starved is False
    assert LoadStats(wall_s=100.0, cpu_s=None, samples=0).starved is None


def test_contended_host_is_not_blamed_on_the_kernel():
    """nearmiss_wrong_batch_stride: passed in 825s, then blew 1800s because a
    game launched mid-stage. The kernel must not be implicated."""
    stats = LoadStats(wall_s=1800.0, cpu_s=700.0, samples=400)
    msg = hostload.timeout_diagnosis(1800.0, stats)
    assert "HOST was contended" in msg
    assert "idle host" in msg
    assert "not evidence that the kernel fails to terminate" in msg


def test_an_undescheduled_timeout_points_at_the_code_without_exonerating_the_host():
    """A full CPU share means NOT DESCHEDULED, which is weaker than "the host
    was idle" -- and the message must not overclaim.

    This assertion was originally `"not starving it" in msg`, and the message
    it pinned was wrong. On 2026-08-13 a byte-identical near-miss ELF ran 1218s
    once and exceeded 1800s twice, all at ~99% CPU share: memory-bandwidth
    contention and SMT siblings slow a process that is never descheduled. A
    diagnosis that reads "the host was not starving it" sends the next reader
    to hunt a kernel bug that is not there.
    """
    stats = LoadStats(wall_s=900.0, cpu_s=890.0, samples=400)
    msg = hostload.timeout_diagnosis(900.0, stats)
    assert "not DESCHEDULED" in msg
    assert "WITHOUT ruling out a loaded host" in msg


def test_unmeasured_timeout_reports_the_ambiguity():
    """Without psutil the honest answer is 'cannot tell', not a default blame."""
    stats = LoadStats(wall_s=900.0, cpu_s=None, samples=0)
    msg = hostload.timeout_diagnosis(900.0, stats)
    assert "could not be measured" in msg
    assert "does NOT distinguish" in msg


def test_threshold_boundary_is_not_starved():
    stats = LoadStats(wall_s=100.0, cpu_s=hostload.STARVED_BELOW * 100.0, samples=9)
    assert stats.starved is False


def test_monitor_without_psutil_degrades_to_unmeasured(monkeypatch):
    """An absent optional dependency must not fail a gate."""
    monkeypatch.setattr(hostload, "psutil", None)
    with hostload.SimLoadMonitor() as mon:
        pass
    stats = mon.stats()
    assert stats.cpu_s is None
    assert stats.starved is None
    assert stats.wall_s >= 0.0


def test_monitor_survives_a_sampler_that_raises(monkeypatch):
    """Sampling is diagnostic; an error in it must never be why a gate fails."""
    class Boom:
        @staticmethod
        def Process(*a, **k):
            raise RuntimeError("no such process")
        Error = RuntimeError

    monkeypatch.setattr(hostload, "psutil", Boom)
    with hostload.SimLoadMonitor(interval_s=0.01) as mon:
        pass
    assert mon.stats().cpu_s is None

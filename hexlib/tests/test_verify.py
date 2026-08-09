import json

from hexlib.anticheat import AccelProof
from hexlib.verify import NEARMISS_ACCEPTED, NEARMISS_REJECTED, VerifyReport


def _report(**over):
    base = dict(
        task_id="rmsnorm_fp16",
        correct=True,
        kernel_cycles=9798,
        accel=AccelProof(used_hvx=True, used_hvx_compute=True, used_hmx=False),
        nearmiss={"nearmiss_no_eps.c": NEARMISS_REJECTED},
        toolchain_version="19.0.04",
        sdk_version="6.4.0.2",
        host="testbox",
        timestamp="2026-08-09T00:00:00Z",
        expert_kernel_cycles=9798,
    )
    base.update(over)
    return VerifyReport(**base)


def test_table_states_the_conditions_it_was_produced_under():
    """A pasted table a reviewer cannot date or attribute is not evidence."""
    table = _report().to_table()
    for token in ("19.0.04", "6.4.0.2", "testbox", "2026-08-09T00:00:00Z", "v75"):
        assert token in table


def test_table_reports_cycles_and_accel():
    table = _report().to_table()
    assert "9798" in table
    assert "hvx" in table.lower()


def test_table_never_claims_silicon_validation():
    """The simulator is cycle-approximate, not cycle-accurate."""
    table = _report().to_table().lower()
    assert "simulat" in table
    assert "silicon-validated" not in table


def test_nearmiss_that_passed_is_a_failure_of_the_harness():
    """A near-miss that the harness ACCEPTS means the harness does not
    discriminate — the kernel's pass proves nothing."""
    r = _report(nearmiss={"nearmiss_no_eps.c": NEARMISS_ACCEPTED})
    assert not r.gate_passed()
    assert "nearmiss_no_eps.c" in r.to_table()
    assert "WRONGLY ACCEPTED" in r.to_table()


def test_incorrect_kernel_fails_the_gate():
    assert not _report(correct=False).gate_passed()


def test_no_accel_at_all_fails_the_gate():
    r = _report(accel=AccelProof(False, False, False))
    assert not r.gate_passed()


def test_load_only_hvx_fails_the_gate():
    """Moving bytes through the vector unit is not acceleration."""
    r = _report(accel=AccelProof(used_hvx=True, used_hvx_compute=False, used_hmx=False))
    assert not r.gate_passed()


def test_all_green_passes_the_gate():
    assert _report().gate_passed()


def test_json_round_trips():
    d = json.loads(_report().to_json())
    assert d["task_id"] == "rmsnorm_fp16"
    assert d["kernel_cycles"] == 9798
    assert d["accel"]["used_hvx_compute"] is True
    assert d["toolchain_version"] == "19.0.04"


def test_inconclusive_nearmiss_fails_the_gate():
    """A near-miss that never built or ran was never offered to the harness,
    so it proves nothing and must not be treated as a rejection."""
    r = _report(
        nearmiss={"nearmiss_no_eps.c": "inconclusive: did not build — syntax error"}
    )
    assert not r.gate_passed()
    assert "INCONCLUSIVE" in r.to_table()


def test_no_nearmiss_at_all_fails_the_gate():
    """all({}) is True; gate_passed must not depend on an upstream check that
    a near-miss exists at all."""
    r = _report(nearmiss={})
    assert not r.gate_passed()


def test_zero_kernel_cycles_does_not_crash_the_table():
    """to_table() runs before verify() decides Ok/Err, so a zero-cycle report
    must render rather than raising ZeroDivisionError."""
    r = _report(kernel_cycles=0, expert_kernel_cycles=9798)
    table = r.to_table()
    assert "0" in table


def test_json_round_trips_with_string_valued_nearmiss():
    d = json.loads(_report().to_json())
    assert d["nearmiss"]["nearmiss_no_eps.c"] == NEARMISS_REJECTED

import json
import os

import pytest

from hexlib import verify as verify_mod
from hexlib.anticheat import AccelProof
from hexlib.kerneldir import KernelSpec
from hexlib.result import is_ok
from hexlib.verify import NEARMISS_ACCEPTED, NEARMISS_REJECTED, VerifyReport, verify


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


def test_table_reports_max_error_and_n_wrong():
    """Finding 10: n_wrong and max_err are measured on every run (SimOutcome
    carries both) and were previously discarded before reaching VerifyReport,
    the table, or the JSON -- the only signal a reviewer would have that a
    kernel's tolerance had been quietly widened to paper over a near-correct
    but not-quite-right implementation."""
    r = _report(n_wrong=3, max_err=0.015625)
    table = r.to_table()
    assert "0.015625" in table
    assert "n_wrong 3" in table
    d = json.loads(r.to_json())
    assert d["n_wrong"] == 3
    assert d["max_err"] == 0.015625


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


def test_inconclusive_nearmiss_is_not_shown_with_a_doubled_label():
    """Ledger #2: the stored state already carries the 'inconclusive: ' prefix
    from NEARMISS_INCONCLUSIVE, so naively prefixing 'INCONCLUSIVE -- ' again
    rendered 'INCONCLUSIVE -- inconclusive: did not build -- ...' in the
    PR-facing evidence table."""
    r = _report(
        nearmiss={"nearmiss_no_eps.c": "inconclusive: did not build -- syntax error"}
    )
    table = r.to_table()
    assert "INCONCLUSIVE -- did not build -- syntax error" in table
    assert "INCONCLUSIVE -- inconclusive" not in table


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


def _stub_gate(monkeypatch, tmp_path, *, correct=True, used_hvx_compute=True):
    """Monkeypatch verify()'s dependencies so it runs end to end without an
    SDK: a fake build, a fake sim outcome, one near-miss that is always
    (correctly) rejected, and real ELF-proof plumbing bypassed entirely."""
    from hexlib.anticheat import AccelProof
    from hexlib.build import BuildOutput
    from hexlib.sim import SimOutcome

    monkeypatch.setattr(verify_mod.kd, "validate_dir", lambda d: [])
    monkeypatch.setattr(
        verify_mod.kd,
        "load_spec",
        lambda d: KernelSpec(task_id="k", dtype="fp16"),
    )
    monkeypatch.setattr(
        verify_mod.kd,
        "nearmiss_files",
        lambda d: [os.path.join(d, "nearmiss_x.c")],
    )
    monkeypatch.setattr(
        verify_mod,
        "build_kernel",
        lambda kernel_dir, out_dir, caps, impl="kernel.c", sdk_root=None: BuildOutput(
            elf="x.elf", obj=impl, bin_dir=str(tmp_path), toolchain_version="19.0.04"
        ),
    )
    monkeypatch.setattr(
        verify_mod,
        "run_sim",
        lambda built, caps, timeout=None: SimOutcome(
            correct=(correct if built.obj == "kernel.c" else False),
            n_wrong=0, max_err=0.0, kernel_cycles=100,
        ),
    )
    monkeypatch.setattr(
        verify_mod,
        "prove_accel",
        lambda *a, **k: AccelProof(
            used_hvx=True, used_hvx_compute=used_hvx_compute, used_hmx=False
        ),
    )


def test_passing_gate_writes_result_md_into_the_kernel_directory(monkeypatch, tmp_path):
    """Finding 7: the file CI validates must be the file the tool produced --
    not a manual copy of a gitignored _work/ path a contributor has to
    discover on their own."""
    kernel_dir = tmp_path / "k"
    kernel_dir.mkdir()
    _stub_gate(monkeypatch, tmp_path)

    result = verify(str(kernel_dir), str(tmp_path / "_work"))
    assert is_ok(result)
    result_md = kernel_dir / "RESULT.md"
    assert result_md.is_file()
    assert "**gate** | **PASS**" in result_md.read_text(encoding="utf-8")


def test_failing_gate_does_not_write_result_md_into_the_kernel_directory(
    monkeypatch, tmp_path
):
    """A FAIL belongs in _work, not committed next to the kernel."""
    kernel_dir = tmp_path / "k"
    kernel_dir.mkdir()
    _stub_gate(monkeypatch, tmp_path, correct=False)

    result = verify(str(kernel_dir), str(tmp_path / "_work"))
    assert not is_ok(result)
    assert not (kernel_dir / "RESULT.md").is_file()


def test_declared_mechanism_the_elf_does_not_show_is_rejected(monkeypatch, tmp_path):
    """Finding 9: spec.json can claim a mechanism (e.g. hmx) that the ELF the
    gate already disassembled does not show. Nothing else in the codebase
    ever compares spec.mechanisms against the AccelProof -- to_table() prints
    only the PROVEN mechanisms, so the false claim would appear nowhere a
    reviewer looks, while spec.json is exactly what a future index or docs
    generator would read."""
    kernel_dir = tmp_path / "k"
    kernel_dir.mkdir()
    _stub_gate(monkeypatch, tmp_path, used_hvx_compute=True)
    monkeypatch.setattr(
        verify_mod.kd,
        "load_spec",
        lambda d: KernelSpec(task_id="k", dtype="fp16", mechanisms=["hvx", "hmx"]),
    )

    result = verify(str(kernel_dir), str(tmp_path / "_work"))
    assert not is_ok(result)
    assert "hmx" in result.reason
    assert not (kernel_dir / "RESULT.md").is_file()


def test_declared_mechanism_the_elf_does_show_is_accepted(monkeypatch, tmp_path):
    """Sanity check: a spec.json claim the ELF backs up must not be rejected."""
    kernel_dir = tmp_path / "k"
    kernel_dir.mkdir()
    _stub_gate(monkeypatch, tmp_path, used_hvx_compute=True)
    monkeypatch.setattr(
        verify_mod.kd,
        "load_spec",
        lambda d: KernelSpec(task_id="k", dtype="fp16", mechanisms=["hvx"]),
    )

    result = verify(str(kernel_dir), str(tmp_path / "_work"))
    assert is_ok(result)


def test_a_stale_passing_result_does_not_survive_a_later_build_failure(
    monkeypatch, tmp_path
):
    """Finding 8: a contributor greens the gate, edits kernel.c, re-runs and
    gets a build failure. Before this fix, _work/<task>.result.md and the
    committed kernels/<task>/RESULT.md from the earlier PASS were left in
    place untouched -- genuine output of a genuine run, just not of this
    one -- and nothing downstream could tell the difference."""
    kernel_dir = tmp_path / "k"
    kernel_dir.mkdir()
    out_dir = tmp_path / "_work"
    out_dir.mkdir()

    stale_work_md = out_dir / "k.result.md"
    stale_work_json = out_dir / "k.result.json"
    stale_kernel_md = kernel_dir / "RESULT.md"
    for p in (stale_work_md, stale_work_json, stale_kernel_md):
        p.write_text("STALE PASS FROM AN EARLIER RUN", encoding="utf-8")

    monkeypatch.setattr(verify_mod.kd, "validate_dir", lambda d: [])
    monkeypatch.setattr(
        verify_mod.kd, "load_spec", lambda d: KernelSpec(task_id="k", dtype="fp16")
    )

    def _boom(*a, **k):
        from hexlib.build import BuildError

        raise BuildError("compile failed", "kernel.c:1:1: error: bad")

    monkeypatch.setattr(verify_mod, "build_kernel", _boom)

    result = verify(str(kernel_dir), str(out_dir))
    assert not is_ok(result)
    assert not stale_work_md.exists()
    assert not stale_work_json.exists()
    assert not stale_kernel_md.exists()


def test_missing_sdk_is_reported_not_raised(monkeypatch, tmp_path):
    """Finding 4: build_kernel raises FileNotFoundError/ValueError directly
    (toolchain discovery happens before any subprocess call), so verify() must
    catch those alongside BuildError -- otherwise a missing or partially
    installed SDK crashes to a raw traceback instead of an Err result."""
    monkeypatch.setattr(verify_mod.kd, "validate_dir", lambda d: [])
    monkeypatch.setattr(
        verify_mod.kd,
        "load_spec",
        lambda d: KernelSpec(task_id="k", dtype="fp16"),
    )

    def _boom(*a, **k):
        raise FileNotFoundError("No Hexagon toolchain found under '/nope'")

    monkeypatch.setattr(verify_mod, "build_kernel", _boom)

    result = verify(str(tmp_path), str(tmp_path / "_work"))
    assert not is_ok(result)
    assert "SDK" in result.reason or "SDK" in (result.detail or "")

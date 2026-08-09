import pytest

from hexlib import sim


def test_parses_verdict_and_cycles():
    text = (
        "some simulator preamble\n"
        "HEXLIB_VERDICT correct=1 wrong=0 maxerr=0.000244141\n"
        "HEXLIB_KCYCLES kernel=9798\n"
        "T0: Insns=12345 Packets=6789\n"
    )
    assert sim.parse_verdict(text) == (True, 0, pytest.approx(0.000244141))
    assert sim.parse_kernel_cycles(text) == 9798


def test_parses_a_failing_verdict():
    text = "HEXLIB_VERDICT correct=0 wrong=567 maxerr=1.5\nHEXLIB_KCYCLES kernel=42\n"
    correct, n_wrong, max_err = sim.parse_verdict(text)
    assert correct is False and n_wrong == 567


def test_missing_lines_parse_as_none():
    assert sim.parse_verdict("simulator crashed") is None
    assert sim.parse_kernel_cycles("simulator crashed") is None


def test_run_sim_fails_closed_when_no_verdict(monkeypatch, tmp_path):
    """A simulator run that completes but emits nothing is a FAILURE.
    This is the QDC false-pass in a different costume: process exited 0,
    zero results recovered, and the old code called that a pass."""
    from hexlib.build import BuildOutput

    monkeypatch.setattr(
        sim.tc, "run", lambda *a, **k: (0, "simulator ran, said nothing", "", False)
    )
    monkeypatch.setattr(sim.tc, "toolchain_env", lambda *a, **k: {})
    out = BuildOutput(elf="x.elf", obj="x.o", bin_dir=str(tmp_path),
                      toolchain_version="19.0.04")
    with pytest.raises(sim.SimError, match="no verdict"):
        sim.run_sim(out, caps=[])


def test_run_sim_fails_closed_when_no_cycles(monkeypatch, tmp_path):
    from hexlib.build import BuildOutput

    monkeypatch.setattr(
        sim.tc, "run",
        lambda *a, **k: (0, "HEXLIB_VERDICT correct=1 wrong=0 maxerr=0\n", "", False),
    )
    monkeypatch.setattr(sim.tc, "toolchain_env", lambda *a, **k: {})
    out = BuildOutput(elf="x.elf", obj="x.o", bin_dir=str(tmp_path),
                      toolchain_version="19.0.04")
    with pytest.raises(sim.SimError, match="no kernel cycle count"):
        sim.run_sim(out, caps=[])


def test_run_sim_reports_timeout_distinctly(monkeypatch, tmp_path):
    from hexlib.build import BuildOutput

    monkeypatch.setattr(sim.tc, "run", lambda *a, **k: (None, "", "", True))
    monkeypatch.setattr(sim.tc, "toolchain_env", lambda *a, **k: {})
    out = BuildOutput(elf="x.elf", obj="x.o", bin_dir=str(tmp_path),
                      toolchain_version="19.0.04")
    with pytest.raises(sim.SimError, match="timed out"):
        sim.run_sim(out, caps=[])


def test_two_verdicts_is_a_failure(monkeypatch, tmp_path):
    """re.search takes the first match; a second verdict must not be silently
    discarded, and a kernel must not be able to pre-empt the harness."""
    from hexlib.build import BuildOutput

    text = (
        "HEXLIB_VERDICT correct=1 wrong=0 maxerr=0\nHEXLIB_KCYCLES kernel=1\n"
        "HEXLIB_VERDICT correct=0 wrong=512 maxerr=9.9\n"
        "HEXLIB_KCYCLES kernel=70000\n"
    )
    monkeypatch.setattr(sim.tc, "run", lambda *a, **k: (0, text, "", False))
    monkeypatch.setattr(sim.tc, "toolchain_env", lambda *a, **k: {})
    out = BuildOutput(elf="x.elf", obj="x.o", bin_dir=str(tmp_path),
                      toolchain_version="19.0.04")
    with pytest.raises(sim.SimError, match="exactly one"):
        sim.run_sim(out, caps=[])


def test_two_kernel_cycle_lines_is_a_failure(monkeypatch, tmp_path):
    """A run with two cycle counts has no single measurement, even if the
    verdict itself is unambiguous."""
    from hexlib.build import BuildOutput

    text = (
        "HEXLIB_VERDICT correct=1 wrong=0 maxerr=0\n"
        "HEXLIB_KCYCLES kernel=1\nHEXLIB_KCYCLES kernel=2\n"
    )
    monkeypatch.setattr(sim.tc, "run", lambda *a, **k: (0, text, "", False))
    monkeypatch.setattr(sim.tc, "toolchain_env", lambda *a, **k: {})
    out = BuildOutput(elf="x.elf", obj="x.o", bin_dir=str(tmp_path),
                      toolchain_version="19.0.04")
    with pytest.raises(sim.SimError, match="exactly one"):
        sim.run_sim(out, caps=[])


def test_nonzero_return_code_is_a_failure_even_with_a_flushed_verdict(monkeypatch, tmp_path):
    """hexlib_report() fflushes before returning, so a crash during CRT
    teardown or a harness that exits nonzero can still leave a full PASS
    verdict on the wire. The return code must not be discarded."""
    from hexlib.build import BuildOutput

    text = (
        "HEXLIB_VERDICT correct=1 wrong=0 maxerr=0\n"
        "HEXLIB_KCYCLES kernel=2021\nSegmentation fault\n"
    )
    monkeypatch.setattr(sim.tc, "run", lambda *a, **k: (139, text, "", False))
    monkeypatch.setattr(sim.tc, "toolchain_env", lambda *a, **k: {})
    out = BuildOutput(elf="x.elf", obj="x.o", bin_dir=str(tmp_path),
                      toolchain_version="19.0.04")
    with pytest.raises(sim.SimError, match="exited 139"):
        sim.run_sim(out, caps=[])


def test_sim_command_pins_the_bus_model():
    cmd = sim.sim_command("hexagon-sim", "a.elf", caps=[])
    assert "-mv75" in cmd
    assert "--timing" in cmd
    assert cmd[cmd.index("--buspenalty") + 1] == "75"
    assert cmd[cmd.index("--busratio") + 1] == "2"
    assert cmd[-1] == "a.elf"


def test_sim_command_adds_hmx_only_when_capped():
    assert "--mhmx" in sim.sim_command("s", "a.elf", ["hmx"])
    assert "--mhmx" not in sim.sim_command("s", "a.elf", [])

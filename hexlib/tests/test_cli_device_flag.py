# hexlib/tests/test_cli_device_flag.py
"""`hexlib test <kernel> --device sim|local|qdc`.

Every guard here is checked to run BEFORE `cli._qdc_submit` -- the function
that actually touches the SDK, a credential, or the network -- ever gets
called. `_qdc_submit` itself is monkeypatched in every test that reaches it,
so nothing here builds a real device artifact, reads a credential, or makes
a network call; the tests that exercise the guards past the missing-timeout
check are, structurally, offline tests of argument handling and print
ordering, nothing more.

THE KERNEL ARGUMENT IS `scale_fp16` THROUGHOUT, and until 2026-08-11 it was
`some/kernel` -- which passed, because `--device qdc` read the argument
NOWHERE. `hexlib test add_fp16 --device qdc --timeout-min 20 --yes` would
spend 20 non-renewable minutes running `scale_fp16` (test_on_device.py
hard-codes `./hexlib_run --self-test`, main.c hard-codes build_scale_batch)
and return green. Every test below that expects submission to PROCEED must
therefore name a kernel stage 3 can genuinely run, and the refusal itself is
pinned by the tests near the end of this file.
"""
import os

import pytest

from hexlib import cli
from hexlib.result import Measurements, Ok


def _measurements() -> Measurements:
    return Measurements(
        kernel_cycles=886,
        toolchain_version="test",
        sdk_version="test",
        host="test",
        timestamp="2026-08-11T00:00:00",
    )


def test_device_defaults_to_sim(monkeypatch, tmp_path):
    """No --device at all must reach the SAME code path as --device sim --
    verified by watching `verify` actually get called with this kernel's own
    path, not by inference from an error message. A cli.py that quietly
    changed the default to `local` or `qdc` would still print SOME output;
    only checking that `verify` itself ran, with the right argument, catches
    that."""
    calls = {}

    def fake_verify(kernel, out):
        calls["kernel"] = kernel
        calls["out"] = out
        return Ok(_measurements())

    monkeypatch.setattr(cli, "verify", fake_verify)
    kernel_dir = str(tmp_path / "k")
    os.makedirs(kernel_dir)
    rc = cli.main(["test", kernel_dir, "--out", str(tmp_path / "out")])
    assert rc == 0
    assert calls == {"kernel": kernel_dir, "out": str(tmp_path / "out")}


def test_device_sim_is_the_same_path_as_no_flag_at_all(monkeypatch, tmp_path):
    """The mirror of the test above, with --device sim spelled out
    explicitly -- both must land on the identical `verify` call."""
    calls = []

    def fake_verify(kernel, out):
        calls.append((kernel, out))
        return Ok(_measurements())

    monkeypatch.setattr(cli, "verify", fake_verify)
    kernel_dir = str(tmp_path / "k")
    os.makedirs(kernel_dir)
    cli.main(["test", kernel_dir, "--device", "sim", "--out", str(tmp_path / "out")])
    assert calls == [(kernel_dir, str(tmp_path / "out"))]


def test_device_local_says_it_is_not_implemented_rather_than_failing_obscurely(capsys):
    rc = cli.main(["test", "some/kernel", "--device", "local"])
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "not implemented" in err
    assert "no device available" in err


def test_device_local_never_reaches_verify_or_qdc(monkeypatch, capsys):
    """A --device local that fell through to the sim path (or the qdc path)
    would still print SOME error, which could look like "the right shape" to
    a weaker test. This one proves it took neither: `verify` and
    `cli._qdc_submit` are both wired to raise if reached at all."""
    def boom_verify(kernel, out):
        raise AssertionError("verify must not run for --device local")

    def boom_submit(args):
        raise AssertionError("_qdc_submit must not run for --device local")

    monkeypatch.setattr(cli, "verify", boom_verify)
    monkeypatch.setattr(cli, "_qdc_submit", boom_submit)
    rc = cli.main(["test", "some/kernel", "--device", "local"])
    assert rc != 0


def test_qdc_refuses_without_a_timeout(monkeypatch, capsys):
    def boom_submit(args):
        raise AssertionError("_qdc_submit must not run without --timeout-min")

    monkeypatch.setattr(cli, "_qdc_submit", boom_submit)
    rc = cli.main(["test", "scale_fp16", "--device", "qdc"])
    assert rc != 0
    err = capsys.readouterr().err
    assert "--timeout-min" in err
    assert "timeout" in err.lower()


def test_qdc_prints_the_budget_before_submitting(monkeypatch, capsys):
    """The budget line must be PRESENT in stdout, and it must appear before
    `_qdc_submit` is ever reached -- checked by making the fake submit
    itself assert the budget line already printed, not merely by checking
    final output order after the fact (which could pass even if a future
    edit moved the print to happen lazily, inside submit, or not at all,
    as long as it eventually landed in stdout somewhere)."""
    order = []

    def fake_submit(args):
        order.append("submit")
        # By the time submit runs, the budget line must already be in stdout.
        out = capsys.readouterr().out
        assert "remaining budget" in out, (
            "the budget must print BEFORE submission is attempted, not after"
        )
        return 0

    monkeypatch.setattr(cli, "_qdc_submit", fake_submit)
    monkeypatch.delenv(cli._QDC_BUDGET_ENV, raising=False)
    rc = cli.main(["test", "scale_fp16", "--device", "qdc", "--timeout-min", "5"])
    assert rc == 0
    assert order == ["submit"]


def test_qdc_prints_the_actual_env_budget_when_set(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_qdc_submit", lambda args: 0)
    monkeypatch.setenv(cli._QDC_BUDGET_ENV, "42")
    cli.main(["test", "scale_fp16", "--device", "qdc", "--timeout-min", "5"])
    out = capsys.readouterr().out
    assert "42" in out
    assert "remaining budget" in out


def test_qdc_requires_yes_above_the_threshold(monkeypatch, capsys):
    def boom_submit(args):
        raise AssertionError("_qdc_submit must not run above the threshold without --yes")

    monkeypatch.setattr(cli, "_qdc_submit", boom_submit)
    above = cli._QDC_YES_THRESHOLD_MIN + 1
    rc = cli.main(["test", "scale_fp16", "--device", "qdc", "--timeout-min", str(above)])
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "--yes" in err
    assert "threshold" in err


def test_qdc_proceeds_above_the_threshold_with_yes(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "_qdc_submit", lambda args: calls.append(args) or 0)
    above = cli._QDC_YES_THRESHOLD_MIN + 1
    rc = cli.main([
        "test", "scale_fp16", "--device", "qdc",
        "--timeout-min", str(above), "--yes",
    ])
    assert rc == 0
    assert len(calls) == 1


def test_qdc_proceeds_at_or_below_the_threshold_without_yes(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_qdc_submit", lambda args: calls.append(args) or 0)
    rc = cli.main([
        "test", "scale_fp16", "--device", "qdc",
        "--timeout-min", str(cli._QDC_YES_THRESHOLD_MIN),
    ])
    assert rc == 0
    assert len(calls) == 1


# ==============================================================================
# THE <kernel> ARGUMENT IS READ, NOT IGNORED.
#
# `_cmd_test_qdc` and `_qdc_submit` both used to ignore `args.kernel` entirely
# -- `hexlib/tests/test_cli_qdc_results.py` even built an `argparse.Namespace`
# with NO `kernel` attribute at all and `_qdc_submit` ran fine. So
# `hexlib test add_fp16 --device qdc --timeout-min 20 --yes` spent 20
# non-renewable minutes, measured `scale_fp16` (the staged script hard-codes
# `./hexlib_run --self-test`; main.c's run_self_test hard-codes
# build_scale_batch), exited 0, and told the operator add_fp16 was validated on
# silicon. Making the argument genuinely work means parameterising main.c -- a
# real change, not made. Refusing is the honest state.
# ==============================================================================


def _boom_submit(args):
    raise AssertionError("_qdc_submit must not run for an unsupported kernel")


@pytest.mark.parametrize("kernel", ["add_fp16", "kernels/add_fp16",
                                    "rmsnorm_fp16", "layernorm_fp16"])
def test_qdc_refuses_any_kernel_but_scale_fp16(monkeypatch, capsys, kernel):
    monkeypatch.setattr(cli, "_qdc_submit", _boom_submit)
    rc = cli.main([
        "test", kernel, "--device", "qdc", "--timeout-min", "5",
    ])
    assert rc != 0
    err = capsys.readouterr().err
    assert "scale_fp16-only" in err
    assert os.path.basename(kernel) in err


def test_qdc_refuses_an_unsupported_kernel_before_anything_else(monkeypatch, capsys):
    """The kernel refusal comes FIRST -- before the missing-timeout check and
    before the budget line. "This command cannot run what you asked for" is
    the more useful message when more than one guard would fire, and it costs
    nothing to check."""
    monkeypatch.setattr(cli, "_qdc_submit", _boom_submit)
    rc = cli.main(["test", "add_fp16", "--device", "qdc"])   # no --timeout-min
    assert rc != 0
    out, err = capsys.readouterr()
    assert "scale_fp16-only" in err
    assert "remaining budget" not in out


def test_qdc_accepts_the_kernel_directory_path_form_too(monkeypatch):
    """`--device sim` takes `kernels/scale_fp16`; docs/STATE.md's stage-3 entry
    point spells it `scale_fp16`. One CLI, both forms."""
    calls = []
    monkeypatch.setattr(cli, "_qdc_submit", lambda args: calls.append(args) or 0)
    rc = cli.main([
        "test", os.path.join("kernels", "scale_fp16"), "--device", "qdc",
        "--timeout-min", "5",
    ])
    assert rc == 0
    assert len(calls) == 1


# ==============================================================================
# QDC_BUDGET_MIN IS COMPARED TO --timeout-min, not merely printed.
#
# VERIFIED BEFORE THE FIX: `QDC_BUDGET_MIN=3 hexlib test k --device qdc
# --timeout-min 240 --yes` printed `remaining budget: 3 minutes (from
# QDC_BUDGET_MIN)` and then submitted a 240-minute job -- an 80x overspend of
# non-renewable minutes passing every guard, with the number that should have
# stopped it on screen. `QDC_BUDGET_MIN=abc` printed "remaining budget: abc
# minutes".
# ==============================================================================


def test_qdc_refuses_a_timeout_larger_than_the_recorded_budget(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_qdc_submit", _boom_submit)
    monkeypatch.setenv(cli._QDC_BUDGET_ENV, "3")
    rc = cli.main([
        "test", "scale_fp16", "--device", "qdc", "--timeout-min", "240", "--yes",
    ])
    assert rc != 0
    err = capsys.readouterr().err
    assert "240" in err and "3" in err
    assert "budget" in err.lower()


def test_qdc_allows_a_timeout_exactly_equal_to_the_budget(monkeypatch):
    """Spending the last recorded minutes deliberately is a real thing to
    want; only EXCEEDING the budget is refused."""
    calls = []
    monkeypatch.setattr(cli, "_qdc_submit", lambda args: calls.append(args) or 0)
    monkeypatch.setenv(cli._QDC_BUDGET_ENV, "5")
    rc = cli.main(["test", "scale_fp16", "--device", "qdc", "--timeout-min", "5"])
    assert rc == 0
    assert len(calls) == 1


@pytest.mark.parametrize("bad", ["abc", "", "  ", "5.5", "-5", "5 minutes", "1e3"])
def test_qdc_refuses_a_malformed_budget_rather_than_printing_it(
    monkeypatch, capsys, bad
):
    """A budget that cannot be compared is refused, not echoed. A guard that
    silently disables itself on a typo is worse than no guard, because the
    operator believes it is watching."""
    monkeypatch.setattr(cli, "_qdc_submit", _boom_submit)
    monkeypatch.setenv(cli._QDC_BUDGET_ENV, bad)
    rc = cli.main(["test", "scale_fp16", "--device", "qdc", "--timeout-min", "5"])
    assert rc != 0
    err = capsys.readouterr().err
    assert cli._QDC_BUDGET_ENV in err


def test_an_unset_budget_means_unknown_and_says_no_check_was_made(monkeypatch, capsys):
    """THE STATED DECISION, pinned so it cannot drift into an accident: unset
    means UNKNOWN, submission proceeds, and the line says outright that no
    budget check happened. It must not read as "unlimited" or as an assurance
    that the job fits."""
    calls = []
    monkeypatch.setattr(cli, "_qdc_submit", lambda args: calls.append(args) or 0)
    monkeypatch.delenv(cli._QDC_BUDGET_ENV, raising=False)
    rc = cli.main(["test", "scale_fp16", "--device", "qdc", "--timeout-min", "5"])
    assert rc == 0
    assert len(calls) == 1
    out = capsys.readouterr().out.lower()
    assert "unknown" in out
    assert "no budget check" in out
    assert "not unlimited" in out


def test_the_budget_is_read_from_the_environment_and_never_queried(monkeypatch):
    """Nothing in the budget path may reach QDC: there is no reliable
    remaining-minutes API on this account, and a test must never make a
    network call. Proven by making the whole QDC client constructor explode
    if touched."""
    from hexlib.device.qdc import job

    def boom():
        raise AssertionError("the budget path must never build a QDC client")

    monkeypatch.setattr(job, "_client", boom)
    monkeypatch.setenv(cli._QDC_BUDGET_ENV, "17")
    assert cli._qdc_remaining_budget_min() == 17
    assert cli._qdc_budget_guard(5) == 0


# ==============================================================================
# --timeout-min's 1..240 RANGE IS ENFORCED BEFORE THE BUILD.
#
# job.submit raises on the same range, but only after `_qdc_submit` has run a
# full SDK cross-compile of hexlib_run + libhexlib_skel.so and staged a zip.
# The bounds come from job.py, never respelled here.
# ==============================================================================


@pytest.mark.parametrize("bad", [0, -1, 241, 100000])
def test_qdc_refuses_an_out_of_range_timeout_before_building_anything(
    monkeypatch, capsys, bad
):
    monkeypatch.setattr(cli, "_qdc_submit", _boom_submit)
    rc = cli.main([
        "test", "scale_fp16", "--device", "qdc", "--timeout-min", str(bad), "--yes",
    ])
    assert rc != 0
    err = capsys.readouterr().err
    assert "--timeout-min" in err
    assert "1..240" in err or ("1" in err and "240" in err)


def test_the_cli_range_is_job_pys_own_range_not_a_second_copy():
    from hexlib.device.qdc import job

    assert (job.MIN_TIMEOUT_MIN, job.MAX_TIMEOUT_MIN) == (1, 240)


# ==============================================================================
# --timeout-min / --yes ARE REFUSED FOR sim AND local, not silently ignored.
# They are the spend controls for the one backend that spends anything.
# ==============================================================================


@pytest.mark.parametrize("device", ["sim", "local"])
@pytest.mark.parametrize(
    "extra", [["--timeout-min", "20"], ["--yes"], ["--timeout-min", "20", "--yes"]]
)
def test_timeout_and_yes_are_refused_for_non_qdc_devices(
    monkeypatch, capsys, device, extra
):
    def boom_verify(kernel, out):
        raise AssertionError("verify must not run when the flags are refused")

    monkeypatch.setattr(cli, "verify", boom_verify)
    monkeypatch.setattr(cli, "_qdc_submit", _boom_submit)
    rc = cli.main(["test", "kernels/scale_fp16", "--device", device] + extra)
    assert rc == 2
    err = capsys.readouterr().err
    assert "--device qdc" in err
    for flag in ("--timeout-min", "--yes"):
        if flag in extra:
            assert flag in err

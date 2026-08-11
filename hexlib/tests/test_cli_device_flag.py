# hexlib/tests/test_cli_device_flag.py
"""`hexlib test <kernel> --device sim|local|qdc`.

Every guard here is checked to run BEFORE `cli._qdc_submit` -- the function
that actually touches the SDK, a credential, or the network -- ever gets
called. `_qdc_submit` itself is monkeypatched in every test that reaches it,
so nothing here builds a real device artifact, reads a credential, or makes
a network call; the three tests that exercise the guards past the
missing-timeout check are, structurally, offline tests of argument handling
and print ordering, nothing more.
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
    rc = cli.main(["test", "some/kernel", "--device", "qdc"])
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
    rc = cli.main(["test", "some/kernel", "--device", "qdc", "--timeout-min", "5"])
    assert rc == 0
    assert order == ["submit"]


def test_qdc_prints_the_actual_env_budget_when_set(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_qdc_submit", lambda args: 0)
    monkeypatch.setenv(cli._QDC_BUDGET_ENV, "42")
    cli.main(["test", "some/kernel", "--device", "qdc", "--timeout-min", "5"])
    out = capsys.readouterr().out
    assert "42" in out
    assert "remaining budget" in out


def test_qdc_requires_yes_above_the_threshold(monkeypatch, capsys):
    def boom_submit(args):
        raise AssertionError("_qdc_submit must not run above the threshold without --yes")

    monkeypatch.setattr(cli, "_qdc_submit", boom_submit)
    above = cli._QDC_YES_THRESHOLD_MIN + 1
    rc = cli.main(["test", "some/kernel", "--device", "qdc", "--timeout-min", str(above)])
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "--yes" in err
    assert "threshold" in err


def test_qdc_proceeds_above_the_threshold_with_yes(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "_qdc_submit", lambda args: calls.append(args) or 0)
    above = cli._QDC_YES_THRESHOLD_MIN + 1
    rc = cli.main([
        "test", "some/kernel", "--device", "qdc",
        "--timeout-min", str(above), "--yes",
    ])
    assert rc == 0
    assert len(calls) == 1


def test_qdc_proceeds_at_or_below_the_threshold_without_yes(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_qdc_submit", lambda args: calls.append(args) or 0)
    rc = cli.main([
        "test", "some/kernel", "--device", "qdc",
        "--timeout-min", str(cli._QDC_YES_THRESHOLD_MIN),
    ])
    assert rc == 0
    assert len(calls) == 1

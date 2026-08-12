# hexlib/tests/test_cli_qdc_results.py
"""Critical 3: `hexlib._qdc_submit` must never return 0 for a job whose
results.xml reports zero tests, any failures/errors, is unparseable, is
missing entirely, or -- even when it parses clean -- whose fetched logs
never actually show the measurement lines a genuine device run prints.

Before this fix, `_qdc_submit` returned 0 the moment `job.fetch()` finished
downloading files: nothing in the tree parsed `results.xml` at all (no
`ElementTree`, no `failures=` anywhere), so five failed on-device assertions
-- or a results.xml containing only a collection error -- produced a green
CLI. This is the project's own named failure mode (a device-farm job that
ran zero tests and reported passing), one level up from where it was fixed
in job.py's own `wait()`.

Every test here monkeypatches `hexlib.runtime.build.build_device_binary`,
`hexlib.device.qdc.artifact.stage`, and `hexlib.device.qdc.job.submit` /
`.wait` / `.fetch` directly -- the same modules `_qdc_submit` imports
lazily -- exactly the way `hexlib/tests/test_qdc.py` monkeypatches the SDK
boundary. No credential, no network, no real device artifact anywhere in
this file.
"""
import argparse
import os

import pytest

from hexlib import cli
from hexlib.device.qdc import artifact, job
from hexlib.runtime import build as runtime_build

# The exact strings hexlib_run's own run_self_test() prints on a genuine
# pass (hexlib/runtime/host/main.c) -- pinned as constants here too so a
# typo in one fabricated log can't accidentally satisfy the other.
PASS_LINE = "hexlib: --self-test: PASS (4100 values, bit-exact)"
CYCLES_LINE = "hexlib: --self-test: cycles_total=886"
GOOD_LOG = f"{PASS_LINE}\n{CYCLES_LINE}\n"


def _args(tmp_path, kernel="scale_fp16", timeout_min=5, yes=None):
    """A REALISTIC Namespace, `kernel` included. It used to be built with no
    `kernel` attribute at all and `_qdc_submit` ran fine -- which was itself
    the evidence that `--device qdc` ignored the argument and would spend real
    minutes measuring scale_fp16 no matter which kernel was asked for.
    `_qdc_submit` now refuses an args object with no kernel on it, so leaving
    it out here would fail loudly instead of passing silently.

    `yes` defaults to WHAT THIS TIMEOUT ACTUALLY REQUIRES, because
    `_qdc_submit` now re-checks the confirmation threshold and the budget as
    well as the kernel -- it is the function that spends the minutes, and it is
    reachable without going through `_cmd_test_qdc`. Every test here means to
    submit a legitimate job, so a `timeout_min` above the threshold implies
    `--yes`; hard-coding `yes=False` for all of them made three tests assert on
    a job the CLI would have refused at the front door. Pass `yes` explicitly
    to test the guard itself.
    """
    if yes is None:
        yes = timeout_min > cli._QDC_YES_THRESHOLD_MIN
    return argparse.Namespace(
        out=str(tmp_path / "out"), timeout_min=timeout_min, yes=yes, kernel=kernel
    )


def _stub_build_submit_and_wait(monkeypatch):
    """Stub out everything before job.fetch(): building a real device binary
    needs the Hexagon SDK, staging needs real files, and submit/wait need a
    real QDC account -- none of which is what this file exists to check."""

    def fake_build_device_binary(build_dir, sdk_root=None):
        os.makedirs(build_dir, exist_ok=True)
        exe = os.path.join(build_dir, "hexlib_run")
        open(exe, "wb").close()
        open(os.path.join(build_dir, "libhexlib_skel.so"), "wb").close()
        return exe

    def fake_stage(binaries, test_script, out_base, support_files=None):
        zip_path = out_base + ".zip"
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        open(zip_path, "wb").close()
        return zip_path

    monkeypatch.setattr(runtime_build, "build_device_binary", fake_build_device_binary)
    monkeypatch.setattr(artifact, "stage", fake_stage)
    monkeypatch.setattr(job, "submit", lambda zip_path, *, timeout_min: 999)
    monkeypatch.setattr(job, "wait", lambda job_id, **kw: True)


def _write_log(log_dir, rel, text):
    """One fetched file at `rel`, a QDC-style relative path with forward
    slashes, mirrored beneath `log_dir` the way `job.fetch` mirrors it."""
    p = os.path.join(log_dir, *rel.split("/"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def _fake_fetch(tmp_path, *, results_xml, extra_logs=None, decoy_results_xml=None):
    """Write fabricated fetched log files under out/qdc_logs and return the
    list of local paths -- the same shape `job.fetch`'s real return value
    has (job.py's own `fetch()` returns exactly this: local paths it wrote
    from QDC's log files).

    THE REPORT GOES AT `TestLogs/results.xml`, NOT AT THE TOP LEVEL. It used
    to be written flat, which quietly made every test here unable to see the
    defect that `job.fetch` mirrors QDC's directory layout -- so two files
    named results.xml can now exist, and the reader used to pick by basename.
    `job.RESULTS_MARKER` is the real remote name and is used here rather than
    respelled, so the fixture cannot drift away from what `wait()` looks for.

    `decoy_results_xml` writes a SECOND results.xml at `logs/results.xml`,
    which is the shape of the framework placeholder, and puts it FIRST in the
    returned list -- QDC's listing order is not ours to choose.
    """
    log_dir = os.path.join(str(tmp_path / "out"), "qdc_logs")
    os.makedirs(log_dir, exist_ok=True)
    paths = []
    if decoy_results_xml is not None:
        paths.append(_write_log(log_dir, "logs/results.xml", decoy_results_xml))
    if results_xml is not None:
        paths.append(_write_log(log_dir, job.RESULTS_MARKER, results_xml))
    for name, text in (extra_logs or {}).items():
        paths.append(_write_log(log_dir, name, text))
    return paths


def _boom_build(monkeypatch):
    """Make the FIRST thing after the guards explode. `_qdc_submit`'s guards all
    run before `build_device_binary`, so if a guard refuses we get its exit code
    and if it does not we get this -- no test here can pass by accident because
    something further down happened to fail."""

    def boom(build_dir, sdk_root=None):
        raise AssertionError(
            "reached the device build: _qdc_submit accepted a job its own "
            "guards should have refused"
        )

    monkeypatch.setattr(runtime_build, "build_device_binary", boom)


def test_qdc_submit_refuses_an_over_budget_job_on_its_own(monkeypatch, tmp_path, capsys):
    """THE MONEY GUARD, CHECKED WHERE THE MONEY IS SPENT.

    `_qdc_submit` re-checked `_qdc_kernel_refusal` -- the cheapest of the
    guards -- with an explicit rationale: it is the function that spends the
    minutes and it is called directly, by tests today and any second caller
    tomorrow. The budget comparison and the `--yes` threshold were left one
    frame up in `_cmd_test_qdc`, so the rationale was written and then applied
    to the one guard whose absence costs nothing.

    This is the 80x overspend `_qdc_budget_guard`'s docstring says it exists to
    prevent, reached past a guard that is present: a 240-minute job under a
    3-minute recorded budget, submitted with `--yes` so the confirmation
    threshold cannot be what stops it.
    """
    monkeypatch.setenv(cli._QDC_BUDGET_ENV, "3")
    _boom_build(monkeypatch)
    rc = cli._qdc_submit(_args(tmp_path, timeout_min=240, yes=True))
    assert rc == 2
    err = capsys.readouterr().err.lower()
    assert "240" in err and "budget" in err


def test_qdc_submit_refuses_above_the_confirmation_threshold_on_its_own(
    monkeypatch, tmp_path, capsys
):
    """The other half, and the half that does not depend on the operator
    having recorded a budget anywhere -- so it is the guard that must hold on
    every path. Budget deliberately unset: nothing but the threshold can
    refuse this."""
    monkeypatch.delenv(cli._QDC_BUDGET_ENV, raising=False)
    _boom_build(monkeypatch)
    rc = cli._qdc_submit(_args(tmp_path, timeout_min=240, yes=False))
    assert rc == 2
    err = capsys.readouterr().err.lower()
    assert "--yes" in err


def test_qdc_submit_refuses_a_timeout_it_cannot_compare_to_a_budget(
    monkeypatch, tmp_path, capsys
):
    """`_cmd_test_qdc` rejects `timeout_min=None` before `_qdc_submit` sees it.
    A direct caller carries no such guarantee, and `None > budget` raises on
    one path and passes on another -- neither may decide whether money is
    spent, so it is refused."""
    monkeypatch.setenv(cli._QDC_BUDGET_ENV, "60")
    _boom_build(monkeypatch)
    rc = cli._qdc_submit(_args(tmp_path, timeout_min=None, yes=True))
    assert rc == 2
    err = capsys.readouterr().err.lower()
    assert "timeout-min" in err


def test_qdc_submit_prints_the_budget_line_exactly_once_via_the_command(
    monkeypatch, tmp_path, capsys
):
    """The reason the duplicated guards use the SILENT `_refusal` forms.

    `_cmd_test_qdc` prints the recorded budget as a courtesy, then calls
    `_qdc_submit`, which now repeats the comparison. Repeating
    `_qdc_budget_guard` instead would print the figure twice on the normal
    route and read as two different checks disagreeing about nothing. Money
    guards are duplicated; the courtesy print is not.
    """
    monkeypatch.setenv(cli._QDC_BUDGET_ENV, "60")
    _boom_build(monkeypatch)
    # Reaching the build is the SUCCESS condition here: it means both frames'
    # guards passed, which is the only situation in which the line could be
    # printed twice.
    with pytest.raises(AssertionError, match="reached the device build"):
        cli._cmd_test_qdc(_args(tmp_path, timeout_min=5))
    out = capsys.readouterr().out
    assert out.count("remaining budget:") == 1, (
        f"the budget line was printed {out.count('remaining budget:')} times"
    )


def test_a_good_run_with_measurements_present_exits_zero(monkeypatch, tmp_path):
    _stub_build_submit_and_wait(monkeypatch)
    paths = _fake_fetch(
        tmp_path,
        results_xml='<testsuite tests="4" failures="0" errors="0" skipped="0"></testsuite>',
        extra_logs={"hexlib_selftest.log": GOOD_LOG},
    )
    monkeypatch.setattr(job, "fetch", lambda job_id, dest: paths)
    rc = cli._qdc_submit(_args(tmp_path))
    assert rc == 0


def test_zero_tests_is_a_failure_never_a_pass(monkeypatch, tmp_path, capsys):
    """THE ORIGINAL DEFECT'S OWN SHAPE: a results.xml that parses clean but
    reports it ran nothing at all must not be a pass."""
    _stub_build_submit_and_wait(monkeypatch)
    paths = _fake_fetch(
        tmp_path,
        results_xml='<testsuite tests="0" failures="0" errors="0" skipped="0"></testsuite>',
        extra_logs={"hexlib_selftest.log": GOOD_LOG},
    )
    monkeypatch.setattr(job, "fetch", lambda job_id, dest: paths)
    rc = cli._qdc_submit(_args(tmp_path))
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "0 test" in err


def test_any_failures_is_a_failure(monkeypatch, tmp_path, capsys):
    _stub_build_submit_and_wait(monkeypatch)
    paths = _fake_fetch(
        tmp_path,
        results_xml='<testsuite tests="5" failures="1" errors="0" skipped="0"></testsuite>',
        extra_logs={"hexlib_selftest.log": GOOD_LOG},
    )
    monkeypatch.setattr(job, "fetch", lambda job_id, dest: paths)
    rc = cli._qdc_submit(_args(tmp_path))
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "failure" in err


def test_any_errors_is_a_failure(monkeypatch, tmp_path, capsys):
    _stub_build_submit_and_wait(monkeypatch)
    paths = _fake_fetch(
        tmp_path,
        results_xml='<testsuite tests="5" failures="0" errors="2" skipped="0"></testsuite>',
        extra_logs={"hexlib_selftest.log": GOOD_LOG},
    )
    monkeypatch.setattr(job, "fetch", lambda job_id, dest: paths)
    rc = cli._qdc_submit(_args(tmp_path))
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "error" in err


def test_an_unparseable_results_xml_is_a_failure(monkeypatch, tmp_path, capsys):
    """A truncated or non-XML results.xml -- e.g. a collection error that
    never produced a real report -- must not be silently skipped."""
    _stub_build_submit_and_wait(monkeypatch)
    paths = _fake_fetch(
        tmp_path,
        results_xml="this is not xml at all <<< not even close",
        extra_logs={"hexlib_selftest.log": GOOD_LOG},
    )
    monkeypatch.setattr(job, "fetch", lambda job_id, dest: paths)
    rc = cli._qdc_submit(_args(tmp_path))
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "pars" in err or "xml" in err


def test_missing_results_xml_entirely_is_a_failure(monkeypatch, tmp_path, capsys):
    _stub_build_submit_and_wait(monkeypatch)
    paths = _fake_fetch(
        tmp_path, results_xml=None, extra_logs={"hexlib_selftest.log": GOOD_LOG}
    )
    monkeypatch.setattr(job, "fetch", lambda job_id, dest: paths)
    rc = cli._qdc_submit(_args(tmp_path))
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "results.xml" in err


def test_a_clean_decoy_results_xml_cannot_mask_the_real_failing_report(
    monkeypatch, tmp_path, capsys
):
    """THE BASENAME SELECTION, WITH BOTH FILES PRESENT AND THE DECOY FIRST.

    `898089c` made `job.fetch` mirror QDC's directory layout precisely because
    `TestLogs/results.xml` and `logs/results.xml` used to overwrite each other
    on disk. From that commit onwards both exist -- and the verdict reader
    still did `os.path.basename(p) == "results.xml"`, taking whichever QDC
    listed first.

    So: `logs/results.xml` is a framework placeholder reporting one passing
    test, listed FIRST; `TestLogs/results.xml` is the real report with three
    failures. The self-test log carries a genuine PASS line and a positive
    `cycles_total=`, because `write_qdc_log` runs before the asserts do -- so
    every other check in `_qdc_check_results` is satisfied and the ONLY thing
    standing between this job and `exit 0` is which file gets parsed.

    A failed device job reported as a pass is this project's own named
    Critical, and this is it on the money path. Both a nonzero exit and the
    reason are asserted: passing for the wrong reason would be worth nothing
    here.
    """
    _stub_build_submit_and_wait(monkeypatch)
    paths = _fake_fetch(
        tmp_path,
        results_xml='<testsuite tests="5" failures="3" errors="0" skipped="0"></testsuite>',
        decoy_results_xml='<testsuite tests="1" failures="0" errors="0" skipped="0"/>',
        extra_logs={"hexlib_selftest.log": GOOD_LOG},
    )
    # The decoy is first in QDC's listing order -- that ordering is the defect's
    # trigger and is not ours to choose, so it is pinned rather than assumed.
    assert paths[0].endswith(os.path.join("logs", "results.xml"))
    monkeypatch.setattr(job, "fetch", lambda job_id, dest: paths)

    rc = cli._qdc_submit(_args(tmp_path))
    assert rc != 0, (
        "a job whose real report has 3 failures exited 0 because a clean "
        "placeholder results.xml was parsed instead"
    )
    err = capsys.readouterr().err.lower()
    assert "results.xml" in err


def test_two_matching_results_files_are_refused_rather_than_guessed_between(
    monkeypatch, tmp_path, capsys
):
    """Ambiguity is a failure, even when BOTH candidates look clean.

    The check above passes as soon as the reader stops preferring the decoy --
    including if it simply preferred the other one. That is still a guess, and
    the next layout change flips it back. So when two files match the marker,
    the reader must refuse and say so, not parse either.

    Both reports here are clean, so nothing except the ambiguity itself can
    make this fail.
    """
    _stub_build_submit_and_wait(monkeypatch)
    clean = '<testsuite tests="4" failures="0" errors="0" skipped="0"></testsuite>'
    log_dir = os.path.join(str(tmp_path / "out"), "qdc_logs")
    os.makedirs(log_dir, exist_ok=True)
    paths = [
        _write_log(log_dir, job.RESULTS_MARKER, clean),
        _write_log(log_dir, "run2/" + job.RESULTS_MARKER, clean),
        _write_log(log_dir, "hexlib_selftest.log", GOOD_LOG),
    ]
    monkeypatch.setattr(job, "fetch", lambda job_id, dest: paths)

    rc = cli._qdc_submit(_args(tmp_path))
    assert rc != 0, (
        "two files matched the results marker and one of them was parsed "
        "anyway -- a verdict read from a guess"
    )
    err = capsys.readouterr().err.lower()
    assert "2 files match" in err or "refusing to guess" in err


def test_a_clean_result_missing_the_measurement_lines_is_still_a_failure(
    monkeypatch, tmp_path, capsys
):
    """A results.xml that parses clean with zero failures is not enough on
    its own -- if the fetched logs never actually show `cycles_total=` or
    the `--self-test` PASS line, that is 'a success value constructible
    with zero measurements inside it', which is exactly what this whole
    check exists to make impossible."""
    _stub_build_submit_and_wait(monkeypatch)
    paths = _fake_fetch(
        tmp_path,
        results_xml='<testsuite tests="4" failures="0" errors="0" skipped="0"></testsuite>',
        extra_logs={"hexlib_selftest.log": "nothing useful in this log\n"},
    )
    monkeypatch.setattr(job, "fetch", lambda job_id, dest: paths)
    rc = cli._qdc_submit(_args(tmp_path))
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "cycles_total" in err or "measurement" in err


def test_a_clean_result_missing_only_the_pass_line_is_still_a_failure(
    monkeypatch, tmp_path, capsys
):
    _stub_build_submit_and_wait(monkeypatch)
    paths = _fake_fetch(
        tmp_path,
        results_xml='<testsuite tests="4" failures="0" errors="0" skipped="0"></testsuite>',
        extra_logs={"hexlib_selftest.log": f"{CYCLES_LINE}\n"},
    )
    monkeypatch.setattr(job, "fetch", lambda job_id, dest: paths)
    rc = cli._qdc_submit(_args(tmp_path))
    assert rc != 0


def test_a_clean_result_missing_only_cycles_total_is_still_a_failure(
    monkeypatch, tmp_path, capsys
):
    _stub_build_submit_and_wait(monkeypatch)
    paths = _fake_fetch(
        tmp_path,
        results_xml='<testsuite tests="4" failures="0" errors="0" skipped="0"></testsuite>',
        extra_logs={"hexlib_selftest.log": f"{PASS_LINE}\n"},
    )
    monkeypatch.setattr(job, "fetch", lambda job_id, dest: paths)
    rc = cli._qdc_submit(_args(tmp_path))
    assert rc != 0


def test_a_testsuite_nested_inside_another_testsuite_is_refused_not_summed(
    monkeypatch, tmp_path, capsys
):
    """Critical 3's own aftermath: `.findall(".//testsuite")` sums every
    `<testsuite>` at ANY depth, so a report with one `<testsuite>` nested
    inside another -- whose parent's own `tests`/`failures`/`errors`
    attributes already include the child's counts -- would be double-counted
    if summed naively. pytest's own `--junitxml` (device/qdc/artifact.py's
    pytest.ini) never produces this shape, so this must be REFUSED as an
    unparseable report (never silently summed into a false pass or a
    misleading count) -- a parse failure here blocks a false pass, which is
    the safe direction to fail in."""
    _stub_build_submit_and_wait(monkeypatch)
    xml = (
        "<testsuites>"
        '<testsuite tests="5" failures="0" errors="0" skipped="0">'
        '<testsuite tests="2" failures="0" errors="0" skipped="0"></testsuite>'
        "</testsuite>"
        "</testsuites>"
    )
    paths = _fake_fetch(
        tmp_path, results_xml=xml, extra_logs={"hexlib_selftest.log": GOOD_LOG}
    )
    monkeypatch.setattr(job, "fetch", lambda job_id, dest: paths)
    rc = cli._qdc_submit(_args(tmp_path))
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "nested" in err or "pars" in err


def test_testsuites_wrapper_with_multiple_suites_is_summed(monkeypatch, tmp_path):
    """pytest's junit-xml can emit a <testsuites> root wrapping one or more
    <testsuite> children -- the counts must be summed across all of them,
    not read only off whichever element happens to be the root."""
    _stub_build_submit_and_wait(monkeypatch)
    xml = (
        '<testsuites>'
        '<testsuite tests="2" failures="0" errors="0" skipped="0"></testsuite>'
        '<testsuite tests="3" failures="0" errors="0" skipped="0"></testsuite>'
        "</testsuites>"
    )
    paths = _fake_fetch(
        tmp_path, results_xml=xml, extra_logs={"hexlib_selftest.log": GOOD_LOG}
    )
    monkeypatch.setattr(job, "fetch", lambda job_id, dest: paths)
    rc = cli._qdc_submit(_args(tmp_path))
    assert rc == 0


# ==============================================================================
# Direct unit tests of `_qdc_parse_results_xml` itself -- pinning the exact
# shape it accepts (a single <testsuite>, or a <testsuites> wrapping
# <testsuite> children directly, matching pytest's own --junitxml output;
# see device/qdc/artifact.py's pytest.ini) and confirming it REFUSES anything
# else -- most importantly a genuinely nested
# <testsuites><testsuite><testsuite> document, which an earlier
# `.findall(".//testsuite")` would have summed and silently double-counted.
# ==============================================================================


def test_parse_bare_testsuite_root(tmp_path):
    p = tmp_path / "results.xml"
    p.write_text('<testsuite tests="4" failures="1" errors="0" skipped="2"></testsuite>')
    assert cli._qdc_parse_results_xml(str(p)) == (4, 1, 0, 2)


def test_parse_testsuites_wrapper_sums_direct_children_only(tmp_path):
    p = tmp_path / "results.xml"
    p.write_text(
        "<testsuites>"
        '<testsuite tests="2" failures="0" errors="1" skipped="1"></testsuite>'
        '<testsuite tests="3" failures="1" errors="0" skipped="0"></testsuite>'
        "</testsuites>"
    )
    assert cli._qdc_parse_results_xml(str(p)) == (5, 1, 1, 1)


def test_parse_refuses_a_testsuite_nested_inside_a_testsuite(tmp_path):
    """THE DOUBLE-COUNT GUARD, tested directly against the parser (see the
    end-to-end version above via `_qdc_submit`). A parent <testsuite>'s own
    tests="5" already includes whatever its nested child's tests="2"
    contributed -- summing both, as `.findall(".//testsuite")` would, reports
    7 when only 5 tests genuinely ran. This must be refused outright rather
    than guessed at."""
    p = tmp_path / "results.xml"
    p.write_text(
        "<testsuites>"
        '<testsuite tests="5" failures="0" errors="0" skipped="0">'
        '<testsuite tests="2" failures="0" errors="0" skipped="0"></testsuite>'
        "</testsuite>"
        "</testsuites>"
    )
    with pytest.raises(cli._QdcResultsError, match="nested"):
        cli._qdc_parse_results_xml(str(p))


def test_parse_refuses_a_testsuite_nested_directly_under_bare_testsuite_root(tmp_path):
    """The same nested shape, but with the outer element as the document
    root itself (no <testsuites> wrapper) -- must be refused the same way,
    not accepted just because the root tag matched the simple case."""
    p = tmp_path / "results.xml"
    p.write_text(
        '<testsuite tests="5" failures="0" errors="0" skipped="0">'
        '<testsuite tests="2" failures="0" errors="0" skipped="0"></testsuite>'
        "</testsuite>"
    )
    with pytest.raises(cli._QdcResultsError, match="nested"):
        cli._qdc_parse_results_xml(str(p))


def test_parse_refuses_an_unrecognized_root_tag(tmp_path):
    p = tmp_path / "results.xml"
    p.write_text('<report tests="4" failures="0" errors="0"></report>')
    with pytest.raises(cli._QdcResultsError, match="testsuite"):
        cli._qdc_parse_results_xml(str(p))


# ==============================================================================
# `cycles_total=0` -- A MEASUREMENT THAT MEASURED NOTHING.
#
# The measurement-lines check was `_CYCLES_TOTAL_MARKER not in combined`, a
# pure substring test. `cycles_total=0` contains `cycles_total=`, so it
# passed: a run in which the DSP's PCYCLE counter never advanced at all
# printed "measurement lines present" and exited 0. That was reproduced
# against fabricated local logs before this fix -- a log carrying
# `hexlib: --self-test: cycles_total=0` and a clean
# `<testsuite tests="5" failures="0" errors="0">` produced
# "job 1: 5 test(s), 0 failures, 0 errors, measurement lines present" and
# return code 0.
#
# It is not a hypothetical shape. The skel reads PCYCLE inside a user-mode
# unsigned PD, where SYSCFG.PCYCLEEN cannot be set (skel_dispatch.c's
# hexlib_read_pcycle, and include/hexlib/hexlib_harness.h, which sets that bit
# explicitly for the standalone runtime because the register reads 0 without
# it). So zero is exactly what the FIRST silicon job would print if the
# counter is dead there -- the single most important thing that job can
# report, and the one thing this check used to swallow.
# ==============================================================================

ZERO_CYCLES_LINE = "hexlib: --self-test: cycles_total=0"


def test_a_zero_cycle_count_is_a_failure_never_a_pass(monkeypatch, tmp_path, capsys):
    """THE DEFECT ITSELF. The PASS line is present, results.xml is clean, and
    `cycles_total=` is literally in the logs -- and this must still fail,
    because the value is 0 and a run that measured nothing did not measure
    anything."""
    _stub_build_submit_and_wait(monkeypatch)
    paths = _fake_fetch(
        tmp_path,
        results_xml='<testsuite tests="5" failures="0" errors="0" skipped="0"></testsuite>',
        extra_logs={"hexlib_selftest.log": f"{PASS_LINE}\n{ZERO_CYCLES_LINE}\n"},
    )
    monkeypatch.setattr(job, "fetch", lambda job_id, dest: paths)
    rc = cli._qdc_submit(_args(tmp_path))
    assert rc != 0, (
        "cycles_total=0 satisfied the old substring check and exited 0 -- a "
        "success value with a literal zero measurement inside it"
    )
    err = capsys.readouterr().err.lower()
    assert "cycles_total" in err
    assert "0" in err


def test_a_malformed_cycle_count_is_a_failure(monkeypatch, tmp_path, capsys):
    """A `cycles_total=` whose value is not a decimal integer at all -- a
    truncated log, interleaved output, or a format change nobody updated this
    for. Must fail, and must say it is malformed rather than reporting it as
    absent (it is not absent) or as zero (it is not zero)."""
    _stub_build_submit_and_wait(monkeypatch)
    paths = _fake_fetch(
        tmp_path,
        results_xml='<testsuite tests="5" failures="0" errors="0" skipped="0"></testsuite>',
        extra_logs={
            "hexlib_selftest.log": f"{PASS_LINE}\nhexlib: cycles_total=<gar\n"
        },
    )
    monkeypatch.setattr(job, "fetch", lambda job_id, dest: paths)
    rc = cli._qdc_submit(_args(tmp_path))
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "cycles_total" in err
    assert "integer" in err or "malform" in err


def test_a_positive_cycle_count_alongside_a_zero_one_still_passes(monkeypatch, tmp_path):
    """One job's logs carry SEVERAL `cycles_total=` lines -- `--self-test`
    prints one and `--coherency-check` prints another -- and there is no
    requirement that every mode a job ran produced a nonzero count. At least
    one genuine measurement is the bar. If PCYCLE were dead in the unsigned
    PD, every line would read 0 and this leniency could not hide it, which is
    what the test above pins."""
    _stub_build_submit_and_wait(monkeypatch)
    paths = _fake_fetch(
        tmp_path,
        results_xml='<testsuite tests="5" failures="0" errors="0" skipped="0"></testsuite>',
        extra_logs={
            "hexlib_selftest.log": f"{PASS_LINE}\n{ZERO_CYCLES_LINE}\n{CYCLES_LINE}\n"
        },
    )
    monkeypatch.setattr(job, "fetch", lambda job_id, dest: paths)
    assert cli._qdc_submit(_args(tmp_path)) == 0


# Direct unit tests of the verdict function, so each of its four states is
# pinned without going through the whole submit path.


def test_cycles_verdict_absent():
    ok, detail = cli._qdc_cycles_total_verdict("nothing measured here at all\n")
    assert ok is False
    assert "anywhere" in detail


def test_cycles_verdict_zero():
    ok, detail = cli._qdc_cycles_total_verdict("hexlib: cycles_total=0\n")
    assert ok is False
    assert "0" in detail


def test_cycles_verdict_negative_is_malformed_not_a_measurement():
    """`cycles_total` is printed with `%llu` (main.c), so a minus sign cannot
    come from a healthy run. It must not parse as an integer that then fails
    the `> 0` test for the WRONG stated reason, and it must certainly not be
    accepted."""
    ok, detail = cli._qdc_cycles_total_verdict("hexlib: cycles_total=-5\n")
    assert ok is False
    assert "integer" in detail


def test_cycles_verdict_malformed():
    ok, detail = cli._qdc_cycles_total_verdict("hexlib: cycles_total=abc\n")
    assert ok is False
    assert "integer" in detail


def test_cycles_verdict_positive():
    ok, detail = cli._qdc_cycles_total_verdict("hexlib: cycles_total=1287\n")
    assert ok is True
    assert "1287" in detail


def test_cycles_verdict_takes_the_largest_positive_value():
    ok, detail = cli._qdc_cycles_total_verdict(
        "cycles_total=0\ncycles_total=42\ncycles_total=1287\n"
    )
    assert ok is True
    assert "1287" in detail


# ==============================================================================
# `skipped` -- COLLECTED BUT NEVER RUN IS NOT PASSED.
#
# `skipped` was parsed NOWHERE. VERIFIED before this fix:
# `<testsuite tests="5" failures="0" errors="0" skipped="5">` plus a good
# self-test log printed "job 999: 5 test(s), 0 failures, 0 errors, measurement
# lines present" and EXITED 0 -- five tests collected, none run, reported as a
# clean pass, with the skip count not even mentioned in the output.
#
# It is reachable the moment anyone `skipif`s the two aspirational
# discriminators in device/qdc/test_on_device.py (the unmapped-fd and the
# coherency check), and commit 6ac7c3e ("stop skipping silently") is the record
# of that being a live temptation. On device a skip cannot mean "not applicable
# here": every test in that file is unconditional, so a skip means the farm
# collected a test and then did not run it.
# ==============================================================================


def test_every_test_skipped_is_a_failure_never_a_pass(monkeypatch, tmp_path, capsys):
    """THE DEFECT ITSELF: a report whose every test was skipped, with a
    perfectly good measurement log beside it."""
    _stub_build_submit_and_wait(monkeypatch)
    paths = _fake_fetch(
        tmp_path,
        results_xml='<testsuite tests="5" failures="0" errors="0" skipped="5"></testsuite>',
        extra_logs={"hexlib_selftest.log": GOOD_LOG},
    )
    monkeypatch.setattr(job, "fetch", lambda job_id, dest: paths)
    rc = cli._qdc_submit(_args(tmp_path))
    assert rc != 0, (
        "5 collected, 5 skipped, 0 run exited 0 before this fix -- a skip is "
        "not a pass, and on device it means the test could not run at all"
    )
    err = capsys.readouterr().err.lower()
    assert "skip" in err
    assert "5" in err


def test_even_one_skipped_test_is_a_failure(monkeypatch, tmp_path, capsys):
    """Not a threshold. One test silently not running is one discriminator
    silently not applied."""
    _stub_build_submit_and_wait(monkeypatch)
    paths = _fake_fetch(
        tmp_path,
        results_xml='<testsuite tests="5" failures="0" errors="0" skipped="1"></testsuite>',
        extra_logs={"hexlib_selftest.log": GOOD_LOG},
    )
    monkeypatch.setattr(job, "fetch", lambda job_id, dest: paths)
    rc = cli._qdc_submit(_args(tmp_path))
    assert rc != 0
    assert "skip" in capsys.readouterr().err.lower()


def test_the_skip_count_is_reported_on_the_pass_line_too(monkeypatch, tmp_path, capsys):
    """A success line that omits a count it checked leaves a reader unable to
    tell "0 skipped" from "skips were never looked at" -- which is exactly the
    state this line was in."""
    _stub_build_submit_and_wait(monkeypatch)
    paths = _fake_fetch(
        tmp_path,
        results_xml='<testsuite tests="4" failures="0" errors="0" skipped="0"></testsuite>',
        extra_logs={"hexlib_selftest.log": GOOD_LOG},
    )
    monkeypatch.setattr(job, "fetch", lambda job_id, dest: paths)
    assert cli._qdc_submit(_args(tmp_path)) == 0
    assert "0 skipped" in capsys.readouterr().out


# ==============================================================================
# A MISSING COUNT ATTRIBUTE IS A MALFORMED REPORT, NEVER A ZERO.
#
# `suite.get("failures", "0")` / `suite.get("errors", "0")` defaulted the two
# attributes that decide the verdict. VERIFIED before this fix:
# `<testsuite tests="5"></testsuite>` plus a good log EXITED 0 and printed
# "5 test(s), 0 failures, 0 errors". That directly contradicted
# `_QdcResultsError`'s own docstring, which says the exception exists for a
# report "missing the attributes a JUnit report always carries".
# ==============================================================================


@pytest.mark.parametrize("xml", [
    '<testsuite tests="5"></testsuite>',
    '<testsuite tests="5" errors="0" skipped="0"></testsuite>',        # no failures
    '<testsuite tests="5" failures="0" skipped="0"></testsuite>',      # no errors
    '<testsuite tests="5" failures="0" errors="0"></testsuite>',       # no skipped
    '<testsuite failures="0" errors="0" skipped="0"></testsuite>',     # no tests
])
def test_a_results_xml_missing_any_count_attribute_is_a_failure(
    monkeypatch, tmp_path, capsys, xml
):
    _stub_build_submit_and_wait(monkeypatch)
    paths = _fake_fetch(
        tmp_path, results_xml=xml, extra_logs={"hexlib_selftest.log": GOOD_LOG}
    )
    monkeypatch.setattr(job, "fetch", lambda job_id, dest: paths)
    rc = cli._qdc_submit(_args(tmp_path))
    assert rc != 0, f"{xml} must not be read as a clean report"
    assert "attribute" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("attr", ["tests", "failures", "errors", "skipped"])
def test_parse_refuses_a_suite_missing_any_required_attribute(tmp_path, attr):
    attrs = {"tests": "5", "failures": "0", "errors": "0", "skipped": "0"}
    del attrs[attr]
    body = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    p = tmp_path / "results.xml"
    p.write_text(f"<testsuite {body}></testsuite>")
    with pytest.raises(cli._QdcResultsError, match=attr):
        cli._qdc_parse_results_xml(str(p))


def test_parse_refuses_a_missing_attribute_on_a_later_suite_too(tmp_path):
    """Summing across a <testsuites> wrapper must not let a well-formed first
    suite cover for a malformed second one."""
    p = tmp_path / "results.xml"
    p.write_text(
        "<testsuites>"
        '<testsuite tests="2" failures="0" errors="0" skipped="0"></testsuite>'
        '<testsuite tests="3" errors="0" skipped="0"></testsuite>'
        "</testsuites>"
    )
    with pytest.raises(cli._QdcResultsError, match="failures"):
        cli._qdc_parse_results_xml(str(p))


# ==============================================================================
# NEGATIVE AND ZERO COUNTS.
#
# `if tests == 0` was the whole bound. VERIFIED before this fix:
# `<testsuite tests="-1" failures="0" errors="0">` plus a good log EXITED 0 and
# printed "-1 test(s)".
# ==============================================================================


@pytest.mark.parametrize("xml", [
    '<testsuite tests="-1" failures="0" errors="0" skipped="0"></testsuite>',
    '<testsuite tests="5" failures="-1" errors="0" skipped="0"></testsuite>',
    '<testsuite tests="5" failures="0" errors="-2" skipped="0"></testsuite>',
    '<testsuite tests="5" failures="0" errors="0" skipped="-1"></testsuite>',
])
def test_a_negative_count_anywhere_is_a_failure(monkeypatch, tmp_path, capsys, xml):
    _stub_build_submit_and_wait(monkeypatch)
    paths = _fake_fetch(
        tmp_path, results_xml=xml, extra_logs={"hexlib_selftest.log": GOOD_LOG}
    )
    monkeypatch.setattr(job, "fetch", lambda job_id, dest: paths)
    assert cli._qdc_submit(_args(tmp_path)) != 0, f"{xml} must not read as a pass"


def test_parse_refuses_a_negative_count(tmp_path):
    p = tmp_path / "results.xml"
    p.write_text('<testsuite tests="-1" failures="0" errors="0" skipped="0"></testsuite>')
    with pytest.raises(cli._QdcResultsError, match="NEGATIVE"):
        cli._qdc_parse_results_xml(str(p))


# ==============================================================================
# THE PARSER STRICTNESS IS CHECKED AGAINST A REPORT PYTEST ACTUALLY WROTE.
#
# Requiring all four attributes is only safe if the producer really emits all
# four. Rather than assert that from memory, this runs pytest's own --junitxml
# and reads the result back: if a future pytest stops emitting one of them,
# this fails HERE (with a report in hand) instead of the device gate refusing
# every genuine results.xml after the minutes are spent.
# ==============================================================================


def test_a_real_pytest_junitxml_carries_all_four_counts_and_parses(tmp_path):
    import subprocess
    import sys
    import xml.etree.ElementTree as ET

    probe = tmp_path / "test_probe.py"
    probe.write_text(
        "import pytest\n"
        "def test_pass(): pass\n"
        '@pytest.mark.skip(reason="probe")\n'
        "def test_skipped(): pass\n"
    )
    xml_path = tmp_path / "results.xml"
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         f"--junitxml={xml_path}", str(probe)],
        cwd=str(tmp_path), capture_output=True, text=True,
    )
    assert xml_path.is_file(), "pytest wrote no junitxml at all"

    root = ET.parse(str(xml_path)).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    for attr in cli._REQUIRED_SUITE_ATTRS:
        assert suite.get(attr) is not None, (
            f"pytest's own --junitxml did not emit {attr!r} -- cli.py requires "
            "it, so this is the check that must fail, not the device gate "
            "after the minutes are spent"
        )

    counts = cli._qdc_parse_results_xml(str(xml_path))
    assert counts.tests == 2
    assert counts.skipped == 1
    assert (counts.failures, counts.errors) == (0, 0)


# ==============================================================================
# THE WAIT CAP COMES FROM THE JOB'S OWN TIMEOUT, AND A TIMEOUT FETCHES LOGS.
#
# `job.wait(job_id)` was called with no cap_s, taking job.py's 1800 s default
# while `submit()` accepts 240 minutes. Scenario: `--timeout-min 60` on a job
# that legitimately finishes at 35 minutes -- at 30 minutes the CLI printed
# "produced no results.xml within the wait cap", exited 1, and downloaded ZERO
# log files. The minutes were spent, the results.xml that appeared five minutes
# later was never fetched, and the operator had nothing to diagnose from.
# ==============================================================================


def test_the_wait_cap_is_derived_from_the_jobs_own_timeout(monkeypatch, tmp_path):
    seen = {}

    def recording_wait(job_id, **kw):
        seen.update(kw)
        return True

    _stub_build_submit_and_wait(monkeypatch)
    monkeypatch.setattr(job, "wait", recording_wait)
    paths = _fake_fetch(
        tmp_path,
        results_xml='<testsuite tests="4" failures="0" errors="0" skipped="0"></testsuite>',
        extra_logs={"hexlib_selftest.log": GOOD_LOG},
    )
    monkeypatch.setattr(job, "fetch", lambda job_id, dest: paths)
    assert cli._qdc_submit(_args(tmp_path, timeout_min=60)) == 0
    assert seen.get("cap_s", 0) >= 60 * 60, (
        "the wait cap must be at least the job's own timeout -- 1800s against "
        f"a 60-minute job abandons it half way; got {seen}"
    )


def test_the_wait_cap_covers_the_largest_timeout_submit_accepts():
    from hexlib.device.qdc import job as jobmod

    assert cli._qdc_wait_cap_s(jobmod.MAX_TIMEOUT_MIN) > jobmod.MAX_TIMEOUT_MIN * 60
    assert cli._qdc_wait_cap_s(1) > 60


def test_a_wait_timeout_still_fetches_whatever_logs_exist(monkeypatch, tmp_path, capsys):
    """A timeout that discards the evidence is worse than one that waits. The
    minutes are already spent; the partial logs are all the operator has."""
    _stub_build_submit_and_wait(monkeypatch)
    monkeypatch.setattr(job, "wait", lambda job_id, **kw: False)
    fetched = []
    paths = _fake_fetch(tmp_path, results_xml=None,
                        extra_logs={"logcat.txt": "some device noise\n"})

    def recording_fetch(job_id, dest):
        fetched.append(dest)
        return paths

    monkeypatch.setattr(job, "fetch", recording_fetch)
    rc = cli._qdc_submit(_args(tmp_path, timeout_min=60))
    assert rc != 0, "no results.xml within the cap is a failure, never a pass"
    assert fetched, (
        "the timeout branch downloaded ZERO log files before giving up -- the "
        "minutes are spent and this is the only evidence there is"
    )
    err = capsys.readouterr().err
    assert "results.xml" in err
    assert "1 log file" in err


def test_a_wait_timeout_whose_log_fetch_also_fails_still_fails_cleanly(
    monkeypatch, tmp_path, capsys
):
    """Best-effort means best-effort: the fetch blowing up on the giving-up
    path must not turn an exit-1 into a traceback."""
    _stub_build_submit_and_wait(monkeypatch)
    monkeypatch.setattr(job, "wait", lambda job_id, **kw: False)

    def boom_fetch(job_id, dest):
        raise job.QdcError("QDC refused the log listing")

    monkeypatch.setattr(job, "fetch", boom_fetch)
    rc = cli._qdc_submit(_args(tmp_path, timeout_min=20))
    assert rc == 1
    err = capsys.readouterr().err
    assert "results.xml" in err
    assert "QDC refused the log listing" in err


def test_a_fetch_failure_on_the_happy_path_is_a_failure_never_a_pass(
    monkeypatch, tmp_path, capsys
):
    _stub_build_submit_and_wait(monkeypatch)

    def boom_fetch(job_id, dest):
        raise job.QdcError("connection reset while downloading")

    monkeypatch.setattr(job, "fetch", boom_fetch)
    rc = cli._qdc_submit(_args(tmp_path))
    assert rc != 0
    assert "fetch" in capsys.readouterr().err.lower()


# ==============================================================================
# `_qdc_submit` READS args.kernel. It is the function that spends the minutes,
# and it is reachable without going through `_cmd_test_qdc`'s guards at all --
# this file's own `_args` used to prove that by omitting `kernel` entirely.
# ==============================================================================


@pytest.mark.parametrize("kernel", ["add_fp16", "kernels/rmsnorm_fp16", "", None])
def test_qdc_submit_refuses_a_kernel_stage_three_cannot_run(
    monkeypatch, tmp_path, capsys, kernel
):
    def boom_build(build_dir, sdk_root=None):
        raise AssertionError("the device build must not start for a bad kernel")

    monkeypatch.setattr(runtime_build, "build_device_binary", boom_build)
    rc = cli._qdc_submit(_args(tmp_path, kernel=kernel))
    assert rc == 2
    assert "scale_fp16" in capsys.readouterr().err


def test_qdc_submit_refuses_an_args_object_with_no_kernel_attribute(
    monkeypatch, tmp_path, capsys
):
    """The exact shape this file used to pass in. A missing attribute is a
    refusal, not a default."""
    def boom_build(build_dir, sdk_root=None):
        raise AssertionError("the device build must not start for a bad kernel")

    monkeypatch.setattr(runtime_build, "build_device_binary", boom_build)
    args = argparse.Namespace(out=str(tmp_path / "out"), timeout_min=5, yes=False)
    assert cli._qdc_submit(args) == 2
    assert "scale_fp16" in capsys.readouterr().err

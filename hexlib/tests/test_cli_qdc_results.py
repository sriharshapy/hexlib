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


def _args(tmp_path):
    return argparse.Namespace(out=str(tmp_path / "out"), timeout_min=5, yes=False)


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

    def fake_stage(binaries, test_script, out_base):
        zip_path = out_base + ".zip"
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        open(zip_path, "wb").close()
        return zip_path

    monkeypatch.setattr(runtime_build, "build_device_binary", fake_build_device_binary)
    monkeypatch.setattr(artifact, "stage", fake_stage)
    monkeypatch.setattr(job, "submit", lambda zip_path, *, timeout_min: 999)
    monkeypatch.setattr(job, "wait", lambda job_id, **kw: True)


def _fake_fetch(tmp_path, *, results_xml, extra_logs=None):
    """Write fabricated fetched log files under out/qdc_logs and return the
    list of local paths -- the same shape `job.fetch`'s real return value
    has (job.py's own `fetch()` returns exactly this: local paths it wrote
    from QDC's log files)."""
    log_dir = os.path.join(str(tmp_path / "out"), "qdc_logs")
    os.makedirs(log_dir, exist_ok=True)
    paths = []
    if results_xml is not None:
        p = os.path.join(log_dir, "results.xml")
        with open(p, "w", encoding="utf-8") as f:
            f.write(results_xml)
        paths.append(p)
    for name, text in (extra_logs or {}).items():
        p = os.path.join(log_dir, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        paths.append(p)
    return paths


def test_a_good_run_with_measurements_present_exits_zero(monkeypatch, tmp_path):
    _stub_build_submit_and_wait(monkeypatch)
    paths = _fake_fetch(
        tmp_path,
        results_xml='<testsuite tests="4" failures="0" errors="0"></testsuite>',
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
        results_xml='<testsuite tests="0" failures="0" errors="0"></testsuite>',
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
        results_xml='<testsuite tests="5" failures="1" errors="0"></testsuite>',
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
        results_xml='<testsuite tests="5" failures="0" errors="2"></testsuite>',
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
        results_xml='<testsuite tests="4" failures="0" errors="0"></testsuite>',
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
        results_xml='<testsuite tests="4" failures="0" errors="0"></testsuite>',
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
        results_xml='<testsuite tests="4" failures="0" errors="0"></testsuite>',
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
        '<testsuite tests="5" failures="0" errors="0">'
        '<testsuite tests="2" failures="0" errors="0"></testsuite>'
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
        '<testsuite tests="2" failures="0" errors="0"></testsuite>'
        '<testsuite tests="3" failures="0" errors="0"></testsuite>'
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
    p.write_text('<testsuite tests="4" failures="1" errors="0"></testsuite>')
    assert cli._qdc_parse_results_xml(str(p)) == (4, 1, 0)


def test_parse_testsuites_wrapper_sums_direct_children_only(tmp_path):
    p = tmp_path / "results.xml"
    p.write_text(
        "<testsuites>"
        '<testsuite tests="2" failures="0" errors="1"></testsuite>'
        '<testsuite tests="3" failures="1" errors="0"></testsuite>'
        "</testsuites>"
    )
    assert cli._qdc_parse_results_xml(str(p)) == (5, 1, 1)


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
        '<testsuite tests="5" failures="0" errors="0">'
        '<testsuite tests="2" failures="0" errors="0"></testsuite>'
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
        '<testsuite tests="5" failures="0" errors="0">'
        '<testsuite tests="2" failures="0" errors="0"></testsuite>'
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
        results_xml='<testsuite tests="5" failures="0" errors="0"></testsuite>',
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
        results_xml='<testsuite tests="5" failures="0" errors="0"></testsuite>',
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
        results_xml='<testsuite tests="5" failures="0" errors="0"></testsuite>',
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

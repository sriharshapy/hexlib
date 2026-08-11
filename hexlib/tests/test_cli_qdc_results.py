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

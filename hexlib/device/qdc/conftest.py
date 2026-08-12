"""Runs ON THE DEVICE, beside `test_on_device.py`, under the farm's own pytest.

WHY THIS FILE EXISTS. `pytest.ini` (written by `artifact.py`) points
`--junitxml` at the RELATIVE path `TestLogs/results.xml`, which lands in
whatever directory QDC's runner happens to invoke pytest from. QDC does not
collect that. It collects `/data/local/tmp/QDC_logs` -- which is precisely
what `utils.write_qdc_log` writes to, and which NOTHING called with the
report until this file existed.

That gap is not theoretical: job 756124 (2026-08-12, the first hexlib job QDC
ever accepted) reached state Completed and returned exactly one log file --
a stale `LauncherUI` log from an unrelated job four days earlier. No
`TestLogs/results.xml`, so `wait()` polled for its whole cap and returned
False on a job that may well have run correctly. The report was written; it
just never left the device.

`llama.cpp`'s own QDC runner (`scripts/snapdragon/qdc/tests/conftest.py`)
solves the same problem the same way, and is the working reference this was
matched to. It does the copy in `pytest_sessionfinish`; this uses
`pytest_unconfigure`, which is strictly later -- the junitxml plugin writes
the file during its OWN `pytest_sessionfinish`, and two hookimpls for one hook
have no ordering guarantee worth betting a device job on.

THE SUBDIRECTORY IS LOAD-BEARING. `job.wait()` matches a log whose name ENDS
WITH `TestLogs/results.xml`, and QDC lists collected logs as
`<job_id>/<name>`. Writing a flat `results.xml` would be listed as
`756124/results.xml`, which does not match that suffix, and `wait()` would
miss a report that had arrived. The nesting here and `RESULTS_MARKER` in
job.py are one decision recorded in two places.

FAIL CLOSED. If the junitxml is missing or unreadable, this writes a file at
the same path SAYING SO rather than writing nothing. Writing nothing is
indistinguishable from the job never finishing, costs the full wait cap, and
tells the operator nothing; a file that exists and does not parse is caught
immediately by `cli._qdc_check_results` and names its own cause.
"""
import os
import traceback

from utils import QDC_LOG_DIR, write_qdc_log

_RESULTS_NAME = os.path.join("TestLogs", "results.xml")


def _copy_report(config):
    xml_path = getattr(config.option, "xmlpath", None)
    if not xml_path:
        return (
            "<!-- no --junitxml path was configured, so pytest wrote no report. "
            "artifact.py's pytest.ini is what sets it. -->"
        )
    if not os.path.exists(xml_path):
        return (
            f"<!-- pytest was told to write {xml_path} and no such file exists "
            f"at the end of the run. The session most likely died before the "
            f"junitxml plugin wrote it. -->"
        )
    try:
        with open(xml_path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return f"<!-- reading {xml_path} raised:\n{traceback.format_exc()}\n-->"


def pytest_unconfigure(config):
    """Copy the JUnit report into QDC's collected log directory.

    Never raises: an exception here would be reported as an error in the
    runner's own teardown, on a path whose entire job is to make the real
    result visible. Any failure is written to a second log instead.
    """
    try:
        write_qdc_log(_RESULTS_NAME, _copy_report(config))
    except Exception:
        try:
            write_qdc_log(
                "hexlib_conftest_error.txt",
                "copying the junit report into "
                f"{QDC_LOG_DIR} raised:\n{traceback.format_exc()}",
            )
        except Exception:
            pass

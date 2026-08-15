# hexlib/device/qdc/artifact.py
"""Stage the stage-2 binaries and the on-device pytest into a zip QDC can run.

The zip is uploaded as a flat TestScript (see job.py's _real_upload_artifact
for why that artifact type and not TestPackage): hexlib_run, libhexlib_iface_skel.so, and the
on-device test script sit next to a pytest.ini and requirements.txt, matching
what TestFramework.APPIUM finds once QDC extracts it at /qdc/appium. There is
no subdirectory nesting here on purpose -- the on-farm scripts invoke a plain
`adb`, and a path that only exists relative to some assumed staging root is
exactly the kind of thing that fails silently on real hardware and nowhere
else.

StagingError is raised the moment any declared input is missing OR EMPTY, and
again if -- somehow -- something staged does not make it into the zip. A job
that runs against a binary that silently wasn't there is how you burn device
minutes for nothing.

WHY EMPTINESS IS CHECKED AND NOT JUST EXISTENCE. `os.path.isfile` was the
whole test, and `stage` was verified to accept four 0-byte files and produce a
perfectly submittable zip. A link or a copy that fails part-way leaves exactly
that: a `libhexlib_iface_skel.so` of length zero, present, named correctly, and
completely unrunnable -- discovered on the device, after the minutes are spent,
as a dlopen failure with no obvious cause. Size zero is the one truncation
that is unambiguous and free to detect here; deeper validation (ELF magic,
machine type) deliberately is NOT done in this function, because `binaries`
also legitimately carries a `.py` file (see cli.py's own call, which stages
`utils.py` through this list) and a check that has to special-case its inputs
by extension is a check that will be wrong about the next input added.
"""
from __future__ import annotations

import os
import shutil
import zipfile

# --junitxml=results.xml, FLAT. The framework publishes pytest's report itself
# as `<job>/<subid>/TestLogs/results.xml` -- that path is QDC's, not ours, and
# it is what job.RESULTS_MARKER matches. Verified against two working jobs on
# this account (744001 hexbench, 743551 llama.cpp), both of which set exactly
# this and both of which have a TestLogs/results.xml.
#
# Writing `--junitxml=TestLogs/results.xml` ourselves, as this used to, is
# actively harmful: conftest.py also pushes a copy into QDC_logs, and two
# collected files whose names both end in `TestLogs/results.xml` is the
# "TWO MATCHES IS A FAILURE" case cli._qdc_check_results deliberately refuses.
_PYTEST_INI = "[pytest]\naddopts = --junitxml=results.xml\n"
# Appium-Python-Client is here because the job runs under
# TestFramework.APPIUM and conftest.py opens a session with it; the version is
# the one llama.cpp's own QDC runner pins against this same account.
_REQUIREMENTS = "pytest\nAppium-Python-Client==5.2.4\n"


class StagingError(Exception):
    """A declared binary or test script does not exist, is empty, or did not
    survive into the zip. Never produce an artifact that is missing what it
    claims to carry."""


def _require_real_file(path: str, what: str) -> None:
    """Present AND non-empty. See the module docstring for why the second
    half is not pedantry."""
    if not os.path.isfile(path):
        raise StagingError(f"{what} not found: {path}")
    if os.path.getsize(path) == 0:
        raise StagingError(
            f"{what} is 0 bytes, refusing to stage it: {path} -- an empty "
            "artifact is what a failed link or a truncated copy leaves "
            "behind, and it would be discovered on the device after the "
            "minutes are spent"
        )


def stage(
    binaries: list[str],
    test_script: str | None,
    out_base: str,
    support_files: list[str] | None = None,
) -> str:
    """Stage `binaries` under `bin/`, `test_script` and `support_files` under
    `tests/`, generate `pytest.ini` and `requirements.txt` at the root, zip it
    all to `<out_base>.zip`, and return that path.

    THE LAYOUT IS COPIED FROM A JOB THAT WORKS, not chosen. Job 744001
    (hexbench, this account, this device) has pytest report
    `rootdir: /qdc/appium`, `configfile: pytest.ini`, collect
    `tests/test_capprobe.py`, and push `/qdc/appium/bin/capprobe`. Job 743551
    (llama.cpp) has the same shape.

    A FLAT ZIP DOES NOT RUN. hexlib's first three jobs (756124, 756159,
    756206) staged everything at the root, were accepted, dispatched, and
    reached Completed having produced NO `<job>/<subid>/` tree at all -- no
    TestLogs, no install.txt, no screen recording -- which is what a job whose
    test stage never started looks like. Nothing inside the tests could have
    mattered while that was true.

    Raises StagingError if any input is missing or empty, or if the zip that
    would result is missing anything that was staged.
    """
    support_files = list(support_files or [])
    for b in binaries:
        _require_real_file(b, "binary")
    for s in support_files:
        _require_real_file(s, "support file")
    if test_script is not None:
        _require_real_file(test_script, "test script")

    stage_dir = out_base + "_stage"
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)
    os.makedirs(stage_dir, exist_ok=True)

    # (absolute path on disk, name inside the zip) -- the second is what the
    # runner sees, and it is the whole point of this function.
    staged: list[tuple[str, str]] = []

    def _place(src: str, subdir: str) -> None:
        rel = os.path.join(subdir, os.path.basename(src)) if subdir else os.path.basename(src)
        dest = os.path.join(stage_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src, dest)
        staged.append((dest, rel.replace(os.sep, "/")))

    for src in binaries:
        _place(src, "bin")
    for src in support_files:
        _place(src, "tests")
    if test_script is not None:
        _place(test_script, "tests")

    for name, body in (("pytest.ini", _PYTEST_INI),
                       ("requirements.txt", _REQUIREMENTS)):
        path = os.path.join(stage_dir, name)
        with open(path, "w") as f:
            f.write(body)
        staged.append((path, name))

    zip_path = out_base + ".zip"
    zip_dir = os.path.dirname(zip_path)
    if zip_dir:
        os.makedirs(zip_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, arcname in staged:
            zf.write(path, arcname)

    names = set(zipfile.ZipFile(zip_path).namelist())
    missing = [arc for _, arc in staged if arc not in names]
    if missing:
        raise StagingError(f"declared file(s) missing from zip: {missing}")

    return zip_path

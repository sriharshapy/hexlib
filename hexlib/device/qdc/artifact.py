# hexlib/device/qdc/artifact.py
"""Stage the stage-2 binaries and the on-device pytest into a zip QDC can run.

The zip is uploaded as a flat TestScript (see job.py's _real_upload_artifact
for why that artifact type and not TestPackage): hexlib_run, libhexlib_skel.so, and the
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
that: a `libhexlib_skel.so` of length zero, present, named correctly, and
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

_PYTEST_INI = "[pytest]\naddopts = --junitxml=TestLogs/results.xml\n"
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


def stage(binaries: list[str], test_script: str | None, out_base: str) -> str:
    """Copy `binaries` (and `test_script`, if given) into a staging tree
    next to a generated pytest.ini and requirements.txt, zip it to
    `<out_base>.zip`, and return that path.

    Raises StagingError if any input is missing or empty, or if the zip that
    would result is missing anything that was staged.
    """
    for b in binaries:
        _require_real_file(b, "binary")
    if test_script is not None:
        _require_real_file(test_script, "test script")

    stage_dir = out_base + "_stage"
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)
    os.makedirs(stage_dir, exist_ok=True)

    staged = []
    for src in binaries:
        dest = os.path.join(stage_dir, os.path.basename(src))
        shutil.copy2(src, dest)
        staged.append(dest)

    if test_script is not None:
        dest = os.path.join(stage_dir, os.path.basename(test_script))
        shutil.copy2(test_script, dest)
        staged.append(dest)

    pytest_ini = os.path.join(stage_dir, "pytest.ini")
    with open(pytest_ini, "w") as f:
        f.write(_PYTEST_INI)
    staged.append(pytest_ini)

    requirements = os.path.join(stage_dir, "requirements.txt")
    with open(requirements, "w") as f:
        f.write(_REQUIREMENTS)
    staged.append(requirements)

    zip_path = out_base + ".zip"
    zip_dir = os.path.dirname(zip_path)
    if zip_dir:
        os.makedirs(zip_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in staged:
            zf.write(path, os.path.basename(path))

    names = set(zipfile.ZipFile(zip_path).namelist())
    missing = [p for p in staged if os.path.basename(p) not in names]
    if missing:
        raise StagingError(f"declared file(s) missing from zip: {missing}")

    return zip_path

# hexlib/device/qdc/utils.py
"""Helpers for `test_on_device.py`, staged into the SAME flat TestPackage
directory (see `artifact.py`'s own "no subdirectory nesting" reasoning) and
imported there as a bare top-level module -- `from utils import sh,
write_qdc_log` -- because there is no `hexlib` package once QDC's runner
extracts the zip; only whatever files were zipped up sit next to each other.

THIS FILE RUNS ON THE DEVICE, alongside `test_on_device.py`, under the farm's
own pytest -- never under `hexlib/tests`. See that file's own module
docstring for the exclusion this project relies on, and note that this module
would be just as uncollectable there: it assumes a POSIX shell and
`/data/local/tmp` exist, neither of which is true of the machine running our
own offline suite.

`sh()` WRAPS EVERY COMMAND IN `adb shell`, AND THAT IS A CORRECTION.

This file used to argue the opposite at length: that QDC provisions a Python
ON the device and runs the TestPackage there, so on-device paths could be
named directly with no `adb` prefix. That reasoning ended with "if that
assumption is ever wrong, the fix belongs in `sh()`, in one place." It was
wrong, and this is that one place.

WHAT SETTLED IT. llama.cpp's own QDC runner -- the working reference on this
same account and framework -- reaches the device exclusively through `adb`:
`run_adb_command` is `adb shell "<cmd>; echo __RC__:$?"`, its `write_qdc_log`
does `adb push` into `/data/local/tmp/QDC_logs`, and its `SCRIPTS_DIR`
(`/qdc/appium`) is a HOST path holding the extracted zip. pytest runs on the
QDC RUNNER, not on the phone.

WHAT THE OLD ASSUMPTION COST. Jobs 756124 and 756159 (2026-08-12) both
reached Completed and returned no logs of their own. Under the old model
`write_qdc_log` wrote to `/data/local/tmp/QDC_logs` ON THE RUNNER, a
directory QDC never collects because it collects that path from the DEVICE --
so a report could be written perfectly and still be invisible. Worse,
`test_binaries_are_present_and_executable` ran a plain `cp`, which on a Linux
runner SUCCEEDS at copying an AArch64 binary into a host directory, and the
failure only surfaces later as an exec-format error on a file that "landed"
correctly.

So: `sh()` runs on the device, `push()` moves staged files there, and
`write_qdc_log` pushes the report to where QDC actually collects from.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

QDC_LOG_DIR = "/data/local/tmp/QDC_logs"

# Where QDC extracted the artifact ON THE RUNNER. Derived from this file's own
# location rather than hardcoded to `/qdc/appium`: that is the documented
# extraction point, but this module is the one thing guaranteed to sit beside
# the staged binaries wherever they actually landed.
STAGE_DIR = os.path.dirname(os.path.abspath(__file__))


class ShError(Exception):
    """`sh()` raised because the command exited nonzero. See `sh()`'s own
    docstring for why a command that embeds `echo RC=$?` never reaches this
    at all -- that is ordinary shell semantics, not special-cased here."""


def sh(cmd: str) -> str:
    """Run `cmd` through a POSIX shell and return its combined stdout+stderr.

    Raises `ShError` if the shell's own exit status is nonzero. A caller that
    wants to inspect a COMMAND's failure explicitly (`--self-test --unmapped`
    is *expected* to make `hexlib_run` exit nonzero; that expected failure is
    the whole point of the test) appends `; echo RC=$?` to `cmd` itself: the
    shell's own exit status then becomes `echo`'s -- always 0 -- regardless of
    what the real command did, and the caller reads the real code back out of
    the text this function returns. Nothing here special-cases that string;
    it is a consequence of how `;`-joined shell commands report their exit
    status, not a convention this function has to know about.

    FAIL CLOSED: a command that exits nonzero WITHOUT that trick is a genuine,
    unexpected failure -- of the command, of `cd`, of a path that does not
    exist -- and must stop the test right there rather than let a later
    assertion run against output that was never produced for the reason the
    test assumed.
    """
    proc = subprocess.run(
        ["adb", "shell", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = proc.stdout or ""
    if proc.returncode != 0:
        raise ShError(
            f"`adb shell` exited {proc.returncode}, and the command did not "
            f"embed its own `echo RC=$?` to report that itself: {cmd!r}\n{out}"
        )
    return out


def push(src_name: str, dest_dir: str) -> str:
    """`adb push` a file staged beside this module into `dest_dir` on the
    device, and return the resulting device path.

    A PUSH, NOT A COPY. `cp` was what this used to be, back when the test was
    believed to run on the phone; on the QDC runner that copies an AArch64
    binary from one host directory to another, reports success, and defers
    the failure to an exec-format error nobody would connect back to it.
    """
    src = os.path.join(STAGE_DIR, src_name)
    if not os.path.isfile(src):
        raise ShError(
            f"{src_name} is not beside this module ({STAGE_DIR}); the artifact "
            f"did not stage what it claimed to. Contents: "
            f"{sorted(os.listdir(STAGE_DIR))}"
        )
    proc = subprocess.run(
        ["adb", "push", src, dest_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise ShError(
            f"adb push {src!r} -> {dest_dir!r} exited {proc.returncode}:\n"
            f"{proc.stdout or ''}"
        )
    return dest_dir.rstrip("/") + "/" + src_name


def write_qdc_log(name: str, text: str) -> str:
    """Push `text` to `{QDC_LOG_DIR}/{name}` ON THE DEVICE and return that
    device path.

    A PUSH, not a local write. QDC collects that directory FROM THE PHONE;
    writing it on the runner (which is what this did) produces a report that
    is real, correct, and never collected -- see the module docstring for the
    two jobs that cost.

    Always writes, even if `text` is empty -- an empty or missing log must
    never be silently indistinguishable from "nothing worth logging"; that
    exact confusion is what let a device-farm job on this account once
    complete having run zero tests and report passing. This function's job is
    only to guarantee the write happens at a real, returned path; it does not
    itself judge whether `text` is meaningful -- the caller's own assertions,
    run BEFORE this is called, are what a reader should trust for that.
    """
    # POSIX join, never os.path.join: this path is on the DEVICE, and a
    # Windows runner would otherwise build `TestLogs\results.xml`.
    device_path = QDC_LOG_DIR + "/" + name.replace("\\", "/").lstrip("/")

    # mkdir -p THE PARENT, on the device. `name` legitimately carries a
    # subdirectory -- conftest.py writes `TestLogs/results.xml`, because
    # job.wait() matches that suffix and QDC lists collected logs as
    # `<job_id>/<name>`, so a flat name would never be recognised as the
    # report at all.
    parent = device_path.rsplit("/", 1)[0]
    subprocess.run(
        ["adb", "shell", f"mkdir -p {parent}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False, encoding="utf-8"
    ) as f:
        f.write(text)
        tmp_path = f.name
    try:
        proc = subprocess.run(
            ["adb", "push", tmp_path, device_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if proc.returncode != 0:
            raise ShError(
                f"adb push of the log to {device_path!r} exited "
                f"{proc.returncode}:\n{proc.stdout or ''}"
            )
    finally:
        os.unlink(tmp_path)
    return device_path

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

WHY `sh()` DOES NOT WRAP THE COMMAND IN `adb shell`. Every command
`test_on_device.py` passes here already names on-device paths directly
(`/data/local/tmp/hexlib/...`) with no `adb shell` prefix anywhere, which only
makes sense if this file's own process already has a working directory and a
shell on the device itself -- consistent with `artifact.py`'s
`TestFramework.APPIUM` packaging shipping a `requirements.txt` that `pip
install`s `pytest`, i.e. QDC's own runner provisions a Python (and therefore a
shell) ON the device and runs the whole TestPackage there. "On-farm scripts
have plain `adb`" (this project's own measured fact) describes what the FARM's
*other* scripts use, not what has to happen inside a test the farm executes
in an environment that already has device-local shell access. If that
assumption is ever wrong, the fix belongs in `sh()`, in one place.
"""
from __future__ import annotations

import os
import subprocess

QDC_LOG_DIR = "/data/local/tmp/QDC_logs"


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
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = proc.stdout or ""
    if proc.returncode != 0:
        raise ShError(
            f"command exited {proc.returncode}, and did not embed its own "
            f"`echo RC=$?` to report that itself: {cmd!r}\n{out}"
        )
    return out


def write_qdc_log(name: str, text: str) -> str:
    """Write `text` to `{QDC_LOG_DIR}/{name}` (creating the directory if
    needed) and return the path written.

    Always writes, even if `text` is empty -- an empty or missing log must
    never be silently indistinguishable from "nothing worth logging"; that
    exact confusion is what let a device-farm job on this account once
    complete having run zero tests and report passing. This function's job is
    only to guarantee the write happens at a real, returned path; it does not
    itself judge whether `text` is meaningful -- the caller's own assertions,
    run BEFORE this is called, are what a reader should trust for that.
    """
    os.makedirs(QDC_LOG_DIR, exist_ok=True)
    path = os.path.join(QDC_LOG_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

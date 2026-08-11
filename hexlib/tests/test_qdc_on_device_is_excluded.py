# hexlib/tests/test_qdc_on_device_is_excluded.py
"""`hexlib/device/qdc/test_on_device.py` runs ON THE PHONE, under the farm's
own pytest -- never here. This file proves the mechanism that keeps it out
of `hexlib`'s own suite actually works, by invoking pytest exactly the way
this project's own offline suite is run (`python -m pytest hexlib/tests -q`)
and reading the real collected node ids back, rather than merely asserting
that the on-device file's path string looks separate from `hexlib/tests/`
(which would pass even if pytest's own collection rules changed underneath
it, or if a future `conftest.py` widened `rootdir`/`testpaths` to sweep it
back in).
"""
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ON_DEVICE_TEST = os.path.join("hexlib", "device", "qdc", "test_on_device.py")


def test_the_on_device_file_actually_exists():
    """A prerequisite, not the point of this file: if this ever goes
    missing, every other assertion here about it being "excluded" would be
    vacuously true for the wrong reason."""
    assert os.path.isfile(os.path.join(REPO_ROOT, ON_DEVICE_TEST))


def test_pytest_hexlib_tests_does_not_collect_the_on_device_test():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "hexlib/tests"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert "test_on_device.py" not in result.stdout, (
        "hexlib/device/qdc/test_on_device.py was collected by "
        f"`pytest hexlib/tests` -- it must run only on the phone:\n{result.stdout}"
    )


def test_pytest_hexlib_tests_does_not_walk_into_device_qdc_at_all():
    """A second, independent way of asking the same question: even the
    DIRECTORY must never be walked, not merely this one file's node id --
    catches a future file added next to test_on_device.py that this test's
    sibling above would not, by name, think to look for."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "hexlib/tests"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert "device" + os.sep + "qdc" not in result.stdout
    assert "device/qdc" not in result.stdout

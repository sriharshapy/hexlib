# hexlib/tests/test_qdc_on_device_is_excluded.py
"""`hexlib/device/qdc/test_on_device.py` runs ON THE PHONE, under the farm's
own pytest -- never here. This file proves that the mechanism keeping it out of
hexlib's own suite works, by running pytest in a subprocess and reading the
real collected node ids back.

WHAT IS ACTUALLY GUARANTEED, AND WHAT THE PREVIOUS VERSION OF THIS DOCSTRING
CLAIMED. The previous version claimed these tests would survive "a future
`conftest.py` widened `rootdir`/`testpaths` to sweep it back in". They would
not have, for two reasons, and both were real defects rather than hypotheses:

  1. THEY INSPECTED A DIFFERENT COMMAND THAN CI RUNS. They invoked pytest with
     an explicit `hexlib/tests` path. `.github/workflows/ci.yml` runs
     `pytest -q -m "not sdk"` with NO PATH, so collection starts at the repo
     root and walks into `hexlib/device`, where `test_on_device.py`'s flat
     `from utils import sh, write_qdc_log` (correct on the farm, where the
     artifact is extracted unpackaged) raises ModuleNotFoundError at
     collection time. Reproduced: `Interrupted: 1 error during collection`,
     exit 2, ZERO tests run. Naming `hexlib/tests` on the command line hid
     exactly the failure this file exists to prevent.
  2. EVERY ASSERTION WAS `assert "..." not in result.stdout`, AND
     `result.returncode` WAS NEVER READ. Absence read as success: run the same
     command with a deliberately broken `-p` plugin and you get rc=1,
     len(stdout)==0, and all three assertions True. Any collection error --
     including the one in (1), had the command been the bare one -- turned this
     file green. Verified by mutation, both before and after this rewrite.

So what is guaranteed now is narrower and checkable: the EXACT command CI runs
exits 0, collects this very test file's own first test, and collects nothing
from `hexlib/device`. Plus the same for a `pytest hexlib` path invocation,
which is what pins the choice of mechanism (see the root conftest.py: a
`collect_ignore` there covers every invocation form, where a `testpaths` entry
in pyproject.toml would have covered only the bare one -- mutation-verified:
making that swap fails `test_naming_a_path_does_not_reach_the_on_device_test_
either` below and nothing else).

The self-referential node id is deliberate. Asserting on some OTHER file's test
name couples this file to a name it does not own; asserting that the
subprocess collected THIS file's own first test cannot drift, and is impossible
to satisfy with empty stdout or a collection error.
"""
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ON_DEVICE_TEST = os.path.join("hexlib", "device", "qdc", "test_on_device.py")

# The exact arguments .github/workflows/ci.yml's `offline` job passes, minus
# --collect-only. NO PATH ARGUMENT: that is the whole point (see 1. above).
CI_PYTEST_ARGS = ["-q", "-m", "not sdk"]

# This file's own module path, as pytest prints it: node ids are
# rootdir-relative with forward slashes on every platform, Windows included.
# Built from __file__ rather than typed out, so a rename cannot leave a stale
# literal behind that still "passes".
_THIS_MODULE_NODE = "hexlib/tests/" + os.path.basename(__file__)


def test_the_on_device_file_actually_exists():
    """A prerequisite, not the point of this file: if this ever goes
    missing, every other assertion here about it being "excluded" would be
    vacuously true for the wrong reason."""
    assert os.path.isfile(os.path.join(REPO_ROOT, ON_DEVICE_TEST))


# Derived from the function object above, never retyped -- renaming that test
# moves this with it instead of leaving a node id that no longer exists (which
# would fail loudly, but for the wrong reason).
_KNOWN_GOOD_NODE_ID = (
    f"{_THIS_MODULE_NODE}::{test_the_on_device_file_actually_exists.__name__}"
)


def _collect(*args):
    """Run `pytest --collect-only ARGS` at the repo root and return the
    CompletedProcess. `--collect-only` is the only difference from the real
    command: it makes the node ids readable without running 600+ tests inside
    a test."""
    return subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", *args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


@pytest.fixture(scope="module")
def ci_collection():
    """Collection under the command CI actually runs: bare, no path."""
    return _collect(*CI_PYTEST_ARGS)


@pytest.fixture(scope="module")
def hexlib_path_collection():
    """Collection when a path IS given (`pytest hexlib`). `testpaths` is
    ignored in this case; only a `collect_ignore` covers it."""
    return _collect("-q", "hexlib")


def _assert_collection_succeeded(result, how):
    """The two things the old version of this file never checked. Order
    matters: report the rc first, because a collection error is what makes
    every "not in stdout" assertion vacuously true."""
    assert result.returncode == 0, (
        f"`pytest --collect-only {how}` exited {result.returncode}, not 0 -- "
        "collection itself failed, so nothing below this line would have "
        "proved anything about what was or was not collected:\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert _KNOWN_GOOD_NODE_ID in result.stdout, (
        f"`pytest --collect-only {how}` exited 0 but did not collect "
        f"{_KNOWN_GOOD_NODE_ID} -- this file's own first test. Empty or "
        "unrecognizable output must never be read as 'the on-device test was "
        f"excluded':\n--- stdout ---\n{result.stdout}"
    )


def _assert_device_qdc_absent(result, how):
    assert "test_on_device.py" not in result.stdout, (
        f"hexlib/device/qdc/test_on_device.py was collected by `pytest {how}` "
        f"-- it must run only on the phone:\n{result.stdout}"
    )
    # The DIRECTORY, not merely this one file's node id: catches a second
    # on-device file added next to test_on_device.py that the check above
    # would not, by name, think to look for. Unlike the version of this
    # assertion that named `hexlib/tests` on the command line -- where a node
    # id could never have contained `device/qdc` in the first place, so it had
    # no discriminating power at all -- both invocations here start at or
    # above `hexlib`, so `hexlib/device/qdc/...` node ids are exactly what
    # WOULD appear if the exclusion were removed.
    assert "device/qdc" not in result.stdout
    assert "device" + os.sep + "qdc" not in result.stdout


def test_the_bare_command_ci_runs_collects_cleanly(ci_collection):
    """THE LOAD-BEARING ONE. `pytest -q -m "not sdk"` -- CI's own command, no
    path -- must exit 0 and collect real tests. This is exactly what was
    broken: it exited 2 with zero tests run, and no test in this repo could
    see it."""
    _assert_collection_succeeded(ci_collection, " ".join(CI_PYTEST_ARGS))


def test_the_bare_command_ci_runs_does_not_collect_the_on_device_test(ci_collection):
    """Binds to the mechanism: empty `collect_ignore` in the root conftest.py
    and this fails -- on the rc, since the flat `import utils` breaks
    collection outright, and on the node-id checks here if that import were
    ever made to work. Mutation-verified both ways round."""
    _assert_collection_succeeded(ci_collection, " ".join(CI_PYTEST_ARGS))
    _assert_device_qdc_absent(ci_collection, " ".join(CI_PYTEST_ARGS))


def test_naming_a_path_does_not_reach_the_on_device_test_either(
    hexlib_path_collection,
):
    """`pytest hexlib` -- a path argument, so `testpaths` would NOT apply.
    This is the case that makes the root conftest.py's `collect_ignore` the
    right mechanism rather than a `testpaths` entry; if someone swaps one for
    the other, this test is the only thing that notices."""
    _assert_collection_succeeded(hexlib_path_collection, "hexlib")
    _assert_device_qdc_absent(hexlib_path_collection, "hexlib")

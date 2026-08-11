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

THIRD DEFECT, FIXED 2026-08-11: THE ABSENCE CHECK MATCHED BARE SUBSTRINGS
AGAINST THE WHOLE OF STDOUT. It was `assert "test_on_device.py" not in
result.stdout` plus `assert "device/qdc" not in result.stdout`. Collection
output is a list of NODE IDS, but those assertions searched every byte of it,
so any new parametrize id, test name, or (in the failure path) traceback text
that merely NAMED that file or that directory tripped them -- and the failure
message then claimed the on-device file WAS collected when it was not, which is
a confusing thing to debug under merge pressure. It was hit for real. The
checks below parse the node ids out of stdout and match a node-id PREFIX
instead, and `test_the_absence_check_does_not_fire_on_a_mere_mention` /
`test_the_absence_check_still_fires_on_a_real_device_node_id` pin both
directions of that against fabricated output, so neither half is taken on
trust.

THE EXCLUSION MECHANISM ITSELF IS UNCHANGED -- still the root conftest.py's
`collect_ignore`. Only how this file VERIFIES it changed.
"""
import os
import re
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


# A collected node id, as `--collect-only -q` prints one per line:
# `hexlib/tests/test_x.py::test_y`, or `...::test_y[param]` when parametrized.
# Anchored, with no whitespace before the `::`, so a line of prose that happens
# to contain both a `.py` and a `::` (a traceback, a message quoting a node id
# mid-sentence) is not mistaken for a collected test.
_NODE_ID_LINE = re.compile(r"\A(?P<file>\S+\.py)::(?P<rest>\S+)\Z")

# Node ids from the on-device tree, which is what WOULD appear if the root
# conftest.py's `collect_ignore` stopped working. The whole DIRECTORY, not just
# test_on_device.py's own name: a second on-device file added next to it must be
# caught without anyone having to remember to come back here.
_ON_DEVICE_NODE_PREFIX = "hexlib/device/"


def _collected_node_ids(stdout):
    """Every collected node id in `--collect-only` output, separators
    normalized to forward slashes.

    PARSED, NOT SUBSTRING-SEARCHED. This is the whole fix for the third defect
    in this file's docstring: the previous version searched raw stdout, so any
    line that merely mentioned a filename or a directory counted as evidence
    that it had been collected.
    """
    ids = []
    for raw in stdout.splitlines():
        m = _NODE_ID_LINE.match(raw.strip().replace("\\", "/"))
        if m:
            ids.append(m.group(0))
    return ids


def _on_device_node_ids(stdout):
    return [
        n for n in _collected_node_ids(stdout)
        if n.startswith(_ON_DEVICE_NODE_PREFIX)
    ]


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
    collected = _collected_node_ids(result.stdout)
    assert _KNOWN_GOOD_NODE_ID in collected, (
        f"`pytest --collect-only {how}` exited 0 but did not collect "
        f"{_KNOWN_GOOD_NODE_ID} -- this file's own first test. Empty or "
        "unrecognizable output must never be read as 'the on-device test was "
        f"excluded'. Parsed {len(collected)} node id(s) from:\n"
        f"--- stdout ---\n{result.stdout}"
    )


def _assert_device_qdc_absent(result, how):
    offenders = _on_device_node_ids(result.stdout)
    assert not offenders, (
        f"`pytest {how}` collected node id(s) under {_ON_DEVICE_NODE_PREFIX} "
        f"-- those tests run only on the phone: {offenders}"
    )


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


# ==============================================================================
# BOTH DIRECTIONS OF THE ABSENCE CHECK, against fabricated output.
#
# The previous version of that check (`assert "test_on_device.py" not in
# result.stdout`) was wrong in the FALSE-POSITIVE direction: any test name,
# parametrize id or traceback line that merely NAMED the file failed it, with a
# message claiming the file had been collected when it had not. It was hit for
# real. Fixing that without also pinning the true-positive direction would just
# trade one silent failure for another, so both are checked here -- and neither
# needs a subprocess, so they cannot be skipped for being slow.
# ==============================================================================


def test_the_absence_check_does_not_fire_on_a_mere_mention():
    """A collected test whose NAME contains the on-device filename, plus prose
    quoting the full path -- neither is a collected on-device node id."""
    stdout = (
        f"{_KNOWN_GOOD_NODE_ID}\n"
        "hexlib/tests/test_cli_device_flag.py::test_error_names_test_on_device_py\n"
        "hexlib/tests/test_x.py::test_paths[hexlib/device/qdc/test_on_device.py]\n"
        "  the staged script hexlib/device/qdc/test_on_device.py runs on the phone\n"
        "3 tests collected in 0.42s\n"
    )
    assert _on_device_node_ids(stdout) == [], (
        "a mention is not a collection -- this is the false positive that made "
        "the previous check claim the on-device file had been collected when it "
        "had not"
    )
    assert _KNOWN_GOOD_NODE_ID in _collected_node_ids(stdout)


def test_the_absence_check_still_fires_on_a_real_device_node_id():
    """The direction that matters: an actually-collected on-device test must be
    reported. Without this, the fix above could have been "match nothing"."""
    stdout = (
        f"{_KNOWN_GOOD_NODE_ID}\n"
        "hexlib/device/qdc/test_on_device.py::test_binaries_are_present\n"
        "2 tests collected in 0.42s\n"
    )
    offenders = _on_device_node_ids(stdout)
    assert offenders == [
        "hexlib/device/qdc/test_on_device.py::test_binaries_are_present"
    ]


def test_the_absence_check_catches_a_second_on_device_file_and_windows_paths():
    """The DIRECTORY, not one filename: a new on-device file next to
    test_on_device.py is caught without anyone editing this test. Backslash
    node ids are normalized rather than needing their own assertion."""
    stdout = (
        f"{_KNOWN_GOOD_NODE_ID}\n"
        "hexlib/device/qdc/test_something_new.py::test_z\n"
        "hexlib\\device\\qdc\\test_on_device.py::test_w\n"
    )
    assert len(_on_device_node_ids(stdout)) == 2


def test_a_collection_error_cannot_look_like_a_clean_exclusion():
    """Empty or error output yields ZERO node ids, so `_assert_collection_
    succeeded`'s known-good-id assertion fails -- absence is never read as
    success here."""
    assert _collected_node_ids("") == []
    assert _collected_node_ids("Interrupted: 1 error during collection\n") == []

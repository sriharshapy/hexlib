# conftest.py -- repo root
"""KEEP THE ON-DEVICE TEST OUT OF THE OFFLINE SUITE AS A MECHANISM, NOT A
CONVENTION.

`hexlib/device/qdc/test_on_device.py` runs ON THE PHONE, inside the QDC
artifact zip, under the farm's own pytest -- never here. Beside it in that zip
sits a flat `utils.py` which it imports as a TOP-LEVEL module (`from utils
import sh, write_qdc_log`), because on the farm the artifact is extracted into
one flat directory with no package around it. In this repo the same file lives
inside the `hexlib.device.qdc` package, so pytest imports it as
`hexlib.device.qdc.test_on_device`, `utils` is not a top-level module, and the
import raises ModuleNotFoundError AT COLLECTION TIME -- which pytest reports as
`Interrupted: 1 error during collection`, exit code 2, and ZERO tests run.

That is not a hypothetical. `.github/workflows/ci.yml` runs `pytest -q -m "not
sdk"` with NO PATH, so collection starts at the repo root and walks into
`hexlib/device`. Reproduced on this machine before this file existed:

    python -m pytest -q -m "not sdk"
    -> ModuleNotFoundError: No module named 'utils'
    -> Interrupted: 1 error during collection    (exit 2, zero tests run)

It was latent only because this repo has no git remote yet, so CI has never
actually run.

WHY `collect_ignore` HERE AND NOT `testpaths` IN pyproject.toml. `testpaths`
applies ONLY when no path is given on the command line. That does cover CI's
bare invocation, which is the case that was broken -- but it covers nothing
else: `pytest .`, `pytest hexlib`, or `pytest hexlib/device` would each walk
back into the on-device file and break collection again, and each is a
plausible thing for a human or a future workflow step to type. `collect_ignore`
in the ROOT conftest.py is consulted during directory collection no matter how
collection was started, so it covers the bare invocation AND every path form
above with one mechanism. `hexlib/tests/test_qdc_on_device_is_excluded.py`
pins both properties by actually running pytest, so this reasoning is checked
rather than merely asserted here.

WHAT THIS DELIBERATELY DOES NOT DO. It does not make the on-device file
importable, and it must not: the file is correct as written -- flat `import
utils` is what works on the farm. It also does not stop
`pytest hexlib/device/qdc/test_on_device.py` if someone names the file
directly; that is an explicit request, and it will fail loudly on the import
rather than silently pass, which is the right outcome.
"""

# Paths are relative to this file's directory (the repo root). The whole
# directory, not just the one file: a second on-device test added next to
# test_on_device.py must be excluded for the same reason, without anyone
# having to remember to come back here.
collect_ignore = ["hexlib/device"]

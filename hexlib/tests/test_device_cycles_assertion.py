# hexlib/tests/test_device_cycles_assertion.py
"""The one part of `hexlib/device/qdc/test_on_device.py` that CAN be run here.

DELIBERATELY NOT NAMED `test_on_device_...`. `test_qdc_on_device_is_excluded.py`
proves the on-device file is never collected by asserting the literal string
`test_on_device.py` does not appear in `pytest --collect-only` output; a file
here whose own name shares that prefix is one rename away from making that
proof fail on a correctly-excluded repo.

THE ON-DEVICE FILE ITSELF CANNOT BE. It executes inside the QDC artifact,
against a real phone, and the root `conftest.py` deliberately keeps pytest from
collecting it at all (see `test_qdc_on_device_is_excluded.py` for the mechanism
and the test that proves it). Nothing in this file changes that, and nothing
here runs a device test: `sh()` and `write_qdc_log()` are replaced by stubs and
no `test_*` function from that module is ever called.

WHAT IS RUN, AND WHY IT IS WORTH RUNNING. `assert_cycles_total_is_a_real_
measurement` is a pure function of one string -- the captured stdout of a
`hexlib_run` invocation -- so its behaviour is fully determined offline. It is
also the load-bearing new device assertion: until 2026-08-11 NOTHING on device
asserted the `cycles_total=` line at all (the test's docstring said `main.c`
never printed it, which had been false since an earlier commit), so deleting
`main.c`'s two `printf` lines would have kept all five on-device tests green
while flipping `hexlib/cli.py`'s post-job check to exit 1.

A SOURCE ASSERTION WOULD NOT HAVE BEEN ENOUGH, for the same reason
`test_coherency_lane_classification.py` exists: "the file contains the string
`cycles_total`" cannot tell a check that rejects `cycles_total=0` apart from one
that accepts it, and accepting zero is precisely the defect -- the DSP reads
PCYCLE in a user-mode unsigned PD, where `SYSCFG.PCYCLEEN` cannot be set, so
zero is the expected reading if the counter is dead on this silicon. So the
function is imported and CALLED against real strings, including `cycles_total=0`.

The rest of the on-device file's assertions genuinely cannot be executed here
and are reviewed only; that is stated plainly rather than implied by this
file's existence.
"""
import ast
import importlib.util
import inspect
import pathlib
import sys
import textwrap
import types

import pytest

ON_DEVICE = pathlib.Path("hexlib/device/qdc/test_on_device.py")


@pytest.fixture(scope="module")
def on_device():
    """Load the on-device module by path, with a stub `utils` standing in for
    the flat module that sits beside it in the artifact zip.

    NOT imported as `hexlib.device.qdc.test_on_device`: on the farm the
    artifact is one flat directory and `from utils import sh, write_qdc_log`
    is what works there, so the file is correct as written and must not be
    changed to import differently (conftest.py's own docstring says so). The
    stubs raise if called, so a future refactor that made module import time
    shell out would fail here rather than silently run something."""

    def _no(*a, **kw):
        raise AssertionError(
            "the on-device module must not run shell commands at import time"
        )

    stub = types.ModuleType("utils")
    stub.sh = _no
    stub.write_qdc_log = _no
    # `push` joined these when the on-device model was corrected: pytest runs
    # on the QDC RUNNER, so binaries reach the phone by `adb push`, not `cp`.
    # Stubbed like the others so an import-time call fails here rather than
    # shelling out.
    stub.push = _no

    saved = sys.modules.get("utils")
    sys.modules["utils"] = stub
    try:
        spec = importlib.util.spec_from_file_location(
            "hexlib_on_device_under_test", ON_DEVICE
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if saved is None:
            del sys.modules["utils"]
        else:
            sys.modules["utils"] = saved
    return mod


def test_a_real_measurement_is_accepted_and_returned(on_device):
    out = (
        "hexlib: --self-test: PASS (4100 values, bit-exact)\n"
        "hexlib: --self-test: cycles_total=1287\n"
        "RC=0\n"
    )
    assert on_device.assert_cycles_total_is_a_real_measurement(out, "x") == 1287


def test_a_zero_measurement_is_rejected(on_device):
    """THE DEFECT. `cycles_total=0` contains the substring `cycles_total=`, so
    any presence-only check accepts it -- and a run in which the DSP's cycle
    counter never advanced measured nothing. The failure message must name
    PCYCLEEN, because that is the actionable finding: not "hexlib is broken"
    but "the counter does not work in this PD, so no stage-1 cycle figure
    transfers"."""
    out = (
        "hexlib: --self-test: PASS (4100 values, bit-exact)\n"
        "hexlib: --self-test: cycles_total=0\n"
        "RC=0\n"
    )
    with pytest.raises(AssertionError) as e:
        on_device.assert_cycles_total_is_a_real_measurement(out, "x")
    assert "PCYCLEEN" in str(e.value), (
        "a zero cycle count must be reported as the PCYCLEEN/unsigned-PD "
        "finding it is, not as a generic assertion failure"
    )


def test_an_absent_line_is_rejected_with_a_different_message(on_device):
    """Absent and zero are DIFFERENT findings -- one means hexlib stopped
    printing the line (or got no response), the other means the hardware
    counter is dead -- and must not be reported as each other."""
    out = "hexlib: --self-test: PASS (4100 values, bit-exact)\nRC=0\n"
    with pytest.raises(AssertionError) as e:
        on_device.assert_cycles_total_is_a_real_measurement(out, "x")
    msg = str(e.value)
    assert "no `cycles_total=` line" in msg
    assert "PCYCLEEN" not in msg


def test_a_positive_line_alongside_a_zero_one_is_accepted(on_device):
    """One captured run can legitimately carry more than one `cycles_total=`
    line. At least one genuine measurement is the bar; if the counter were
    dead, EVERY line would read 0 and the zero test above still catches it."""
    out = "cycles_total=0\ncycles_total=1287\n"
    assert on_device.assert_cycles_total_is_a_real_measurement(out, "x") == 1287


def test_the_regex_does_not_match_a_non_numeric_value(on_device):
    """`cycles_total=<garbled>` must not be silently read as a measurement.
    The regex captures digits only, so a malformed value looks ABSENT to this
    helper -- which fails, which is the safe direction. (hexlib/cli.py
    distinguishes malformed from absent on its side, where it has the whole
    job's logs and can say which.)"""
    out = "hexlib: --self-test: cycles_total=<garbled>\n"
    with pytest.raises(AssertionError):
        on_device.assert_cycles_total_is_a_real_measurement(out, "x")


_HELPER = "assert_cycles_total_is_a_real_measurement"


def _calls_in(fn):
    """The set of function names CALLED in `fn`'s body, read out of its parsed
    syntax tree.

    WHY AN AST AND NOT A SUBSTRING SEARCH. This test used to be
    `_HELPER + "(" in src`, over `inspect.getsource` text with only the
    DOCSTRING removed. That is comment-blind: deleting the real call and leaving
    `# assert_cycles_total_is_a_real_measurement(out, tag)` behind kept this
    file at 6 passed while nothing on device checked the measurement at all --
    the same defect class the C source-assertion tests were rewritten twice for,
    reappearing in the one place the subject is Python.

    WHY NOT `csource`, WHICH IS WHERE THE C SIDE'S ANSWER LIVES. It is a C
    lexer: its comment tokens are `/* */` and `//`, and Python's is `#`. Handing
    it Python source blanks the string literals and leaves every `#` comment
    exactly where it was -- so it would not close this hole, only appear to. An
    AST closes it by construction instead: a commented-out call is not a Call
    node, a call named inside a string is not a Call node, and a docstring is
    not a Call node, so none of the three need special handling. It is also
    stricter than the text check ever was, because `_HELPER` appearing in an
    unrelated expression (an f-string, a variable name) no longer counts.

    Attribute calls (`utils.foo()`) contribute their attribute name, so a call
    reached through a module or object alias is still seen."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                found.add(func.id)
            elif isinstance(func, ast.Attribute):
                found.add(func.attr)
    return found


def test_both_device_invocations_assert_the_measurement(on_device):
    """Both `--self-test` and `--self-test --coherency-check` must call the
    helper -- reviewed by source here, since the calls themselves can only run
    on a device. Scoped to each function's own syntax tree, so one call cannot
    cover for the other's absence, and a commented-out call cannot cover for
    either (see `_calls_in`)."""
    for name in (
        "test_scale_fp16_runs_on_the_dsp_and_is_correct",
        "test_cache_coherency_is_independent_of_marshalling_and_of_any_kernel",
    ):
        fn = getattr(on_device, name)
        assert _HELPER in _calls_in(fn), (
            f"{name} does not CALL {_HELPER}() -- nothing on device would then "
            f"check the measurement at all, and a mention of it in a comment or "
            f"a docstring is not a call"
        )

# hexlib/tests/test_coherency_lane_classification.py
"""BEHAVIOURAL test for the -0.0 / sentinel classification in main.c's
--coherency-check (hexlib_classify_coherency_lane()).

WHY A SOURCE ASSERTION IS NOT ENOUGH -- THE SAME DEFECT CLASS AS THE ARCH
DECODE. test_host_source.py can only check that a sign-bit mask (`& 0x7FFF`)
and a bit-exact sentinel compare (`bits == sentinel_bits`) EXIST in
hexlib_classify_coherency_lane() -- it cannot see whether they actually
classify -0.0 (bit pattern 0x8000) as "the kernel wrote zero", which is
exactly the case that produced this project's real false coherency-miss
report: `x * 0.0f` is -0.0 for every negative lane of the self-test's own
input, and a bit-exact compare against `(__fp16) 0.0f` (0x0000) read that
HEALTHY result as "sentinel unchanged". A test that only checks "a magnitude
check and a sentinel check exist" cannot catch a magnitude check that is
subtly wrong for the one input that matters; a test that compiles and RUNS
the actual classification against 0x8000 can. This is the identical shape to
test_session_arch_decode.py's own reason for existing -- see that file's
module docstring -- so this one follows the same recipe.

WHAT THIS DOES. `hexlib_classify_coherency_lane()` -- and the
`enum hexlib_coherency_lane` it returns -- are extracted straight out of
main.c with `csource.function_body`/`csource.block_from`, the SAME
comment-aware slicers every other source-assertion test in this project
uses, dropped into a tiny standalone .c file next to a one-line harness that
exports it under a stable name, compiled with a host C compiler into a
shared library, and called through ctypes with the bit patterns that matter:
0x0000 (+0.0), 0x8000 (-0.0, the exact value that broke this check for
real), an arbitrary sentinel pattern, and a value that is neither.

OVER RAW uint16_t BITS, NEVER __fp16 -- BOTH BECAUSE THAT IS WHAT THE REAL
FUNCTION NOW TAKES, AND BECAUSE THE HOST COMPILER HERE HAS NO __fp16 AT ALL.
`__fp16` is an ARM/AArch64 storage type; on this project's Windows dev
machine, the only host C compiler on PATH is a plain x86_64 mingw gcc, which
rejects `__fp16` outright (confirmed: `unknown type name '__fp16'`) even
though the real device build (NDK clang, aarch64) accepts it without issue.
Rather than skip this gap because of that mismatch, main.c's
hexlib_classify_coherency_lane() was written to operate on raw uint16_t bit
patterns in the first place -- see its own header comment in main.c for why
that is bit-for-bit equivalent to the fabsf()-based check it replaced, not a
weaker stand-in for it. That is what makes this test possible on any host
compiler at all, never a reason to water down what it checks.

THE DECISIVE PROPERTY. If hexlib_classify_coherency_lane() is ever removed
or renamed, `_function_body` raises before this test ever gets to compile or
call anything. If it is present but its zero check regresses to a bit-exact
compare against +0.0 alone (the original bug, reintroduced), the compiled
call on 0x8000 returns SENTINEL or OTHER, never ZERO --
`test_negative_zero_bits_classify_as_the_expected_zero_result` below fails.
`test_mutation_verify_the_original_bit_exact_compare_misclassifies_negative_zero`
goes one step further and proves this directly: it takes the SAME extracted
source, mechanically reverts the mask to the original `bits == 0x0000`
compare, compiles THAT, and confirms 0x8000 is misclassified under it --
concrete, run evidence that this test suite would have caught the original
defect, not just an assertion that it currently doesn't reproduce it.

THE COMPILER-INDEPENDENT GUARD BELOW WAS ONCE FOOLABLE BY A COMMENT. Same
history as test_session_arch_decode.py's -- see that file's docstring.
`csource.function_body` used to return the raw, comment-BEARING body, so
reverting hexlib_classify_coherency_lane() to `bits == 0x0000u` and dropping
HEXLIB_LANE_OTHER, with the old code left in a comment inside the body, passed
`test_the_classification_function_the_behavioural_test_depends_on_still_exists`
-- the one test in this file that runs when there is no host `cc`, and
therefore the only guard at all on such a machine. `function_body` now returns
comment-blanked text; that exact revert now fails it. Verified by mutation.
"""
import ctypes
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

from hexlib.tests.csource import block_from as _block_from
from hexlib.tests.csource import function_body as _function_body

MAIN_C = pathlib.Path("hexlib/runtime/host/main.c")

HOST_CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
needs_cc = pytest.mark.skipif(
    HOST_CC is None,
    reason=(
        "no host C compiler found (tried: gcc, cc, clang); the BEHAVIOURAL "
        "-0.0/sentinel coherency-classification test is skipped and only "
        "the weaker source assertion in test_host_source.py "
        "(test_coherency_check_treats_negative_zero_as_the_expected_zero_"
        "result / test_coherency_check_verifies_the_surviving_bytes_are_"
        "really_the_sentinel) covers this. Install a host C compiler to "
        "restore it."
    ),
)


@pytest.fixture(scope="module")
def main_source():
    return MAIN_C.read_text()


def test_the_classification_function_the_behavioural_test_depends_on_still_exists(
    main_source,
):
    """COMPILER-INDEPENDENT -- runs even when needs_cc above would skip
    everything else, so a rename or removal of the function this file's
    behavioural tests extract is never invisible on a machine with no host C
    compiler. Pairs the exact NAME the fixtures below extract
    (hexlib_classify_coherency_lane, enum hexlib_coherency_lane) with what
    main.c actually contains, so the behavioural test and the source it
    depends on cannot silently drift apart -- see this module's own
    docstring and test_session_arch_decode.py's identical guard for the arch
    decode."""
    body = _function_body(main_source, "hexlib_classify_coherency_lane")
    assert "HEXLIB_LANE_ZERO" in body
    assert "HEXLIB_LANE_SENTINEL" in body
    assert "HEXLIB_LANE_OTHER" in body
    enum_start = main_source.index("enum hexlib_coherency_lane {")
    enum_block = _block_from(main_source, enum_start)
    assert "HEXLIB_LANE_ZERO" in enum_block
    assert "HEXLIB_LANE_SENTINEL" in enum_block
    assert "HEXLIB_LANE_OTHER" in enum_block


@pytest.fixture(scope="module")
def classify_fn_source(main_source):
    """The real hexlib_classify_coherency_lane() BODY, sliced straight out
    of main.c and never retyped here -- see test_session_arch_decode.py's
    identical fixture for the same rationale."""
    return _function_body(main_source, "hexlib_classify_coherency_lane")


@pytest.fixture(scope="module")
def enum_def(main_source):
    """The real `enum hexlib_coherency_lane { ... }` definition, sliced out
    with `csource.block_from` (comment-aware, same as `function_body`) so the
    standalone harness below returns the SAME symbolic values main.c does,
    never hand-retyped ones that could silently drift from a renumbering."""
    start = main_source.index("enum hexlib_coherency_lane {")
    block = _block_from(main_source, start)
    return "enum hexlib_coherency_lane " + block + ";"


def _enum_value(enum_source, name):
    m = re.search(rf"\b{re.escape(name)}\s*=\s*(\d+)", enum_source)
    assert m, f"could not find {name} in the extracted enum definition"
    return int(m.group(1))


def _compile_harness(tmp_path, name, enum_source, fn_body):
    """Wrap `fn_body` (the extracted or mutated function body) in the real
    enum definition and a stable-named uint16_t-in/int-out harness, compile
    it into a shared library with the host C compiler, and return a ctypes
    callable. Shared by both the real-function test below and the
    mutation-verification test -- so both go through the exact same
    compile-and-call path, and only the function body under test differs."""
    c_path = tmp_path / f"{name}.c"
    c_path.write_text(
        "#include <stdint.h>\n"
        f"{enum_source}\n"
        "static enum hexlib_coherency_lane\n"
        "hexlib_classify_coherency_lane(uint16_t bits, uint16_t sentinel_bits)\n"
        f"{fn_body}\n"
        "#if defined(_WIN32)\n"
        "__declspec(dllexport)\n"
        "#endif\n"
        "int harness_classify(uint16_t bits, uint16_t sentinel_bits) {\n"
        "    return (int) hexlib_classify_coherency_lane(bits, sentinel_bits);\n"
        "}\n"
    )
    lib_path = tmp_path / (f"{name}.dll" if sys.platform == "win32" else f"{name}.so")
    cmd = [HOST_CC, "-shared", "-fPIC", "-o", str(lib_path), str(c_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"compiling {name}.c (extracted hexlib_classify_coherency_lane) "
        f"failed:\n{result.stdout}\n{result.stderr}"
    )
    lib = ctypes.CDLL(str(lib_path))
    lib.harness_classify.restype = ctypes.c_int
    lib.harness_classify.argtypes = [ctypes.c_uint16, ctypes.c_uint16]
    return lib.harness_classify


@pytest.fixture
def compiled_classify(classify_fn_source, enum_def, tmp_path):
    return _compile_harness(tmp_path, "classify", enum_def, classify_fn_source)


@pytest.fixture(scope="module")
def lane_values(enum_def):
    """The real ZERO/SENTINEL/OTHER integer values, read out of the
    extracted enum text itself -- never hand-typed as 0/1/2, so a
    renumbering in main.c is reflected here automatically instead of
    silently comparing against stale constants."""
    return {
        "ZERO": _enum_value(enum_def, "HEXLIB_LANE_ZERO"),
        "SENTINEL": _enum_value(enum_def, "HEXLIB_LANE_SENTINEL"),
        "OTHER": _enum_value(enum_def, "HEXLIB_LANE_OTHER"),
    }


# The four bit patterns this test drives the classifier with. 0x3C00 is fp16
# 1.0 (sign 0, exponent 01111, mantissa 0) -- used as an arbitrary, clearly
# nonzero sentinel pattern, matching main.c's own COHERENCY_SENTINEL (1.0f).
# 0x4000 is fp16 2.0 -- nonzero, and not equal to the sentinel used here,
# i.e. neither classification.
_SENTINEL_BITS = 0x3C00
_POS_ZERO_BITS = 0x0000
_NEG_ZERO_BITS = 0x8000   # THE case: `x * 0.0f` for negative x.
_OTHER_BITS = 0x4000


@needs_cc
def test_negative_zero_bits_classify_as_the_expected_zero_result(
    compiled_classify, lane_values
):
    """THE DECISIVE CASE. -0.0 (0x8000, exactly what `x * 0.0f` produces for
    every negative lane of the self-test's own input) must classify as ZERO
    -- "the kernel wrote its result" -- not as SENTINEL ("the write never
    reached the host") and not as OTHER. This is the exact input that
    produced this project's real false coherency-miss report; see main.c's
    own header comment on run_coherency_check() and COHERENCY_FACTOR."""
    assert compiled_classify(_NEG_ZERO_BITS, _SENTINEL_BITS) == lane_values["ZERO"], (
        "-0.0 (0x8000) must classify as ZERO -- a bit-exact compare against "
        "+0.0 alone would misclassify this as SENTINEL and report a "
        "coherency miss that never happened"
    )


@needs_cc
def test_positive_zero_bits_classify_as_zero(compiled_classify, lane_values):
    assert compiled_classify(_POS_ZERO_BITS, _SENTINEL_BITS) == lane_values["ZERO"]


@needs_cc
def test_the_sentinel_pattern_classifies_as_not_written(compiled_classify, lane_values):
    """A lane that is bit-exact the sentinel that was written before invoke
    must classify as SENTINEL -- "the write never reached the host" (or the
    kernel never ran; cycles_total is what tells those two apart, not this
    function)."""
    assert (
        compiled_classify(_SENTINEL_BITS, _SENTINEL_BITS) == lane_values["SENTINEL"]
    )


@needs_cc
def test_a_value_that_is_neither_zero_nor_sentinel_is_the_third_outcome(
    compiled_classify, lane_values
):
    """A garbled or partially-written lane -- neither the expected zero
    result nor the intact sentinel -- must be its own, third outcome, never
    folded into a coherency-miss (SENTINEL) or a clean-pass (ZERO) claim."""
    result = compiled_classify(_OTHER_BITS, _SENTINEL_BITS)
    assert result == lane_values["OTHER"]
    assert result != lane_values["ZERO"]
    assert result != lane_values["SENTINEL"]


@needs_cc
def test_mutation_verify_the_original_bit_exact_compare_misclassifies_negative_zero(
    classify_fn_source, enum_def, lane_values, tmp_path
):
    """MUTATION-VERIFY. Takes the SAME extracted function body and
    mechanically reverts the sign-bit mask to the ORIGINAL, buggy bit-exact
    compare against +0.0 this project shipped once (`bits == 0x0000` in
    place of `(bits & 0x7FFFu) == 0`), compiles THAT, and confirms -0.0
    (0x8000) is misclassified under it -- concrete, run evidence that the
    tests above would have caught the original defect, not merely an
    assertion that they currently don't reproduce it."""
    target = "(uint16_t) (bits & 0x7FFFu) == 0"
    assert target in classify_fn_source, (
        "mutation target text not found in the extracted function body -- "
        "did the real sign-bit-mask check change shape? update this "
        "mutation to match, don't just delete it"
    )
    mutated = classify_fn_source.replace(target, "bits == 0x0000u")
    assert mutated != classify_fn_source

    buggy = _compile_harness(tmp_path, "classify_buggy", enum_def, mutated)
    result = buggy(_NEG_ZERO_BITS, _SENTINEL_BITS)
    assert result != lane_values["ZERO"], (
        "MUTATION CHECK FAILED TO REPRODUCE THE ORIGINAL BUG: reverting to "
        "a bit-exact compare against +0.0 should misclassify -0.0 as NOT "
        "zero (the exact false coherency-miss this project shipped once), "
        "but the mutated function still returned ZERO -- something about "
        "this mutation no longer matches the real defect shape"
    )
    # And confirm the FIXED function (compiled the ordinary way) does not
    # share that failure -- the mutation is a genuine regression relative to
    # the real, current source, not a mutation of something already broken.
    fixed = _compile_harness(tmp_path, "classify_fixed_for_mutation_check", enum_def, classify_fn_source)
    assert fixed(_NEG_ZERO_BITS, _SENTINEL_BITS) == lane_values["ZERO"]

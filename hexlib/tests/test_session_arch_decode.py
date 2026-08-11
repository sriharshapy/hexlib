# hexlib/tests/test_session_arch_decode.py
"""BEHAVIOURAL test for the driver ARCH_VER BCD decode in session.c.

WHY A SOURCE ASSERTION IS NOT ENOUGH. test_host_source.py can only check that
a comparison and a `return -1` exist in hexlib_open() -- it cannot see that
the two operands being compared are in incommensurable encodings. That is
exactly the shape of the original bug: `arch` (from hexlib_iface_hwinfo,
__HEXAGON_ARCH__, plain decimal, e.g. 75 -- see skel.c, test_dsp_sim.py's own
`info.arch == 75`) was compared directly against `caps.arch_ver` (from
DSPRPC_GET_DSP_INFO/ARCH_VER, a BCD-nibble-packed byte, e.g. 0x8c75 = 35957
-- see device/qdc/test_on_device.py's own `35957` / `0x8c75`). 75 != 35957
unconditionally, so every device session refused before measuring anything.
A test that only checks "a comparison and a failure path exist" cannot catch
that; a test that compiles and RUNS the actual decode can.

WHAT THIS DOES. `hexlib_decode_bcd_arch()` is extracted straight out of
session.c with `csource.function_body` -- the SAME comment-aware slicer
every other source-assertion test in this project uses, never a second
hand-rolled copy (see csource.py's own module docstring for why that
matters) -- dropped into a tiny standalone .c file next to a one-line
harness that exports it under a stable name, compiled with a host C
compiler into a shared library, and called through ctypes with the ONE
measured value this project has on record (0x8c75, from
device/qdc/test_on_device.py) -- never a value invented for this test alone.

THE DECISIVE PROPERTY. If the decode is ever removed -- e.g. reverted to
comparing the raw ARCH_VER against `arch` directly, the original bug --
`hexlib_decode_bcd_arch()` no longer exists in session.c, and
`_function_body` raises before this test ever gets to compile or call
anything. If the decode is present but wrong, the compiled call below
returns something other than 75. Either way, this test fails; it does not
merely fail to notice.

Adapted from llama.cpp's own htpdrv_get_arch (ggml-hexagon/htp-drv.cpp:
412-413, MIT; see ATTRIBUTION.md): `val = arch_ver & 0xff; arch = (val >> 4)
* 10 + (val & 0x0f)`.
"""
import ctypes
import pathlib
import shutil
import subprocess
import sys

import pytest

from hexlib.tests.csource import function_body as _function_body

SESSION_C = pathlib.Path("hexlib/runtime/host/session.c")

HOST_CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
needs_cc = pytest.mark.skipif(HOST_CC is None, reason="no host C compiler on PATH")


@pytest.fixture(scope="module")
def decode_fn_source():
    """The real hexlib_decode_bcd_arch() BODY (from its opening `{` through
    the matching `}` -- `csource.function_body`'s own documented slice;
    it deliberately does not include the signature line, since every other
    caller of this shared helper only ever inspects a body's control flow),
    sliced straight out of session.c and never retyped here, so a change to
    the real logic is exactly what this test exercises, not a
    hand-maintained duplicate that could silently drift from it. Raises
    (failing the test) if the function has been removed or renamed -- see
    the module docstring's "decisive property"."""
    src = SESSION_C.read_text()
    return _function_body(src, "hexlib_decode_bcd_arch")


@pytest.fixture
def compiled_decode(decode_fn_source, tmp_path):
    """Compile the extracted function into a shared library and return a
    ctypes callable bound to it. The signature wrapped around the extracted
    body below (`static uint32_t hexlib_decode_bcd_arch(uint32_t arch_ver)`)
    is copied verbatim from session.c's own declaration -- see
    hexlib_host.h/session.c -- only the BODY, the actual decode logic, comes
    from the slice; nothing about the arithmetic is retyped here."""
    c_path = tmp_path / "decode.c"
    c_path.write_text(
        "#include <stdint.h>\n"
        "static uint32_t hexlib_decode_bcd_arch(uint32_t arch_ver)\n"
        f"{decode_fn_source}\n"
        "#if defined(_WIN32)\n"
        "__declspec(dllexport)\n"
        "#endif\n"
        "uint32_t harness_decode(uint32_t arch_ver) {\n"
        "    return hexlib_decode_bcd_arch(arch_ver);\n"
        "}\n"
    )
    lib_path = tmp_path / ("decode.dll" if sys.platform == "win32" else "decode.so")
    cmd = [HOST_CC, "-shared", "-fPIC", "-o", str(lib_path), str(c_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"compiling the extracted hexlib_decode_bcd_arch() failed:\n"
        f"{result.stdout}\n{result.stderr}"
    )

    lib = ctypes.CDLL(str(lib_path))
    lib.harness_decode.restype = ctypes.c_uint32
    lib.harness_decode.argtypes = [ctypes.c_uint32]
    return lib.harness_decode


@needs_cc
def test_decode_of_the_measured_arch_ver_is_75(compiled_decode):
    """0x8c75 is the exact ARCH_VER this project has measured on real
    silicon (device/qdc/test_on_device.py); 75 is what __HEXAGON_ARCH__
    reports for the same part (skel.c / test_dsp_sim.py). This is the pair
    that hexlib_open() must agree on for a device session to ever open."""
    assert compiled_decode(0x8C75) == 75, (
        "hexlib_decode_bcd_arch(0x8c75) must be 75 -- the exact "
        "ARCH_VER/__HEXAGON_ARCH__ pair measured on real silicon"
    )


@needs_cc
def test_decode_is_a_real_bcd_decode_not_a_lookup_of_one_value(compiled_decode):
    """A couple of adjacent points so a decode that merely happens to get
    0x8c75 right (e.g. a one-entry lookup table, or `& 0xff` alone without
    the nibble split) cannot pass."""
    assert compiled_decode(0x8C73) == 73
    assert compiled_decode(0x0075) == 75
    # Only the LOW byte matters (0x1234 & 0xff == 0x34): high nibble 3,
    # low nibble 4 -> 3*10 + 4 == 34. A lookup keyed on the whole 32-bit
    # value, or one that used the high byte instead, would get this wrong.
    assert compiled_decode(0x1234) == 34


@needs_cc
def test_decode_does_not_return_the_raw_arch_ver(compiled_decode):
    """Guards directly against the original defect: if the decode were
    accidentally bypassed (the function body compiled but just returned its
    input), this value would be 35957, not 75."""
    raw = 0x8C75
    decoded = compiled_decode(raw)
    assert decoded != raw
    assert decoded == 75

import os

import pytest

from hexlib import toolchain as tc


def test_base_flags_are_pinned():
    assert tc.HVX_CFLAGS == [
        "-mv75", "-mhvx", "-mhvx-length=128B", "-std=gnu11", "-O2",
        "-Wall", "-Werror",
    ]


def test_warnings_are_errors_because_nothing_else_reads_them():
    """-Werror IS LOAD-BEARING, not tidiness. Every caller of `tc.run` decides
    success from `rc != 0` alone (hexlib/runtime/build.py's compile loops,
    hexlib/build.py, hexlib/exec/hexagon.py), so a warning is emitted and then
    discarded. The diagnostic that matters is
    -Wincompatible-pointer-types-discards-qualifiers: genentry.py emits each
    kernel's DSP entry by ARGUMENT ORDER, and swapping the `const` input with
    the mutable output is not a crash and not a bad status -- it is a plausible
    wrong answer, and that warning is the only automatic notice of it.

    MEASURED on toolchain 19.0.04 with the real generated scale_fp16_entry.c:
    with the two buffer casts swapped, `-Wall -Werror` -> rc=1 with that exact
    diagnostic; the same file under the OLD flags -> rc=0 with the same text as
    a warning. Removing either flag here restores that silence.
    """
    assert "-Werror" in tc.HVX_CFLAGS
    assert "-Wall" in tc.HVX_CFLAGS
    # -Wextra and -Wpedantic were measured and rejected -- see toolchain.py's
    # own comment for the numbers. Pinned so re-adding one is a deliberate act.
    assert "-Wpedantic" not in tc.HVX_CFLAGS
    assert "-Wextra" not in tc.HVX_CFLAGS


def test_the_hmx_flag_still_lands_after_the_warning_flags():
    """`cflags_for_caps` appends -mhmx, and `test_hmx_cap_adds_compiler_and_sim_
    flags` asserts it is LAST. Adding flags to HVX_CFLAGS must not have quietly
    changed which flag that is."""
    flags = tc.cflags_for_caps(["hmx"])
    assert flags[-1] == "-mhmx"
    assert flags[:-1] == tc.HVX_CFLAGS


def test_compiler_is_the_c_driver():
    """Kernels are GNU C. The vendored ggml-hexagon headers use the `asm`
    keyword and void* arithmetic, which are errors in C++; and every v6 expert
    is plain C already, so nothing wanted C++."""
    assert tc.COMPILER == "hexagon-clang"
    assert tc.STD == "gnu11"
    assert not hasattr(tc, "CXX_STD")


def test_sdk_include_dirs_names_what_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        tc.sdk_include_dirs(str(tmp_path))
    assert "qurt" in str(e.value)


def test_sdk_include_dirs_are_arch_specific(tmp_path):
    for sub in (("rtos", "qurt", "computev75", "include", "qurt"),
                ("rtos", "qurt", "computev75", "include", "posix"),
                ("incs",), ("incs", "stddef")):
        (tmp_path.joinpath(*sub)).mkdir(parents=True)
    dirs = tc.sdk_include_dirs(str(tmp_path))
    assert any("computev75" in d for d in dirs)
    assert len(dirs) == 4


def test_hmx_cap_adds_compiler_and_sim_flags():
    assert tc.cflags_for_caps(["hmx"])[-1] == "-mhmx"
    assert tc.sim_flags_for_caps(["hmx"]) == ["--mhmx", "2"]


def test_no_caps_is_unchanged():
    assert tc.cflags_for_caps([]) == tc.HVX_CFLAGS
    assert tc.cflags_for_caps([]) is not tc.HVX_CFLAGS  # a copy, not the shared list
    assert tc.sim_flags_for_caps([]) == []


def test_sdk_root_prefers_env(monkeypatch):
    monkeypatch.setenv("HEXAGON_SDK_ROOT", "/opt/hexagon")
    assert tc.default_sdk_root() == "/opt/hexagon"


def test_missing_sdk_names_the_variable_and_the_pattern(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        tc.find_toolchain_bin(str(tmp_path))
    msg = str(e.value)
    assert "HEXAGON_SDK_ROOT" in msg
    assert "HEXAGON_Tools" in msg


def test_toolchain_version_read_from_path(tmp_path):
    bin_dir = tmp_path / "tools" / "HEXAGON_Tools" / "19.0.04" / "Tools" / "bin"
    bin_dir.mkdir(parents=True)
    assert tc.toolchain_version(str(bin_dir)) == "19.0.04"


def test_find_toolchain_bin_picks_highest_version(tmp_path):
    for v in ("18.0.00", "19.0.04"):
        (tmp_path / "tools" / "HEXAGON_Tools" / v / "Tools" / "bin").mkdir(parents=True)
    assert tc.toolchain_version(tc.find_toolchain_bin(str(tmp_path))) == "19.0.04"


def test_run_never_raises_on_nonzero_exit():
    rc, out, err, timed_out = tc.run(
        ["python", "-c", "import sys; sys.exit(3)"], dict(os.environ)
    )
    assert rc == 3 and timed_out is False


def test_run_never_raises_on_a_missing_binary():
    """A missing or unexecutable binary must come back as a nonzero rc with a
    diagnostic, not as an OSError through every caller."""
    rc, out, err, timed_out = tc.run(
        ["hexlib-no-such-binary-anywhere"], dict(os.environ)
    )
    assert rc != 0 and timed_out is False
    assert "hexlib-no-such-binary-anywhere" in err


def test_run_decodes_non_utf8_without_dying():
    """Locale decoding (cp1252 on Windows) used to kill subprocess's reader thread
    on a single out-of-codepage byte, surfacing as a bogus compile failure."""
    rc, out, err, timed_out = tc.run(
        ["python", "-c", r"import sys; sys.stdout.buffer.write(b'\xff\xfe ok')"],
        dict(os.environ),
    )
    assert rc == 0 and "ok" in out

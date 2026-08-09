import os

import pytest

from hexlib import toolchain as tc


def test_base_flags_are_pinned():
    assert tc.HVX_CFLAGS == [
        "-mv75", "-mhvx", "-mhvx-length=128B", "-std=gnu11", "-O2",
    ]


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

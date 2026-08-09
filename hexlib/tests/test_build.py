# hexlib/tests/test_build.py
import os

import pytest

from hexlib import build, toolchain


def _have_sdk() -> bool:
    try:
        toolchain.find_toolchain_bin(toolchain.default_sdk_root())
        return True
    except FileNotFoundError:
        return False


def test_build_error_carries_compiler_output():
    e = build.BuildError("compile failed", "kernel.c:3:1: error: expected ';'")
    assert "expected ';'" in e.compiler_output


def test_command_includes_pinned_flags_and_include_path(tmp_path):
    """The compile command is assembled purely, so it can be asserted without an SDK."""
    cmd = build.compile_command(
        compiler="hexagon-clang++",
        sources=["kernel.c", "harness.c"],
        output="out.elf",
        caps=[],
        include_dirs=["include"],
    )
    for flag in ("-mv75", "-mhvx", "-mhvx-length=128B", "-std=c++17", "-O2"):
        assert flag in cmd
    assert "-Iinclude" in cmd
    assert cmd[-2:] == ["-o", "out.elf"] or "out.elf" in cmd


def test_command_adds_hmx_only_when_capped():
    assert "-mhmx" in build.compile_command("c", ["k.c"], "o", ["hmx"], [])
    assert "-mhmx" not in build.compile_command("c", ["k.c"], "o", [], [])


@pytest.mark.sdk
def test_builds_a_trivial_kernel(tmp_path):
    kdir = tmp_path / "trivial"
    kdir.mkdir()
    (kdir / "kernel_api.h").write_text(
        '#ifndef K\n#define K\nextern "C" void trivial(int *o);\n#endif\n'
    )
    (kdir / "kernel.c").write_text(
        '#include "kernel_api.h"\nextern "C" void trivial(int *o) { *o = 7; }\n'
    )
    (kdir / "harness.c").write_text(
        '#include "kernel_api.h"\n#include <stdio.h>\n'
        "int main(void) { int o = 0; trivial(&o); printf(\"%d\\n\", o); return 0; }\n"
    )
    out = build.build_kernel(str(kdir), str(tmp_path / "_work"), caps=[])
    assert os.path.isfile(out.elf)
    assert os.path.isfile(out.obj)
    assert out.toolchain_version == "19.0.04"


@pytest.mark.sdk
def test_compile_failure_reports_the_compiler_message(tmp_path):
    kdir = tmp_path / "broken"
    kdir.mkdir()
    (kdir / "kernel_api.h").write_text("#ifndef K\n#define K\n#endif\n")
    (kdir / "kernel.c").write_text("this is not c++\n")
    (kdir / "harness.c").write_text("int main(void) { return 0; }\n")
    with pytest.raises(build.BuildError) as e:
        build.build_kernel(str(kdir), str(tmp_path / "_work"), caps=[])
    assert "error" in e.value.compiler_output.lower()

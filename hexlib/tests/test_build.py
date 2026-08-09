# hexlib/tests/test_build.py
import os

import pytest

from hexlib import build, toolchain


def test_build_error_carries_compiler_output():
    e = build.BuildError("compile failed", "kernel.c:3:1: error: expected ';'")
    assert "expected ';'" in e.compiler_output


def test_command_includes_pinned_flags_and_include_path(tmp_path):
    """The compile command is assembled purely, so it can be asserted without an SDK."""
    cmd = build.compile_command(
        compiler="hexagon-clang",
        sources=["kernel.c", "harness.c"],
        output="out.elf",
        caps=[],
        include_dirs=["include"],
    )
    for flag in ("-mv75", "-mhvx", "-mhvx-length=128B", "-std=gnu11", "-O2"):
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
        '#ifndef K\n#define K\nvoid trivial(int *o);\n#endif\n'
    )
    (kdir / "kernel.c").write_text(
        '#include "kernel_api.h"\nvoid trivial(int *o) { *o = 7; }\n'
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
def test_harness_header_compiles_with_fp16_compare_calls(tmp_path):
    """Regression test: hexlib_close_f16 used to take __fp16 by value, which
    hexagon-clang rejects on the declaration alone ("parameters cannot have
    __fp16 type"), so any translation unit that merely included
    hexlib_harness.h failed to build. Task 4's other tests never include the
    header, so that break went uncaught. This test builds a real translation
    unit that includes the header and calls both compare functions with
    __fp16 arguments, through build.build_kernel so it goes through the real
    pinned-flag path rather than a hand-rolled compiler invocation.
    """
    kdir = tmp_path / "fp16harness"
    kdir.mkdir()
    (kdir / "kernel_api.h").write_text(
        '#ifndef K\n#define K\nvoid trivial(int *o);\n#endif\n'
    )
    (kdir / "kernel.c").write_text(
        '#include "kernel_api.h"\nvoid trivial(int *o) { *o = 7; }\n'
    )
    (kdir / "harness.c").write_text(
        '#include "kernel_api.h"\n'
        '#include "hexlib/hexlib_harness.h"\n'
        "int main(void) {\n"
        "    int o = 0;\n"
        "    trivial(&o);\n"
        "    __fp16 a = (__fp16) 1.0f, b = (__fp16) 1.0f;\n"
        "    int ok16 = hexlib_close_f16(a, b, 0.01f, 0.001f);\n"
        "    int ok32 = hexlib_close_f32((float) a, (float) b, 0.01f, 0.001f);\n"
        "    int correct = (o == 7) && ok16 && ok32;\n"
        "    hexlib_report(correct, correct ? 0 : 1, 0.0, 0ULL);\n"
        "    return 0;\n"
        "}\n"
    )
    out = build.build_kernel(str(kdir), str(tmp_path / "_work"), caps=[])
    assert os.path.isfile(out.elf)
    assert os.path.isfile(out.obj)


def _run_on_sim(bin_dir: str, elf: str) -> str:
    """Run ELF under hexagon-sim with the project's pinned bus/timing knobs and
    return combined stdout+stderr. Used only to prove runtime behaviour for the
    HEXLIB_TIME_KERNEL shadowing regression test below."""
    sim = os.path.join(bin_dir, toolchain.exe("hexagon-sim"))
    cmd = [
        sim,
        f"-m{toolchain.DSP_ARCH}",
        "--timing",
        "--buspenalty",
        str(toolchain.BUS_PENALTY),
        "--busratio",
        str(toolchain.BUS_RATIO),
        elf,
    ]
    env = toolchain.toolchain_env(bin_dir)
    rc, out, err, timed_out = toolchain.run(cmd, env, timeout=toolchain.SIM_TIMEOUT_S)
    assert not timed_out, f"hexagon-sim timed out: {cmd}"
    assert rc == 0, f"hexagon-sim exited {rc}: {(out + err)}"
    return out + err


@pytest.mark.sdk
def test_time_kernel_does_not_shadow_a_caller_local_named_c0(tmp_path):
    """Regression test: HEXLIB_TIME_KERNEL used to declare its cycle-counter
    temporaries as `_c0`/`_c1` in the same block STMT expands into. A caller
    with its own local named `_c0`, referenced inside the timed statement,
    got it silently shadowed by the cycle counter -- no compile error, wrong
    runtime behaviour. This harness declares its own `_c0 = 99`, passes it
    through a `use()` call inside HEXLIB_TIME_KERNEL, and the simulator run
    proves `use()` still saw 99 and not a cycle count.
    """
    kdir = tmp_path / "shadow"
    kdir.mkdir()
    (kdir / "kernel_api.h").write_text(
        '#ifndef K\n#define K\nvoid trivial(int *o);\n#endif\n'
    )
    (kdir / "kernel.c").write_text(
        '#include "kernel_api.h"\nvoid trivial(int *o) { *o = 7; }\n'
    )
    (kdir / "harness.c").write_text(
        '#include "kernel_api.h"\n'
        '#include "hexlib/hexlib_harness.h"\n'
        "static unsigned long long g_seen = 0;\n"
        "static void use(unsigned long long v) { g_seen = v; }\n"
        "int main(void) {\n"
        "    int o = 0;\n"
        "    trivial(&o);\n"
        "    unsigned long long _c0 = 99;\n"
        "    unsigned long long kcyc = 0;\n"
        "    HEXLIB_TIME_KERNEL(kcyc, use(_c0));\n"
        "    int shadow_intact = (g_seen == 99);\n"
        '    printf("SHADOW_CHECK seen=%llu\\n", g_seen);\n'
        "    hexlib_report(shadow_intact, shadow_intact ? 0 : 1, 0.0, kcyc);\n"
        "    return 0;\n"
        "}\n"
    )
    out = build.build_kernel(str(kdir), str(tmp_path / "_work"), caps=[])
    sim_output = _run_on_sim(out.bin_dir, out.elf)
    assert "SHADOW_CHECK seen=99" in sim_output
    assert "HEXLIB_VERDICT correct=1" in sim_output
    assert "HEXLIB_KCYCLES" in sim_output


@pytest.mark.sdk
def test_vendored_hvx_norm_header_builds_and_links(tmp_path):
    """Proves the Task 10 vendoring is actually usable: a kernel that includes
    hexlib/hvx/hvx-norm.h and calls hvx_fast_rms_norm_f32 must compile and link
    into an ELF through the real build.build_kernel path. Without this test, a
    future flag change (back to C++, or a dropped include dir) could silently
    make the vendored headers unusable again while every other test still
    passes, since none of them touch hvx/*.h.
    """
    kdir = tmp_path / "rmsnorm_vendor"
    kdir.mkdir()
    (kdir / "kernel_api.h").write_text(
        "#ifndef K\n#define K\n"
        "void rmsnorm_vendor(const float *src, float *dst, int n, float eps);\n"
        "#endif\n"
    )
    (kdir / "kernel.c").write_text(
        '#include "kernel_api.h"\n'
        '#include "hexlib/hvx/hvx-norm.h"\n'
        "#include <stdint.h>\n\n"
        "void rmsnorm_vendor(const float *src, float *dst, int n, float eps) {\n"
        "    hvx_fast_rms_norm_f32((const uint8_t *) src, (uint8_t *) dst, n, eps);\n"
        "}\n"
    )
    (kdir / "harness.c").write_text(
        '#include "kernel_api.h"\n'
        "int main(void) { return 0; }\n"
    )
    out = build.build_kernel(str(kdir), str(tmp_path / "_work"), caps=[])
    assert os.path.isfile(out.elf)
    assert os.path.isfile(out.obj)


@pytest.mark.sdk
def test_compile_failure_reports_the_compiler_message(tmp_path):
    kdir = tmp_path / "broken"
    kdir.mkdir()
    (kdir / "kernel_api.h").write_text("#ifndef K\n#define K\n#endif\n")
    (kdir / "kernel.c").write_text("this is not c\n")
    (kdir / "harness.c").write_text("int main(void) { return 0; }\n")
    with pytest.raises(build.BuildError) as e:
        build.build_kernel(str(kdir), str(tmp_path / "_work"), caps=[])
    assert "error" in e.value.compiler_output.lower()

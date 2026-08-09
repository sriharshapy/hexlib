# hexlib/build.py
"""Compile a kernel directory into a standalone simulator ELF.

TWO ARTIFACTS FROM ONE FLAG SET. The linked ELF is what runs; a kernel-only
object (-c, kernel source alone) is what the anti-cheat disassembles. Both use
exactly the same flags, from one place, so the two compiles can never drift and
disagree about what code was judged. Compiling the ELF and judging a differently
optimized object would make the accel proof meaningless.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from hexlib import toolchain as tc


class BuildError(Exception):
    def __init__(self, message: str, compiler_output: str = "") -> None:
        super().__init__(message)
        self.compiler_output = compiler_output


@dataclass(frozen=True)
class BuildOutput:
    elf: str
    obj: str
    bin_dir: str
    toolchain_version: str


def compile_command(
    compiler: str,
    sources: list[str],
    output: str,
    caps: list[str],
    include_dirs: list[str],
    compile_only: bool = False,
) -> list[str]:
    """Assemble a compile command. Pure — assertable without an SDK."""
    cmd = [compiler] + tc.cflags_for_caps(caps)
    for d in include_dirs:
        cmd.append(f"-I{d}")
    if compile_only:
        cmd.append("-c")
    cmd += list(sources)
    cmd += ["-o", output]
    return cmd


def build_kernel(
    kernel_dir: str,
    out_dir: str,
    caps: list[str],
    impl: str = "kernel.c",
    sdk_root: str | None = None,
) -> BuildOutput:
    """Compile <kernel_dir>/<impl> + harness.c into an ELF, and <impl> alone into
    an object for ELF-level accel detection.

    `impl` selects which implementation to build, so the same function builds the
    kernel, a near-miss variant, or a rival candidate during a bake-off.
    """
    bin_dir = tc.find_toolchain_bin(sdk_root or tc.default_sdk_root())
    version = tc.toolchain_version(bin_dir)
    if version != tc.TOOLCHAIN_VERSION:
        raise BuildError(
            f"toolchain is {version}, expected {tc.TOOLCHAIN_VERSION}. Cycle "
            "comparison across toolchain versions is invalid — pin the tools "
            "version or the numbers mean nothing."
        )

    env = tc.toolchain_env(bin_dir)
    compiler = os.path.join(bin_dir, tc.exe(tc.COMPILER))
    os.makedirs(out_dir, exist_ok=True)

    repo_include = os.path.join(os.path.dirname(os.path.dirname(__file__)), "include")
    includes = [kernel_dir, repo_include]

    stem = os.path.splitext(impl)[0]
    elf = os.path.join(out_dir, f"{stem}.elf")
    obj = os.path.join(out_dir, f"{stem}.o")

    link_cmd = compile_command(
        compiler,
        [os.path.join(kernel_dir, impl), os.path.join(kernel_dir, "harness.c")],
        elf,
        caps,
        includes,
    )
    rc, out, err, timed_out = tc.run(link_cmd, env, timeout=tc.SIM_TIMEOUT_S)
    if timed_out or rc != 0:
        raise BuildError(
            f"compile+link failed for {impl}", (out + err).strip()
        )

    obj_cmd = compile_command(
        compiler, [os.path.join(kernel_dir, impl)], obj, caps, includes,
        compile_only=True,
    )
    rc, out, err, timed_out = tc.run(obj_cmd, env, timeout=tc.SIM_TIMEOUT_S)
    if timed_out or rc != 0:
        raise BuildError(
            f"candidate-only compile failed for {impl}", (out + err).strip()
        )

    return BuildOutput(elf=elf, obj=obj, bin_dir=bin_dir, toolchain_version=version)

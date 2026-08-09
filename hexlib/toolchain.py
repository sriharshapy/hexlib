"""Hexagon SDK toolchain discovery and invocation.

Adapted from hexbench's `env/toolchain.py` (same author). The SDK is
license-restricted and is NEVER vendored, bundled, or fetched — it is discovered
through HEXAGON_SDK_ROOT.

WHY THE FLAGS ARE PINNED HERE AND NOWHERE ELSE. Cycle numbers are only
comparable when the toolchain and the simulator's bus model are identical.
Three known corpus outliers were traced to toolchain drift alone. So the
version is recorded in every result and a mismatch is an error, and the bus
knobs are set explicitly rather than left to the SDK's defaults, which move.

BUS_PENALTY/BUS_RATIO are the SDK's representative defaults, not device-matched
values. Simulator output is therefore reproducible, not silicon-validated.
"""
from __future__ import annotations

import glob
import os
import subprocess
from typing import Optional, Union

_WIN_SDK_DEFAULT = r"C:\Hexagon_SDK\6.4.0.2"

DSP_ARCH = "v75"
TOOLCHAIN_VERSION = "19.0.04"

# Engages the cycle-approximate microarchitectural model (caches + bus latency).
# With it off the simulator idealizes memory, which is misleading for
# bandwidth-bound kernels.
TIMING_MODE = True
BUS_PENALTY = 75
BUS_RATIO = 2

# GNU C, not C++. The pin here was inherited from a benchmark harness and was
# never load-bearing: all 351 experts in the source corpus are plain C, and
# building one both ways produces an identical instruction stream (only the
# mangled symbol name differs). Meanwhile the vendored ggml-hexagon headers
# REQUIRE GNU C -- hex-utils.h uses the `asm` keyword and void* arithmetic,
# which are errors in C++ -- so C is what makes that silicon-proven math
# library usable at all. C also removes name mangling, and with it the whole
# class of harness-linkage bugs that `extern "C"` exists to avoid.
STD = "gnu11"
COMPILER = "hexagon-clang"

HVX_CFLAGS = [f"-m{DSP_ARCH}", "-mhvx", "-mhvx-length=128B", f"-std={STD}", "-O2"]

SIM_TIMEOUT_S = 60
# An XL kernel gets more time, but this still kills genuine infinite loops.
# Measured need: DMA/VTCM kernels have run 204-406s under the timing model.
SIM_TIMEOUT_MAX_S = 900


def default_sdk_root() -> str:
    """HEXAGON_SDK_ROOT if set, else the Windows install default."""
    return os.environ.get("HEXAGON_SDK_ROOT", _WIN_SDK_DEFAULT)


def exe(name: str) -> str:
    """Append .exe on Windows; bare name elsewhere."""
    return name + (".exe" if os.name == "nt" else "")


def cflags_for_caps(caps: list[str]) -> list[str]:
    flags = list(HVX_CFLAGS)
    if "hmx" in caps:
        flags.append("-mhmx")
    return flags


def sim_flags_for_caps(caps: list[str]) -> list[str]:
    return ["--mhmx", "2"] if "hmx" in caps else []


def sdk_include_dirs(sdk_root: str, arch: str = DSP_ARCH) -> list[str]:
    """Include directories the vendored ggml-hexagon headers need.

    `hex-utils.h` includes <qurt.h> and <qurt_memory.h>, and `hex-dump.h`
    includes <HAP_farf.h>. These resolve inside the SDK but outside the
    toolchain bin directory, and the qurt path is arch-specific. Including them
    does NOT drag in a QuRT runtime dependency: a standalone simulator ELF using
    hvx-norm.h links and runs with no QuRT present.
    """
    qurt = os.path.join(sdk_root, "rtos", "qurt", f"compute{arch}", "include")
    dirs = [
        os.path.join(qurt, "qurt"),
        os.path.join(qurt, "posix"),
        os.path.join(sdk_root, "incs"),
        os.path.join(sdk_root, "incs", "stddef"),
    ]
    missing = [d for d in dirs if not os.path.isdir(d)]
    if missing:
        raise FileNotFoundError(
            "Hexagon SDK is missing include directories the vendored HVX headers "
            f"need:\n" + "\n".join(f"  {d}" for d in missing) +
            f"\nChecked under {sdk_root!r}. Set HEXAGON_SDK_ROOT to a full SDK "
            "installation."
        )
    return dirs


def find_toolchain_bin(sdk_root: str) -> str:
    """Locate .../tools/HEXAGON_Tools/<version>/Tools/bin inside the SDK."""
    pattern = os.path.join(sdk_root, "tools", "HEXAGON_Tools", "*", "Tools", "bin")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No Hexagon toolchain found under {sdk_root!r}.\n"
            f"Looked for: {pattern}\n"
            "Set HEXAGON_SDK_ROOT to your Hexagon SDK installation. The SDK is "
            "license-restricted and hexlib never downloads or bundles it."
        )
    return matches[-1]


def toolchain_version(bin_dir: str) -> str:
    """The HEXAGON_Tools version directory name, e.g. '19.0.04'."""
    parts = os.path.normpath(bin_dir).split(os.sep)
    try:
        return parts[parts.index("HEXAGON_Tools") + 1]
    except (ValueError, IndexError):
        raise ValueError(
            f"Cannot read a toolchain version from {bin_dir!r}; expected a path "
            "containing .../HEXAGON_Tools/<version>/Tools/bin"
        )


def toolchain_env(bin_dir: str, base_env: Optional[dict] = None) -> dict:
    """Subprocess env with the toolchain on PATH, plus its lib dirs on
    LD_LIBRARY_PATH on non-Windows (the Linux ISS needs them)."""
    env = dict(base_env if base_env is not None else os.environ)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    if os.name != "nt":
        tools = os.path.dirname(bin_dir)
        libs = [
            d
            for d in (os.path.join(tools, "lib"), os.path.join(tools, "lib", "iss"))
            if os.path.isdir(d)
        ]
        if libs:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = os.pathsep.join(libs + ([existing] if existing else []))
    return env


def run(
    cmd: list, env: dict, timeout: Optional[float] = None
) -> tuple[Union[int, None], str, str, bool]:
    """Run a command, capture stdout+stderr, never raise on nonzero exit,
    timeout, or a missing binary.

    Returns (returncode | None if timed out, stdout, stderr, timed_out).

    DECODING. UTF-8 with errors="replace", never the locale codec. Locale decoding
    (cp1252 on Windows) dies inside subprocess's reader thread on one
    out-of-codepage byte; the thread dies, stderr comes back None, and the caller
    reports a compile failure for a kernel that compiled fine. Observed twice on
    real kernels. Diagnostics are for humans and substring matching, so a
    replacement character always beats losing the process result.
    """
    try:
        p = subprocess.run(
            cmd,
            env=env,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
        )
        return p.returncode, p.stdout, p.stderr, False
    except subprocess.TimeoutExpired as e:
        out, err = e.stdout or "", e.stderr or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        if isinstance(err, bytes):
            err = err.decode(errors="replace")
        return None, out, err, True
    except OSError as e:
        # A missing or unexecutable binary. Returning a nonzero rc rather than
        # propagating means every existing "if rc != 0" branch handles it
        # correctly and without change: build.py raises BuildError with this
        # text, and anticheat.disassemble returns '' -- which is what its
        # docstring already promises.
        return 127, "", f"cannot execute {cmd[0]!r}: {e}", False

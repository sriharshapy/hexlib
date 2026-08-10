"""Dispatch a plan's op to a real Hexagon kernel on the simulator.

HOW IT WORKS, and why it works at all. A kernel directory may carry a
`runner.c` alongside its `harness.c`: same kernel, different main(). The runner
reads its inputs from a file and writes its output to a file, and a standalone
simulator ELF turns out to have working host file I/O -- verified directly,
`fopen`/`fread`/`fwrite` against files the host created. That is the whole
mechanism, and it is what makes this possible without FastRPC, without a DSP
skel, and without a device.

WHAT THIS COSTS. One simulator launch per op invocation, tens of seconds each
under the timing model. A 308-step encoder is therefore tens of minutes, not
milliseconds. That is accepted: this path exists to establish that the values a
real kernel produces are right, and it is measured once, not in a loop. Making
it fast means one ELF containing every kernel plus a plan-walking driver, which
is a different piece of work.

WHY THE RUNNER IS A SEPARATE BINARY FROM THE HARNESS. The harness's value is
that it builds its own inputs and cannot be handed a passing answer; a harness
that read host-supplied data would lose exactly that. So the runner never
prints the gate's verdict line, and the harness never reads a file.
"""
from __future__ import annotations

import os
import struct
import tempfile
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from hexlib import kerneldir as kd
from hexlib import toolchain as tc

IN_NAME = "hexlib_in.bin"
OUT_NAME = "hexlib_out.bin"
RUNNER_SRC = "runner.c"


class HexagonBackendError(Exception):
    pass


@dataclass
class RunnerStats:
    calls: int = 0
    kernel_cycles: int = 0
    max_n: int = 0


def has_runner(kernel_dir: str) -> bool:
    return os.path.isfile(os.path.join(kernel_dir, RUNNER_SRC))


def build_runner(kernel_dir: str, out_dir: str, sdk_root: str | None = None) -> str:
    """Compile kernel.c + runner.c into a standalone ELF. Returns its path.

    `harness.c` and `baseline.c` are deliberately NOT linked: the runner has its
    own main(), and linking the harness's would be a duplicate symbol. Excluding
    baseline.c also means the ELF cannot accidentally compute the reference and
    report it as the kernel's output.
    """
    if not has_runner(kernel_dir):
        raise HexagonBackendError(
            f"{kernel_dir} has no {RUNNER_SRC}, so it cannot be driven with "
            "host data. Add one, or use the reference backend for this op."
        )
    root = sdk_root or tc.default_sdk_root()
    bin_dir = tc.find_toolchain_bin(root)
    version = tc.toolchain_version(bin_dir)
    if version != tc.TOOLCHAIN_VERSION:
        raise HexagonBackendError(
            f"toolchain is {version}, expected {tc.TOOLCHAIN_VERSION}"
        )
    spec = kd.load_spec(kernel_dir)
    env = tc.toolchain_env(bin_dir)
    compiler = os.path.join(bin_dir, tc.exe(tc.COMPILER))
    os.makedirs(out_dir, exist_ok=True)

    # This file is hexlib/exec/hexagon.py, so the repo root is THREE levels up,
    # one more than build.py needs from hexlib/build.py.
    _repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    repo_include = os.path.join(_repo_root, "include")
    includes = [kernel_dir, repo_include] + tc.sdk_include_dirs(root)
    elf = os.path.join(out_dir, f"{os.path.basename(kernel_dir)}.runner.elf")

    cmd = [compiler] + tc.cflags_for_caps(spec.caps)
    for d in includes:
        cmd.append(f"-I{d}")
    cmd += [
        os.path.join(kernel_dir, "kernel.c"),
        os.path.join(kernel_dir, RUNNER_SRC),
        "-o",
        elf,
    ]
    rc, out, err, timed_out = tc.run(cmd, env, timeout=tc.SIM_TIMEOUT_S)
    if timed_out or rc != 0:
        raise HexagonBackendError(
            f"compiling the runner for {kernel_dir} failed:\n{(out + err).strip()}"
        )
    return elf


def _run_elf(elf: str, cwd: str, caps: list[str]) -> str:
    """Run the ELF with `cwd` as its working directory, so its relative file
    paths resolve into a directory we control."""
    from hexlib.sim import sim_command

    bin_dir = tc.find_toolchain_bin(tc.default_sdk_root())
    env = tc.toolchain_env(bin_dir)
    sim_exe = os.path.join(bin_dir, tc.exe("hexagon-sim"))
    cmd = sim_command(sim_exe, os.path.abspath(elf), caps)

    prev = os.getcwd()
    os.chdir(cwd)
    try:
        rc, out, err, timed_out = tc.run(cmd, env, timeout=tc.SIM_TIMEOUT_MAX_S)
    finally:
        os.chdir(prev)

    combined = out + err
    if timed_out:
        raise HexagonBackendError(f"simulator timed out running {elf}\n{combined}")
    if rc != 0:
        raise HexagonBackendError(
            f"simulator exited {rc} running {elf}. A faulted run has not "
            f"computed anything.\n{combined}"
        )
    if "RUNNER error=" in combined:
        raise HexagonBackendError(f"runner reported an error:\n{combined}")
    return combined


def _out_shape(spec, arrays: tuple[np.ndarray, ...], attrs: Mapping[str, Any]):
    """The output's shape.

    Elementwise ops keep the first input's shape. A permutation does not -- its
    output is the input's shape reordered by `perm` -- and reusing the input shape
    there would reshape the returned bytes wrongly and hand back garbage that
    still has the right element count.
    """
    shape = tuple(arrays[0].shape)
    perm = attrs.get("perm")
    if perm is not None:
        return tuple(shape[i] for i in perm)
    declared = attrs.get("shape")
    if declared is not None:
        return tuple(declared)
    return shape


def backend_for(
    kind: str,
    work_dir: str | None = None,
    stats: RunnerStats | None = None,
    sdk_root: str | None = None,
    kernel_dir: str | None = None,
):
    """A backend for `kind`, computed by its real kernel, driven by its RunnerSpec.

    This is the general path: the wire format comes from the declarative spec in
    `exec.runner`, so adding a kernel is a spec entry plus a `runner.c`, not
    another bespoke marshaller. Returns None when `kind` has no spec, which is
    what lets the executor fall back to the reference for op kinds whose kernel
    does not exist yet -- so the encoder runs at every stage rather than only at
    the end.
    """
    from hexlib.exec.runner import spec_for

    spec = spec_for(kind)
    if spec is None:
        return None

    directory = kernel_dir or spec.kernel_dir
    work = work_dir or tempfile.mkdtemp(prefix=f"hexlib-{kind}-")
    os.makedirs(work, exist_ok=True)
    elf = build_runner(directory, work, sdk_root=sdk_root)
    caps = kd.load_spec(directory).caps
    tracker = stats if stats is not None else RunnerStats()

    def backend(
        arrays: tuple[np.ndarray, ...], attrs: Mapping[str, Any]
    ) -> tuple[np.ndarray, ...]:
        # An op kind is not always one kernel. Refused before the kernel runs,
        # because the failure mode otherwise is a correctly-shaped wrong answer.
        spec.check_requires(attrs)
        out_shape = _out_shape(spec, arrays, attrs)
        with open(os.path.join(work, IN_NAME), "wb") as f:
            f.write(spec.encode(arrays, attrs))

        out_path = os.path.join(work, OUT_NAME)
        if os.path.exists(out_path):
            # A stale output would be read as this call's result if the runner
            # failed to write one.
            os.remove(out_path)

        text = _run_elf(elf, work, caps)

        if not os.path.exists(out_path):
            raise HexagonBackendError(
                f"{kind}: the runner produced no {OUT_NAME}; nothing was "
                f"computed.\n{text}"
            )
        with open(out_path, "rb") as f:
            raw = f.read()
        result = spec.decode(raw, out_shape)

        tracker.calls += 1
        tracker.max_n = max(tracker.max_n, int(arrays[0].size))
        for token in text.split():
            if token.startswith("cycles="):
                tracker.kernel_cycles += int(token.split("=", 1)[1])
        return (result,)

    return backend


def scale_backend(
    kernel_dir: str,
    work_dir: str | None = None,
    stats: RunnerStats | None = None,
    sdk_root: str | None = None,
):
    """A backend for the `scale` op kind, computed by the real HVX kernel.

    Compiles once, then one simulator launch per call. The op's activations are
    fp16, so the fp32 the interpreter carries is rounded on the way in and
    widened on the way out -- the same boundary the VTCM image applies, and the
    same one the hardware applies.
    """
    work = work_dir or tempfile.mkdtemp(prefix="hexlib-scale-")
    os.makedirs(work, exist_ok=True)
    elf = build_runner(kernel_dir, work, sdk_root=sdk_root)
    caps = kd.load_spec(kernel_dir).caps
    tracker = stats if stats is not None else RunnerStats()

    def backend(
        arrays: tuple[np.ndarray, ...], attrs: Mapping[str, Any]
    ) -> tuple[np.ndarray, ...]:
        if len(arrays) != 1:
            raise HexagonBackendError(
                f"scale takes one input, got {len(arrays)}"
            )
        x = np.ascontiguousarray(arrays[0], dtype=np.float16)
        n = int(x.size)
        factor = float(attrs["factor"])

        payload = struct.pack("<if", n, factor) + x.tobytes()
        with open(os.path.join(work, IN_NAME), "wb") as f:
            f.write(payload)
        out_path = os.path.join(work, OUT_NAME)
        if os.path.exists(out_path):
            # A stale output from a previous call would be read as this call's
            # result if the runner failed to write one.
            os.remove(out_path)

        text = _run_elf(elf, work, caps)

        if not os.path.exists(out_path):
            raise HexagonBackendError(
                f"the runner produced no {OUT_NAME}; nothing was computed.\n{text}"
            )
        raw = open(out_path, "rb").read()
        expect = n * 2
        if len(raw) != expect:
            raise HexagonBackendError(
                f"{OUT_NAME} is {len(raw)} bytes, expected {expect} for n={n}"
            )
        y = np.frombuffer(raw, dtype=np.float16).reshape(arrays[0].shape)

        tracker.calls += 1
        tracker.max_n = max(tracker.max_n, n)
        for token in text.split():
            if token.startswith("cycles="):
                tracker.kernel_cycles += int(token.split("=", 1)[1])
        return (y.astype(np.float32),)

    return backend

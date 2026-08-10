# hexlib/runtime/build.py
"""Build the runtime: qaic, the DSP skel, the simulator qexe, the device binary.

`hexlib/build.py` is untouched — it builds standalone kernel ELFs and its
contract is depended on by the whole existing gate. This is a second builder for
a second kind of artifact, sharing only `toolchain.py`.

THE LINK RECIPE IS NOT RECONSTRUCTED. The simulator flags and libraries below
were recovered from the SDK calculator example's own `calculator_q_link.txt`
after building and running it at v75 on this toolchain, where it printed
`Sum = 32640 / Pass: 2 Fail: 0` at rev_id 0x00008c75. They are known to work.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from hexlib import toolchain as tc


class RuntimeBuildError(Exception):
    def __init__(self, message: str, output: str = "") -> None:
        super().__init__(message)
        self.output = output


@dataclass(frozen=True)
class QaicOutput:
    header: str
    stub: str
    skel: str


def qaic_path(sdk_root: str) -> str:
    """qaic lives in a per-platform directory inside the SDK."""
    if os.name == "nt":
        return os.path.join(sdk_root, "ipc", "fastrpc", "qaic", "bin", "qaic.exe")
    return os.path.join(sdk_root, "ipc", "fastrpc", "qaic", "Ubuntu", "qaic")


def qaic_include_dirs(sdk_root: str) -> list[str]:
    """AEEStdDef.idl and remote.idl live here."""
    return [os.path.join(sdk_root, "incs"), os.path.join(sdk_root, "incs", "stddef")]


def run_qaic(idl: str, out_dir: str, sdk_root: str | None = None) -> QaicOutput:
    root = sdk_root or tc.default_sdk_root()
    if not os.path.isfile(idl):
        raise RuntimeBuildError(f"IDL not found: {idl}")
    qaic = qaic_path(root)
    if not os.path.isfile(qaic):
        raise RuntimeBuildError(f"qaic not found: {qaic}")

    os.makedirs(out_dir, exist_ok=True)
    cmd = [qaic, "-mdll", "-o", out_dir]
    for d in qaic_include_dirs(root):
        cmd += ["-I", d]
    cmd.append(idl)

    rc, out, err, timed_out = tc.run(cmd, os.environ.copy(), timeout=60)
    if timed_out or rc != 0:
        raise RuntimeBuildError(f"qaic failed on {idl}", (out + err).strip())

    stem = os.path.splitext(os.path.basename(idl))[0]
    res = QaicOutput(
        header=os.path.join(out_dir, f"{stem}.h"),
        stub=os.path.join(out_dir, f"{stem}_stub.c"),
        skel=os.path.join(out_dir, f"{stem}_skel.c"),
    )
    # FAIL CLOSED: qaic exiting 0 without writing the files is a failure, not a
    # build we then link and get confusing errors from. Covered by
    # test_qaic_exit_zero_without_files_still_raises, which monkeypatches
    # tc.run to succeed while writing nothing -- do not delete this as
    # "redundant" with the happy-path test; that one can't fail this check.
    for f in (res.header, res.stub, res.skel):
        if not os.path.isfile(f):
            raise RuntimeBuildError(
                f"qaic exited 0 but did not produce {f}", (out + err).strip()
            )
    return res

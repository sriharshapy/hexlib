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


# ============================================================================
# The simulator qexe: skel library + host, one Hexagon ELF.
#
# THE LINK RECIPE IS NOT RECONSTRUCTED. SIM_LINK_FLAGS and SIM_LINK_EXTRAS were
# recovered from the SDK calculator example's own `calculator_q_link.txt` after
# building and running it at v75 on toolchain 19.0.04, where it printed
# `Sum = 32640 / Pass: 2 Fail: 0` at rev_id 0x00008c75. They are also directly
# confirmable in the SDK's own make rules: EXE_LD_FLAGS in
# build/make.d.ext/hexagon/defines_hexagon_1_9.min is exactly LD_FLAGS (-m<arch>
# -G0, the two --defsym flags, --no-threads) plus --dynamic-linker=, -E, and
# --force-dynamic,-u,main.
#
# THE GENERATED STUB IS NEVER COMPILED INTO THIS QEXE, ON PURPOSE. qaic's
# generated hexlib_iface_stub.c defines hexlib_iface_open/_close/_start/_stop/
# _mmap/_munmap/_hwinfo/_invoke as HOST-side wrappers that marshal and call
# remote_handle64_open/_invoke/_close. hexlib/runtime/skel/skel.c defines the
# SAME function names as the DSP-side developer implementation (confirmed by
# running qaic and reading both generated files back). On a device these live
# in two different ELFs (host APK vs. DSP .so) so the names never collide;
# statically linking both into one qexe is a duplicate-symbol link error.
# calculator_q's own hexagon.min settles how the SDK itself avoids this: it
# never adds calculator_stub.c to calculator_q's sources, only the generated
# *_skel.c (present but unused here -- nothing in this qexe references its one
# exported symbol, hexlib_iface_skel_handle_invoke, so the archive's lazy
# member extraction never pulls it in) and the developer's skel-side
# implementation. `hexagon-nm` on rtld.a/test_util.a/atomic.a confirms none of
# them define remote_handle64_open/_close/_invoke at all -- there would be
# nothing for the stub to call even if it were linked. simhost.c therefore
# calls hexlib_iface_open/_start/_mmap/_invoke/_stop/_close as plain C
# functions, which the linker binds directly to skel.c's definitions: one
# address space, one function table, no marshaling.
SIM_LINK_FLAGS = [
    "-G0",
    "-Wl,--defsym=ISDB_TRUSTED_FLAG=2",
    "-Wl,--defsym=ISDB_SECURE_FLAG=2",
    "-Wl,--no-threads",
    "-Wl,--dynamic-linker=",
    "-Wl,-E",
    "-Wl,--force-dynamic,-u,main",
]


def SIM_LINK_EXTRAS(sdk_root: str, tools_root: str) -> list[str]:
    """Prebuilt libraries a standalone (NO_QURT_INC-style) qexe needs.

    test_util.a and atomic.a ship only for v68 and link correctly against v75
    (confirmed: `hexagon-nm test_util.a` resolves cleanly at v75 link time, and
    the SDK's own calculator.min uses the identical v68 archives for a v75
    qexe). test_util.a is also where rpcmem_alloc/rpcmem_to_fd/rpcmem_free are
    actually DEFINED for a standalone Hexagon build -- rpcmem.h has no
    inline/static implementation of them, and the only prebuilt `rpcmem.a` in
    the SDK targets v68, not v75. Using test_util.a's rpcmem avoids that
    mismatch entirely rather than risking it.
    """
    j = os.path.join
    return [
        j(sdk_root, "ipc", "fastrpc", "rtld", "ship", "hexagon_toolv19_v75", "rtld.a"),
        j(sdk_root, "utils", "sim_utils", "prebuilt", "hexagon_toolv19_v68", "test_util.a"),
        j(sdk_root, "libs", "atomic", "prebuilt", "hexagon_toolv19_v68", "atomic.a"),
        j(tools_root, "Tools", "target", "hexagon", "lib", "v75", "G0", "libhexagon.a"),
    ]


def runtime_include_dirs(sdk_root: str, gen_dir: str) -> list[str]:
    repo = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return [
        gen_dir,
        os.path.join(repo, "hexlib", "runtime", "skel"),
        os.path.join(repo, "include"),
        os.path.join(sdk_root, "incs"),
        os.path.join(sdk_root, "incs", "stddef"),
        os.path.join(sdk_root, "ipc", "fastrpc", "rpcmem", "inc"),
        # `tc.sdk_include_dirs` already appends
        # rtos/qurt/compute<arch>/include/qurt (needed by skel_vtcm.c's
        # `#include "qurt_thread.h"`) and its posix/ sibling.
    ] + tc.sdk_include_dirs(sdk_root)


def build_skel_lib(kernels: list[str], out_dir: str,
                   sdk_root: str | None = None) -> str:
    """qaic, generate entries, compile skel + kernels, archive.

    `kernels` is a REQUEST, not the full set that gets compiled: genentry's
    dispatch table always references every kernel that has a RunnerSpec AND a
    directory on disk, regardless of what this function was asked to build, so
    every one of those must be compiled into the archive or the later link
    fails on an undefined symbol for whichever one is missing.
    """
    from hexlib.build import compile_command
    from hexlib.exec.runner import SPECS
    from hexlib.runtime import genentry

    root = sdk_root or tc.default_sdk_root()
    bin_dir = tc.find_toolchain_bin(root)
    version = tc.toolchain_version(bin_dir)
    if version != tc.TOOLCHAIN_VERSION:
        raise RuntimeBuildError(
            f"toolchain is {version}, expected {tc.TOOLCHAIN_VERSION} — cycle "
            "numbers are not comparable across toolchain versions"
        )
    env = tc.toolchain_env(bin_dir)
    compiler = os.path.join(bin_dir, tc.exe(tc.COMPILER))
    ar = os.path.join(bin_dir, tc.exe("hexagon-ar"))

    repo = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    gen = os.path.join(out_dir, "gen")
    os.makedirs(gen, exist_ok=True)

    idl = os.path.join(repo, "hexlib", "runtime", "idl", "hexlib_iface.idl")
    qa = run_qaic(idl, gen, root)

    # The REPO root: spec.kernel_dir already carries the "kernels/" prefix.
    entries = genentry.generate(repo, gen)

    used_kernels = sorted({
        os.path.basename(spec.kernel_dir)
        for spec in SPECS.values()
        if os.path.isdir(os.path.join(repo, spec.kernel_dir))
    })
    all_kernels = sorted(set(kernels) | set(used_kernels))

    skel_dir = os.path.join(repo, "hexlib", "runtime", "skel")
    # Each entry below is (source path, object basename, this file's OWN extra
    # include dirs). Two things go wrong if a single shared include list and a
    # single `basename(s).replace(".c", ".o")` are used for all of these, both
    # confirmed by actually hitting them:
    #   1. Every kernel directory has a "kernel_api.h" with a DIFFERENT
    #      declared function inside, so a generated <fn>_entry.c file must see
    #      ONLY its own kernel's directory on the include path -- putting
    #      every kernel dir on one shared list let scale_fp16_entry.c's
    #      `#include "kernel_api.h"` resolve to add_fp16/kernel_api.h instead
    #      (whichever kernel sorts first), failing with "call to undeclared
    #      function 'scale_fp16'".
    #   2. Every kernel's implementation file is literally named "kernel.c",
    #      so compiling them all to a name derived from their own basename
    #      collided on disk: each subsequent kernel.o silently overwrote the
    #      previous one, and the archive ended up with only the
    #      alphabetically-last kernel's code (transpose_th_fp16) -- `nm` on
    #      the resulting archive showed the other three kernels' *_entry.o
    #      referencing their own kernel function as undefined.
    sim_shims = os.path.join(repo, "hexlib", "runtime", "simhost", "sim_shims.c")
    base = [
        (qa.skel, "hexlib_iface_skel.o", []),
        (os.path.join(skel_dir, "skel.c"), "skel.o", []),
        (os.path.join(skel_dir, "skel_bufs.c"), "skel_bufs.o", []),
        (os.path.join(skel_dir, "skel_vtcm.c"), "skel_vtcm.o", []),
        (os.path.join(skel_dir, "skel_dispatch.c"), "skel_dispatch.o", []),
        (sim_shims, "sim_shims.o", []),
    ]
    for e in entries:
        stem = os.path.basename(e)[:-len(".c")]
        if stem.endswith("_entry"):
            k = stem[: -len("_entry")]
            base.append((e, f"{stem}.o", [os.path.join(repo, "kernels", k)]))
        else:
            base.append((e, f"{stem}.o", []))  # hexlib_kernel_table.c
    for k in all_kernels:
        kdir = os.path.join(repo, "kernels", k)
        base.append((os.path.join(kdir, "kernel.c"), f"{k}_kernel.o", [kdir]))
        hand = os.path.join(kdir, "dsp_entry.c")
        if os.path.isfile(hand):
            base.append((hand, f"{k}_dsp_entry.o", [kdir]))

    common_includes = runtime_include_dirs(root, gen)

    objs = []
    for s, obj_name, extra in base:
        o = os.path.join(out_dir, obj_name)
        cmd = compile_command(
            compiler, [s], o, ["hvx"], common_includes + extra, compile_only=True
        )
        rc, out, err, to = tc.run(cmd, env, timeout=tc.SIM_TIMEOUT_S)
        if to or rc != 0:
            raise RuntimeBuildError(f"compile failed: {s}", (out + err).strip())
        objs.append(o)

    lib = os.path.join(out_dir, "libhexlib_skel.a")
    rc, out, err, to = tc.run([ar, "rcs", lib] + objs, env, timeout=60)
    if to or rc != 0 or not os.path.isfile(lib):
        raise RuntimeBuildError("archiving libhexlib_skel.a failed",
                                (out + err).strip())
    return lib


def build_sim_qexe(out_dir: str, sdk_root: str | None = None) -> str:
    """Link the simulator host + skel + rtld into one runnable ELF.

    The qaic-generated stub is deliberately NOT one of the sources here -- see
    the module-level comment above SIM_LINK_FLAGS for why linking it alongside
    skel.c would be a duplicate-symbol error, not merely redundant.
    """
    root = sdk_root or tc.default_sdk_root()
    bin_dir = tc.find_toolchain_bin(root)
    env = tc.toolchain_env(bin_dir)
    compiler = os.path.join(bin_dir, tc.exe(tc.COMPILER))
    tools_root = os.path.dirname(os.path.dirname(bin_dir))
    repo = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    gen = os.path.join(out_dir, "gen")

    lib = os.path.join(out_dir, "libhexlib_skel.a")
    if not os.path.isfile(lib):
        raise RuntimeBuildError(f"build_skel_lib must run first: {lib} missing")

    elf = os.path.join(out_dir, "hexlib_q")
    cmd = [compiler] + tc.cflags_for_caps(["hvx"]) + SIM_LINK_FLAGS
    for d in runtime_include_dirs(root, gen):
        cmd.append(f"-I{d}")
    cmd += ["-o", elf, "-Wl,--start-group",
            os.path.join(repo, "hexlib", "runtime", "simhost", "simhost.c"),
            lib]
    cmd += SIM_LINK_EXTRAS(root, tools_root)
    cmd += ["-Wl,--end-group"]

    rc, out, err, to = tc.run(cmd, env, timeout=tc.SIM_TIMEOUT_S)
    if to or rc != 0 or not os.path.isfile(elf):
        raise RuntimeBuildError("linking hexlib_q failed", (out + err).strip())
    return elf

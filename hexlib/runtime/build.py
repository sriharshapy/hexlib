# hexlib/runtime/build.py
"""Build the runtime: qaic, the DSP skel, the simulator artifact, the device binary.

`hexlib/build.py` is untouched — it builds standalone kernel ELFs and its
contract is depended on by the whole existing gate. This is a second builder for
a second kind of artifact, sharing only `toolchain.py`.

THE LINK RECIPES ARE NOT RECONSTRUCTED. They were recovered by actually
building SDK reference examples on this toolchain and reading back the exact
commands the SDK's own build system used, never guessed:

- The (now-retired) standalone-qexe recipe was recovered from the SDK
  calculator example's own `calculator_q_link.txt` at v75, where it printed
  `Sum = 32640 / Pass: 2 Fail: 0` at rev_id 0x00008c75.
- The QuRT-hosted `.so` recipe below (`build_sim_so`, `SIM_SO_LINK_FLAGS`) was
  recovered from the SDK's own `libs/run_main_on_hexagon` example's
  `test_main_so_link.txt`, `run_main_on_hexagon_sim_link.txt`, and
  `sim_cmd_line.txt`, produced by `make hexagon BUILD=Debug DSP_ARCH=v75` in
  that example directory (never inside this repo, never checked in — see
  `.superpowers/sdd/2026-08-10-silicon-path-runtime/
  investigation-sim-vtcm-and-marshalling.md`). That run demonstrated a real
  8 MiB VTCM acquisition succeeding (`rc=0`, `ptr=0xd9000000`) under a real
  QuRT kernel on `hexagon-sim`; the standalone qexe cannot do this (VTCM's
  manager object needs real QuRT thread/clock primitives no standalone qexe
  can provide — see the same investigation).

WHY THE STANDALONE QEXE RECIPE IS GONE. `build_sim_qexe`/`SIM_LINK_FLAGS`/
`SIM_LINK_EXTRAS` used to live here (task 7). They still link and still reach
`hexlib_iface_open`, but `hexlib_iface_start` can NEVER succeed through that
path — VTCM acquisition needs real QuRT, which a standalone
`--force-dynamic` qexe structurally cannot host. Keeping a build path around
that is permanently incapable of the one thing this runtime exists to prove
is worse than removing it: a future reader would have to rediscover, by
hitting the same wall again, that it is a dead end. The QuRT-hosted `.so`
below is the only path that reaches a successful `start()` in simulation.
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
# The historical standalone-qexe recipe (task 7's SIM_LINK_FLAGS/
# SIM_LINK_EXTRAS/build_sim_qexe) lived here and is gone -- see the module
# docstring's "WHY THE STANDALONE QEXE RECIPE IS GONE" for why. The recipe
# below replaces it: a QuRT-hosted shared object (see "The simulator
# artifact" further down).
# ============================================================================


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
        # -fpic: this archive is only ever linked into build_sim_so's shared
        # object below (a real device skel is ALSO always built as a shared
        # object loaded by qaic's own dlopen machinery — this is not a
        # simulator-only concession, it is the same code shape a device skel
        # needs). PIC objects link into a --force-dynamic executable too, so
        # this does not foreclose reusing the archive for a non-PIC link
        # later if one is ever needed.
        cmd.insert(1, "-fpic")
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


# ============================================================================
# The simulator artifact: a QuRT-hosted shared object, dlopen'd by the SDK's
# OWN prebuilt `run_main_on_hexagon_sim` under a real booted QuRT kernel.
#
# WHY A SHARED OBJECT, NOT A STANDALONE QEXE. Task 7's standalone
# `--force-dynamic` qexe (gone now, see the module docstring) never links in
# the VTCM manager's weak symbols with real definitions, and forcing that
# object in demands real QuRT thread/clock primitives a standalone qexe
# cannot provide (investigation-sim-vtcm-and-marshalling.md, Q1). The SDK's
# OWN way to run arbitrary code under a real QuRT kernel on `hexagon-sim` is
# `libs/run_main_on_hexagon`: a prebuilt host executable
# (`run_main_on_hexagon_sim`, already shipped for this exact toolchain/arch
# combination at
# `$SDK/libs/run_main_on_hexagon/ship/hexagon_toolv19_v75/run_main_on_hexagon_sim`)
# that links real `libqurt.a` + `rtld.a` + `test_util.a` + `atomic.a`
# `--whole-archive` (confirmed by reading its own recovered link line), boots
# a real QuRT kernel under `hexagon-sim` via `runelf.pbn` + `osam.cfg`, then
# `dlopen()`s a user-supplied `.so` and calls its `main()`. `test_main.so` is
# the SDK's own reference payload for this; `build_sim_so` below produces
# hexlib's own payload the identical way, recovered from that same example's
# `test_main_so_link.txt` (`-fpic -shared -Wl,-Bsymbolic -lc`, nothing else --
# rtld/test_util/atomic/libqurt are NOT relinked into the .so, because they
# are already inside the host process that dlopen's it, and its own -E/
# --export-dynamic-equivalent link makes their symbols visible to the .so at
# dlopen time). This is what makes VTCM acquisition, real: HAP_compute_res.h's
# weak `compute_resource_query_VTCM` pointer, unresolved (null) in a
# standalone qexe, resolves for real here against test_util.a's
# sysmon_vtcm_mgr_client.o -- and its hard qurt_thread_get_id/
# qurt_sysclock_get_hw_ticks/etc. dependencies resolve against the real
# libqurt.a already linked into the host process. Demonstrated end to end in
# the investigation: rc=0, an 8 MiB query, and a real acquired pointer
# (0xd9000000) inside the QuRT kernel's own reported TCM_PHYSPOOL range.
#
# ==========================================================================
# WHAT A SIMULATOR RUN OF THIS .SO DOES NOT PROVE -- READ THIS FIRST.
#
# simhost.c (compiled into this .so) still calls hexlib_iface_open/_start/
# _mmap/_invoke/_stop/_close as PLAIN C FUNCTIONS, bound directly to skel.c's
# definitions -- both are compiled into the SAME .so, so this is still an
# ordinary intra-module call, not a qaic-marshalled one. The qaic-generated
# stub (hexlib_iface_stub.c) is DELIBERATELY NOT ONE OF THE SOURCES LINKED
# HERE, for the identical reason as before: it defines the exact same
# function names as skel.c's DSP-side implementation (confirmed by running
# qaic and reading both generated files back), so linking both into one
# module is a duplicate-symbol error, not merely redundant. Packaging the
# host as a shared object rather than a standalone executable does not
# change this -- it changes HOW VTCM's own weak symbols get resolved (now
# dynamically, against the host process, at dlopen time), not whether the
# qaic stub is linked (it still is not).
#
# CONSEQUENCE: a simulator run through this .so exercises hexlib's OWN code
# -- batch parsing, the buffer table, the kernel dispatch table, kernel
# correctness, PCYCLE accounting, and now VTCM acquisition -- but it does NOT
# exercise qaic's argument marshaling/demarshaling at all. That is a real gap
# against this project's own design spec, which describes the simulator path
# as exercising "a qaic stub/skel invoke": what actually happens is a plain
# function call, and the marshaling layer is completely bypassed. Marshaling
# is only exercised on a real device, where the stub and skel genuinely live
# in separate processes and the call cannot avoid the wire.
# ==========================================================================

SIM_SO_LINK_FLAGS = [
    "-G0",
    "-Wl,--defsym=ISDB_TRUSTED_FLAG=2",
    "-Wl,--defsym=ISDB_SECURE_FLAG=2",
    "-Wl,--no-threads",
    "-fpic",
    "-shared",
    "-Wl,-Bsymbolic",
    "-lc",
]

# SIM_V_ARCH: the SDK's OWN per-arch simulator revision string, recovered
# verbatim from build/make.d.ext/hexagon/defines_hexagon_1_9.min -- NOT a
# mechanical "<arch>na_1" suffix rule (v68 and v81 use different suffixes
# entirely), so this is a lookup, not a format string.
_SIM_V_ARCH = {
    "v68": "v68n_1024",
    "v69": "v69na",
    "v73": "v73na_1",
    "v75": "v75na_1",
    "v79": "v79na_1",
    "v81": "v81qa_1",
}


def sim_v_arch(arch: str = tc.DSP_ARCH) -> str:
    return _SIM_V_ARCH.get(arch, arch)


def _run_main_on_hexagon_dir(sdk_root: str, arch: str = tc.DSP_ARCH) -> str:
    return os.path.join(
        sdk_root, "libs", "run_main_on_hexagon", "ship",
        f"hexagon_toolv19_{arch}",
    )


def run_main_on_hexagon_sim_path(sdk_root: str, arch: str = tc.DSP_ARCH) -> str:
    """The SDK's OWN prebuilt QuRT-hosted launcher. Referenced by path from
    the SDK, never copied into this repo (the SDK is license-restricted)."""
    return os.path.join(_run_main_on_hexagon_dir(sdk_root, arch),
                        "run_main_on_hexagon_sim")


def runelf_pbn_path(sdk_root: str, arch: str = tc.DSP_ARCH) -> str:
    return os.path.join(sdk_root, "rtos", "qurt", f"compute{arch}",
                        "sdksim_bin", "runelf.pbn")


def build_sim_so(out_dir: str, sdk_root: str | None = None) -> str:
    """Link the simulator host + skel into a QuRT-hosted shared object.

    Recovered from the SDK's own `libs/run_main_on_hexagon` example's
    `test_main_so_link.txt` (see the module-level comment above this
    function for the full recipe provenance).
    """
    root = sdk_root or tc.default_sdk_root()
    bin_dir = tc.find_toolchain_bin(root)
    env = tc.toolchain_env(bin_dir)
    compiler = os.path.join(bin_dir, tc.exe(tc.COMPILER))
    gen = os.path.join(out_dir, "gen")

    lib = os.path.join(out_dir, "libhexlib_skel.a")
    if not os.path.isfile(lib):
        raise RuntimeBuildError(f"build_skel_lib must run first: {lib} missing")

    repo = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    simhost_c = os.path.join(repo, "hexlib", "runtime", "simhost", "simhost.c")

    so = os.path.join(out_dir, "hexlib_sim.so")
    cmd = [compiler] + tc.cflags_for_caps(["hvx"]) + SIM_SO_LINK_FLAGS
    for d in runtime_include_dirs(root, gen):
        cmd.append(f"-I{d}")
    cmd += [
        "-Wl,-Map=" + so + ".map",
        "-Wl,-soname=" + os.path.basename(so),
        "-o", so,
        "-Wl,--start-group", simhost_c, lib, "-Wl,--end-group",
    ]

    rc, out, err, to = tc.run(cmd, env, timeout=tc.SIM_TIMEOUT_S)
    if to or rc != 0 or not os.path.isfile(so):
        raise RuntimeBuildError("linking hexlib_sim.so failed", (out + err).strip())
    return so


def write_qurt_sim_configs(out_dir: str, sdk_root: str | None = None,
                           tools_root: str | None = None,
                           arch: str = tc.DSP_ARCH) -> tuple[str, str]:
    """Write osam.cfg and q6ss.cfg into `out_dir` (never the source tree).

    Both filenames are matched BY NAME in `.gitignore` (`osam.cfg`, `q6ss.cfg`
    -- not a blanket `*.cfg`, so a config someone actually means to commit
    elsewhere is not silently swallowed). That rule is what keeps these out
    of the repo; `out_dir` living inside a pytest `tmp_path` today is
    incidental, not the actual protection -- a caller is free to pass a work
    dir INSIDE the repo, and `.gitignore` is what stops the resulting files
    from being committed, not where the caller happened to point `out_dir`.

    NOT VENDORING: each file is one or two lines naming an existing SDK
    artifact BY PATH (the QuRT debugger model, and two cosim timer/interrupt
    controller shims); no SDK file's bytes are copied. Recovered verbatim
    from rtos/qurt/qurt_libs_priv.min's own $(OBJ_DIR)/osam.cfg and
    $(OBJ_DIR)/q6ss.cfg rules (Windows branch: this project only targets
    Windows per toolchain.py/CLAUDE.md conventions already established
    elsewhere in this codebase).
    """
    root = sdk_root or tc.default_sdk_root()
    bin_dir = tc.find_toolchain_bin(root)
    tools = tools_root or os.path.dirname(os.path.dirname(bin_dir))
    os.makedirs(out_dir, exist_ok=True)

    is_arm64 = os.environ.get("PROCESSOR_ARCHITEW6432") == "ARM64"
    debugger_dir = "Win_arm64" if is_arm64 else "Win"
    qurt_model = os.path.join(root, "rtos", "qurt", f"compute{arch}",
                              "debugger", debugger_dir, "qurt_model.dll")
    if not os.path.isfile(qurt_model):
        raise RuntimeBuildError(f"QuRT debugger model not found: {qurt_model}")
    osam_cfg = os.path.join(out_dir, "osam.cfg")
    with open(osam_cfg, "w") as f:
        f.write(qurt_model + "\n")

    iss_dir = os.path.join(tools, "Tools", "lib", "iss")
    qtimer = os.path.join(iss_dir, "qtimer.dll")
    l2vic = os.path.join(iss_dir, "l2vic.dll")
    for f in (qtimer, l2vic):
        if not os.path.isfile(f):
            raise RuntimeBuildError(f"simulator cosim module not found: {f}")
    q6ss_cfg = os.path.join(out_dir, "q6ss.cfg")
    with open(q6ss_cfg, "w") as f:
        f.write(f"{qtimer} --csr_base=0xFC900000 --irq_p=3 --freq=19200000 --cnttid=1\n")
        f.write(f"{l2vic} 32 0xFC910000\n")

    return osam_cfg, q6ss_cfg


def sim_qurt_command(out_dir: str, so_path: str, sdk_root: str | None = None,
                     extra_args: tuple[str, ...] = (),
                     arch: str = tc.DSP_ARCH) -> list[str]:
    """Assemble the full hexagon-sim invocation that boots a real QuRT
    kernel, then dlopen's `so_path` (which must already sit inside
    `out_dir`) and calls its main(argc, argv) with `extra_args`.

    Recovered from the SDK's own `libs/run_main_on_hexagon` example's
    `sim_cmd_line.txt` at v75/toolv19 -- the exact shape its own
    QURT_QEXE_EXEC/QEXE_EXEC make rules produce (rtos/qurt/
    qurt_libs_priv.min), not reconstructed.

    Pure with respect to the filesystem except for locating hexagon-sim and
    the two prebuilt SDK artifacts by path -- assertable without running it,
    like sim.py's own sim_command().

    CALLER MUST SET THE SUBPROCESS cwd TO `out_dir` -- CONFIRMED EMPIRICALLY,
    NOT A GUESS. Under this QuRT-hosted launch, relative fopen() READS inside
    the dlopen'd .so (hexlib_batch.bin, hexlib_in.bin) resolve through
    `--usefs out_dir` correctly, but relative fopen(..., "wb") WRITES
    (hexlib_rsp.bin, hexlib_out.bin) land in the LAUNCHING PROCESS'S OWN
    working directory instead -- QuRT's own POSIX filesystem layer does not
    consult --usefs the same way for file creation. This function cannot fix
    that from here (it only assembles argv; it does not spawn the process),
    so whatever runs this command (e.g. `subprocess.run(cmd, cwd=out_dir,
    ...)`, or `tc.run` -- which does not itself take a cwd -- called from a
    caller that has already os.chdir'd or otherwise pinned its own cwd to
    `out_dir`) MUST set the subprocess's cwd to this same `out_dir`, or the
    response/output files will not be where the batch/input files were
    written. See hexlib/runtime/simhost/simhost.c's own header comment for
    the full empirical writeup.
    """
    root = sdk_root or tc.default_sdk_root()
    bin_dir = tc.find_toolchain_bin(root)
    sim_exe = os.path.join(bin_dir, tc.exe("hexagon-sim"))

    osam_cfg = os.path.join(out_dir, "osam.cfg")
    q6ss_cfg = os.path.join(out_dir, "q6ss.cfg")
    run_main = run_main_on_hexagon_sim_path(root, arch)
    runelf = runelf_pbn_path(root, arch)

    cmd = [
        sim_exe, f"-m{sim_v_arch(arch)}", "--simulated_returnval",
        "--usefs", out_dir,
        "--pmu_statsfile", os.path.join(out_dir, "pmu_stats.txt"),
        "--cosim_file", q6ss_cfg,
        "--l2tcm_base", "0xd800",
        "--rtos", osam_cfg,
        runelf, "--",
        run_main, "--",
        os.path.basename(so_path),
    ]
    cmd += list(extra_args)
    return cmd


# ============================================================================
# STAGE 2 GATE — the aarch64 CPU side and the device skel .so. Built, never
# run: no phone is available, so this proves only "it builds and is the right
# machine type" (aarch64 for hexlib_run, Hexagon for libhexlib_skel.so).
# Stage 3 is the first thing that ever executes either artifact.
#
# THE STUB/SKEL SPLIT INVERTS FROM THE SIMULATOR, AND THIS IS THE CRUX OF THIS
# TASK. build_sim_so (above) deliberately never links hexlib_iface_stub.c,
# because it defines the exact same symbol names as skel.c and both would
# have to live in one process/module. On a device they are two SEPARATE
# binaries, so the arrangement inverts:
#   - hexlib_run (aarch64) links the qaic-generated STUB
#     (hexlib_iface_stub.c) plus hexlib/runtime/host/*.c, against -ldl -llog.
#     Calling hexlib_iface_invoke() from it marshals arguments into a
#     remote_arg[] and calls remote_handle64_invoke() for real (see
#     host/main.c's own header comment).
#   - libhexlib_skel.so (Hexagon, -shared -fPIC) contains the qaic SKEL
#     (hexlib_iface_skel.c) plus skel.c/skel_bufs.c/skel_vtcm.c/
#     skel_dispatch.c, the generated per-kernel entry points, and the
#     kernels themselves -- loaded by the FastRPC framework on the DSP,
#     which calls INTO it, never the reverse.
# Get this backwards (stub in the skel .so, or skel code in the aarch64
# binary) and the result is either a duplicate-symbol link error or a binary
# that can never marshal at all. THE DEVICE PATH IS THE ONLY PATH IN THIS
# PROJECT THAT WILL EVER EXERCISE QAIC'S REAL MARSHALLING — every simulator
# run before this task (build_sim_so, above) calls skel.c as plain C
# functions in one address space; see host/main.c's own file header for the
# same point made from the other side of the wire.
# ============================================================================


def ndk_bin_dir(sdk_root: str) -> str:
    """The NDK's prebuilt toolchain `bin` directory for the CURRENT host OS.
    VERIFIED against the actual installed SDK (not assumed): only a
    `windows-x86_64` prebuilt tree exists there, so that is the only
    non-Windows-host branch this repo could ever actually exercise, but the
    `linux-x86_64` name is the NDK's own documented convention, kept for a
    contributor on a different host."""
    host_tag = "windows-x86_64" if os.name == "nt" else "linux-x86_64"
    return os.path.join(tc.ndk_root(sdk_root), "toolchains", "llvm", "prebuilt",
                        host_tag, "bin")


def ndk_clang(sdk_root: str | None = None) -> str:
    """Path to the NDK's aarch64 Android clang driver, pinned to tc.ANDROID_API.

    ON WINDOWS THIS MUST BE THE `.cmd` FORM, NOT THE BARE NAME — VERIFIED, NOT
    GUESSED. The bare `aarch64-linux-android<API>-clang` next to it is a Bourne
    shell script (confirmed with `file`), which a plain `subprocess.run([...])`
    call (no shell) cannot execute on Windows at all; `subprocess.run` against
    the `.cmd` wrapper was confirmed to actually run and print a real clang
    version banner. `tc.run` (toolchain.py) never sets `shell=True`, so the
    bare name would fail with "cannot execute" on every Windows caller.
    """
    root = sdk_root or tc.default_sdk_root()
    bin_dir = ndk_bin_dir(root)
    name = f"aarch64-linux-android{tc.ANDROID_API}-clang"
    if os.name == "nt":
        name += ".cmd"
    return os.path.join(bin_dir, name)


# Recovered from the SDK's OWN shared-library link recipe for this exact
# toolchain version -- $SDK/build/make.d.ext/hexagon/defines_hexagon_1_9.min's
# `DLL_LD_FLAGS` (1_9 is the "hexagon_toolv19" family TOOLCHAIN_VERSION 19.0.04
# belongs to) -- NOT SIM_SO_LINK_FLAGS above, which is a DIFFERENT recipe for a
# different situation. SIM_SO_LINK_FLAGS builds a .so meant to be dlopen'd into
# an ALREADY-RUNNING QuRT host process (run_main_on_hexagon_sim) that already
# has its own allocator; DLL_LD_FLAGS is the SDK's general-purpose "this is a
# Hexagon shared library" recipe, and it carries five `--wrap=` flags
# SIM_SO_LINK_FLAGS does not: `malloc`/`calloc`/`free`/`realloc`/`memalign`,
# the PD heap-interposition every real Hexagon DLL gets so its allocations are
# routed through the loading process's own signed/unsigned-PD heap manager
# rather than a bare libc allocator. A real FastRPC-loaded skel needs that;
# the sim .so does not (it never leaves the one host process it was dlopen'd
# into). `-Wl,--no-undefined`/`-z defs` is deliberately absent, exactly as in
# the SDK's own recipe: symbols like HAP_mmap2 and the HAP_compute_res_*
# family are resolved dynamically, at dlopen time, against the framework
# already running in the DSP process the skel loads into -- never statically
# linked here, on device OR on the simulator (see sim_shims.c's own header for
# the simulator side of that same fact).
DEVICE_SKEL_LINK_FLAGS = [
    "-G0",
    "-Wl,--defsym=ISDB_TRUSTED_FLAG=2",
    "-Wl,--defsym=ISDB_SECURE_FLAG=2",
    "-Wl,--no-threads",
    "-fpic",
    "-shared",
    "-Wl,-Bsymbolic",
    "-Wl,--wrap=malloc",
    "-Wl,--wrap=calloc",
    "-Wl,--wrap=free",
    "-Wl,--wrap=realloc",
    "-Wl,--wrap=memalign",
    "-lc",
]


def _build_device_skel_so(out_dir: str, root: str, gen: str, qa: QaicOutput) -> str:
    """Compile the skel + kernels + generated entries into a real Hexagon
    SHARED OBJECT (`libhexlib_skel.so`), the device counterpart of
    `build_skel_lib`'s `.a` above. Deliberately NOT a thin wrapper around
    `build_skel_lib` -- the object sets genuinely differ (see below), and
    `build_skel_lib`'s own `-fpic` insertion is asserted, by literal source
    text, by `test_runtime_sim_build.py::
    test_build_skel_lib_compiles_position_independent_code`; reshaping that
    function to share code with this one is out of scope for this task and
    risks that assertion for no benefit, since the compiled objects are not
    even byte-identical between the two paths (see next paragraph).

    `sim_shims.c` IS DELIBERATELY NOT COMPILED IN HERE. It exists only to
    backfill `HAP_mmap2`/`HAP_munmap2` on top of `test_util.a`'s
    simulator-only, int-length `HAP_mmap` (see `simhost/sim_shims.c`'s own
    header) -- this build never links `test_util.a` at all, and a real
    device's FastRPC framework resolves `HAP_mmap2` for real, dynamically,
    inside the signed/unsigned PD process this .so is loaded into.
    """
    from hexlib.build import compile_command
    from hexlib.exec.runner import SPECS
    from hexlib.runtime import genentry

    bin_dir = tc.find_toolchain_bin(root)
    version = tc.toolchain_version(bin_dir)
    if version != tc.TOOLCHAIN_VERSION:
        raise RuntimeBuildError(
            f"toolchain is {version}, expected {tc.TOOLCHAIN_VERSION} — cycle "
            "numbers are not comparable across toolchain versions"
        )
    env = tc.toolchain_env(bin_dir)
    compiler = os.path.join(bin_dir, tc.exe(tc.COMPILER))

    repo = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    entries = genentry.generate(repo, gen)

    used_kernels = sorted({
        os.path.basename(spec.kernel_dir)
        for spec in SPECS.values()
        if os.path.isdir(os.path.join(repo, spec.kernel_dir))
    })

    skel_dir = os.path.join(repo, "hexlib", "runtime", "skel")
    # "dev_"-prefixed object basenames: this compiles into the SAME out_dir
    # build_skel_lib/build_sim_so may also use for a sim artifact, and their
    # object names (skel.o, skel_bufs.o, ...) would otherwise collide on disk
    # with these PIC-but-differently-sourced objects (no sim_shims.o here at
    # all -- seeded from a genuinely different source-file set, not just a
    # different flag).
    base = [
        (qa.skel, "dev_hexlib_iface_skel.o", []),
        (os.path.join(skel_dir, "skel.c"), "dev_skel.o", []),
        (os.path.join(skel_dir, "skel_bufs.c"), "dev_skel_bufs.o", []),
        (os.path.join(skel_dir, "skel_vtcm.c"), "dev_skel_vtcm.o", []),
        (os.path.join(skel_dir, "skel_dispatch.c"), "dev_skel_dispatch.o", []),
    ]
    for e in entries:
        stem = os.path.basename(e)[:-len(".c")]
        if stem.endswith("_entry"):
            k = stem[: -len("_entry")]
            base.append((e, f"dev_{stem}.o", [os.path.join(repo, "kernels", k)]))
        else:
            base.append((e, f"dev_{stem}.o", []))  # hexlib_kernel_table.c
    for k in used_kernels:
        kdir = os.path.join(repo, "kernels", k)
        base.append((os.path.join(kdir, "kernel.c"), f"dev_{k}_kernel.o", [kdir]))
        hand = os.path.join(kdir, "dsp_entry.c")
        if os.path.isfile(hand):
            base.append((hand, f"dev_{k}_dsp_entry.o", [kdir]))

    common_includes = runtime_include_dirs(root, gen)

    objs = []
    for s, obj_name, extra in base:
        o = os.path.join(out_dir, obj_name)
        cmd = compile_command(
            compiler, [s], o, ["hvx"], common_includes + extra, compile_only=True
        )
        cmd.insert(1, "-fpic")  # required for a -shared link, same as build_skel_lib.
        rc, out, err, to = tc.run(cmd, env, timeout=tc.SIM_TIMEOUT_S)
        if to or rc != 0:
            raise RuntimeBuildError(f"device skel compile failed: {s}", (out + err).strip())
        objs.append(o)

    # LIB_HEXAGON, from the SAME defines_hexagon_1_9.min recipe DEVICE_SKEL_LINK_FLAGS
    # is recovered from: "$(HEXAGON_LIB_DIR)/$(V_ARCH)/G0/libhexagon.a", and its
    # own comment notes the linker only pulls symbols from it if something else
    # in the link needs them -- so including it unconditionally is what the
    # SDK's own recipe does, not an addition of convenience.
    tools_root = os.path.dirname(os.path.dirname(bin_dir))
    lib_hexagon = os.path.join(tools_root, "Tools", "target", "hexagon", "lib",
                               tc.DSP_ARCH, "G0", "libhexagon.a")
    if not os.path.isfile(lib_hexagon):
        raise RuntimeBuildError(f"libhexagon.a not found: {lib_hexagon}")

    so = os.path.join(out_dir, "libhexlib_skel.so")
    cmd = [compiler] + tc.cflags_for_caps(["hvx"]) + DEVICE_SKEL_LINK_FLAGS
    cmd += [
        "-Wl,-Map=" + so + ".map",
        "-Wl,-soname=" + os.path.basename(so),
        "-o", so,
        "-Wl,--start-group",
    ] + objs + [lib_hexagon, "-Wl,--end-group"]

    rc, out, err, to = tc.run(cmd, env, timeout=tc.SIM_TIMEOUT_S)
    if to or rc != 0 or not os.path.isfile(so):
        raise RuntimeBuildError("linking libhexlib_skel.so failed", (out + err).strip())
    return so


def build_device_binary(out_dir: str, sdk_root: str | None = None) -> str:
    """Cross-compile `hexlib_run` for Android aarch64, and (as a side effect)
    `libhexlib_skel.so` for the Hexagon device, both into `out_dir`. Returns
    the path to `hexlib_run`.

    NEITHER ARTIFACT IS EVER RUN HERE — no device is available (see the
    module-level "STAGE 2 GATE" comment above). This function's entire job is
    "it builds, and it is the right machine type"; stage 3 is the first thing
    that ever executes either one.
    """
    root = sdk_root or tc.default_sdk_root()
    os.makedirs(out_dir, exist_ok=True)
    gen = os.path.join(out_dir, "gen")
    os.makedirs(gen, exist_ok=True)

    repo = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    idl = os.path.join(repo, "hexlib", "runtime", "idl", "hexlib_iface.idl")
    qa = run_qaic(idl, gen, root)

    # The Hexagon side, built first: a failure here (e.g. a bad SDK path)
    # surfaces before any aarch64 work is wasted.
    _build_device_skel_so(out_dir, root, gen, qa)

    # The aarch64 side: the qaic STUB (never the skel) plus every host/*.c
    # file (Task 9). See the module-level comment for why this is the
    # opposite arrangement from the Hexagon side.
    clang = ndk_clang(root)
    if not os.path.isfile(clang):
        raise RuntimeBuildError(f"NDK clang not found: {clang}")

    host_dir = os.path.join(repo, "hexlib", "runtime", "host")
    skel_dir = os.path.join(repo, "hexlib", "runtime", "skel")
    sources = [
        qa.stub,
        os.path.join(host_dir, "driver.c"),
        os.path.join(host_dir, "session.c"),
        os.path.join(host_dir, "buffers.c"),
        os.path.join(host_dir, "main.c"),
    ]
    # `$SDK/incs` (remote.h, AEEStdDef.h -- needed by the qaic-generated
    # header and stub), `$SDK/incs/stddef`, `$SDK/ipc/fastrpc/rpcmem/inc`
    # (rpcmem.h, buffers.c), `gen` (hexlib_iface.h), `host_dir`
    # (hexlib_host.h), and `skel_dir` (hexlib_dsp.h -- session.c/buffers.c
    # both include it for the wire structs, purely as a header; no skel .c
    # file is compiled into this binary).
    includes = [
        gen,
        host_dir,
        skel_dir,
        os.path.join(root, "incs"),
        os.path.join(root, "incs", "stddef"),
        os.path.join(root, "ipc", "fastrpc", "rpcmem", "inc"),
    ]

    # NO libcdsprpc.so IMPORT LIBRARY HERE -- DELIBERATELY. The qaic-generated
    # stub (hexlib_iface_stub.c, never hand-edited) calls
    # remote_handle64_open/_invoke/_close directly, as ordinary strong
    # `extern` functions declared in <remote.h> (confirmed: no `weak`
    # attribute there, __QAIC_REMOTE(ff) defaults to identity). The first
    # version of this function linked against the SDK's own aarch64 import
    # stub (ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so) to satisfy
    # that -- it linked, but it reintroduced the exact failure mode
    # driver.c's own "WHY DLOPEN AND NOT A LINK-TIME DEPENDENCY" comment
    # exists to avoid: a device missing libcdsprpc.so would fail to even
    # start hexlib_run (a dynamic-linker load error, before main() runs),
    # never reaching hexlib_drv_init()'s readable message at all. Fixed at
    # the source instead: driver.c now DEFINES remote_handle64_open/_invoke/
    # _close itself, as thin forwarders to the hexlib_remote_handle64_*
    # function pointers it already dlsym's -- see driver.c's own comment on
    # them. That satisfies the stub's link-time reference without ever
    # linking libcdsprpc.so at build time, so the driver stays exclusively
    # dlopen'd, exactly as designed.
    exe = os.path.join(out_dir, "hexlib_run")
    # -Wall -Werror for the SAME reason tc.HVX_CFLAGS carries them (see the long
    # comment there): every caller of tc.run checks only `rc != 0`, so a warning
    # is emitted and discarded. This side of the wire assembles the batch blob
    # and the buffer table by hand in host/*.c, which is exactly the kind of
    # code where a pointer/qualifier diagnostic is the only automatic notice
    # that two things were swapped. MEASURED before enabling: this link
    # produces ZERO warnings under -Wall on the pinned NDK (r25c, API 33), so
    # nothing is being grandfathered in. NOT tc.HVX_CFLAGS itself -- that list
    # is Hexagon-specific (-mv75/-mhvx) and means nothing to an aarch64 clang.
    cmd = [clang, "-O2", "-Wall", "-Werror"]
    for d in includes:
        cmd.append(f"-I{d}")
    cmd += sources
    cmd += ["-o", exe, "-ldl", "-llog"]

    rc, out, err, to = tc.run(cmd, os.environ.copy(), timeout=tc.SIM_TIMEOUT_S)
    if to or rc != 0 or not os.path.isfile(exe):
        raise RuntimeBuildError("linking hexlib_run failed", (out + err).strip())
    return exe

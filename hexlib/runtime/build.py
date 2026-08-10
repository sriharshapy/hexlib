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

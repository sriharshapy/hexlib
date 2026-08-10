# hexlib/tests/test_runtime_sim_build.py
"""Task 7b: the QuRT-hosted simulator build that actually reaches
hexlib_iface_start.

Task 7's standalone `--force-dynamic` qexe (build_sim_qexe, gone now) built
and linked and reached hexlib_iface_open, but hexlib_iface_start could NEVER
succeed there: VTCM acquisition needs real QuRT thread/clock primitives a
standalone qexe cannot provide (see
.superpowers/sdd/2026-08-10-silicon-path-runtime/
investigation-sim-vtcm-and-marshalling.md). build_sim_so below produces a
shared object dlopen'd by the SDK's own prebuilt run_main_on_hexagon_sim
under a real booted QuRT kernel instead -- the load-bearing test here,
test_sim_run_reaches_start_and_reports_real_vtcm, is the one that proves the
fix actually works, not merely that a function returned a path string.
"""
import inspect
import os
import re
import struct
import subprocess

import pytest

import hexlib
from hexlib import toolchain as tc
from hexlib.runtime import build as rb
from hexlib.runtime import wire

HAS_SDK = os.path.isdir(tc.default_sdk_root())
sdk = pytest.mark.skipif(not HAS_SDK, reason="Hexagon SDK not present")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(hexlib.__file__)))


# ============================================================================
# Pure tests: no SDK filesystem access, assertable anywhere.
# ============================================================================


def test_sim_so_link_flags_are_pic_shared_not_a_standalone_exe():
    """SIM_SO_LINK_FLAGS is a shared-object recipe, recovered from the SDK's
    own libs/run_main_on_hexagon example's test_main_so_link.txt -- NOT the
    retired standalone-qexe recipe. A regression back to the old
    --force-dynamic exe shape would defeat the whole point of this task (it
    is exactly the shape that can never acquire VTCM), so this also asserts
    the old flags are ABSENT, not just that the new ones are present."""
    flags = rb.SIM_SO_LINK_FLAGS
    assert "-fpic" in flags
    assert "-shared" in flags
    assert "-Wl,-Bsymbolic" in flags
    assert "-lc" in flags
    joined = " ".join(flags)
    assert "--force-dynamic" not in joined, (
        "this is the standalone-qexe flag that makes VTCM acquisition "
        "impossible -- it must not reappear on the .so recipe"
    )
    assert "--dynamic-linker=" not in joined


def test_build_skel_lib_compiles_position_independent_code():
    """Every object build_skel_lib compiles must carry -fpic, because the
    only thing this archive is ever linked into is build_sim_so's shared
    object -- non-PIC objects in a `-shared` link either fail to link outright
    or (worse, silently) produce a .so with text-relocations a real device
    loader would refuse. Reading the function's own source (not merely
    grepping the whole file, which would also match this test module's own
    docstring if it discussed -fpic) so a `-fpic` mentioned only in a comment,
    with the actual compile call unchanged, would NOT be enough to pass this."""
    src = inspect.getsource(rb.build_skel_lib)
    # Line-by-line, and only lines that do not START (after stripping
    # leading whitespace) with a comment marker -- a commented-out call
    # still contains this exact substring, so a plain `re.search` over the
    # whole function body would NOT catch that regression.
    live_lines = [
        ln for ln in src.splitlines() if not ln.strip().startswith("#")
    ]
    assert any(
        re.search(r'cmd\.insert\(\s*1\s*,\s*"-fpic"\s*\)', ln) for ln in live_lines
    ), "build_skel_lib no longer inserts -fpic into every compile command"


def test_sim_v_arch_is_the_sdks_own_lookup_not_a_mechanical_suffix():
    """Recovered from build/make.d.ext/hexagon/defines_hexagon_1_9.min's
    SIM_V_ARCH table. Not reconstructable as "<arch>na_1" -- v68 and v81 use
    entirely different suffixes -- so this pins the exact table, not a
    plausible-looking formula that would happen to pass for v75 alone."""
    assert rb.sim_v_arch("v75") == "v75na_1"
    assert rb.sim_v_arch("v68") == "v68n_1024"
    assert rb.sim_v_arch("v81") == "v81qa_1"


def test_run_main_on_hexagon_sim_path_is_sdk_relative_never_vendored():
    p = rb.run_main_on_hexagon_sim_path("/sdk", "v75")
    joined = p.replace("\\", "/")
    assert joined == "/sdk/libs/run_main_on_hexagon/ship/hexagon_toolv19_v75/run_main_on_hexagon_sim"


def test_runelf_pbn_path_is_sdk_relative_never_vendored():
    p = rb.runelf_pbn_path("/sdk", "v75")
    joined = p.replace("\\", "/")
    assert joined == "/sdk/rtos/qurt/computev75/sdksim_bin/runelf.pbn"


def test_generated_sim_configs_are_actually_gitignored_by_name():
    """FIX 1 (coordinator review, task 7b): write_qurt_sim_configs's own
    docstring used to claim osam.cfg/q6ss.cfg were "already git-ignored via
    the same rules that already ignore *.o/*.a/*.so" -- no rule actually
    matched *.cfg, so that was a stated guarantee with nothing behind it: a
    caller pointing out_dir INSIDE the repo would have committed SDK-derived
    config files, only avoided today by pytest's tmp_path happening to sit
    outside the repo.

    Asks git DIRECTLY (`git check-ignore`) rather than re-parsing
    `.gitignore` in Python, so this cannot drift from what git itself will
    actually do -- reimplementing gitignore's own matching logic would risk
    the test and the real behavior silently disagreeing.

    Uses a path nested under a directory ("_no_such_dir") that appears
    nowhere else in `.gitignore`, and a same-directory sibling with a
    different extension as a negative control, so a pass here can only be
    explained by a rule that names these two files specifically -- not a
    blanket `*.cfg`, and not some unrelated existing rule (e.g. `_work/`)
    catching it by accident.
    """
    for name in ("osam.cfg", "q6ss.cfg"):
        rel = os.path.join("hexlib", "runtime", "_no_such_dir", name)
        rc = subprocess.run(
            ["git", "check-ignore", "-q", rel], cwd=REPO_ROOT
        ).returncode
        assert rc == 0, f"{rel!r} is not matched by .gitignore (rc={rc})"

    # Negative control: a same-shaped path that must NOT be ignored, proving
    # the match above is specific to these two filenames.
    control = os.path.join("hexlib", "runtime", "_no_such_dir", "not_a_generated_sim_config.cfg")
    rc = subprocess.run(["git", "check-ignore", "-q", control], cwd=REPO_ROOT).returncode
    assert rc != 0, (
        f"{control!r} is unexpectedly git-ignored -- the rule added for "
        "osam.cfg/q6ss.cfg may have been written as a blanket *.cfg instead "
        "of the tighter by-name pattern that was asked for"
    )


# ============================================================================
# SDK-gated: actually build and run.
# ============================================================================


@sdk
def test_skel_library_builds(tmp_path):
    lib = rb.build_skel_lib(["scale_fp16"], str(tmp_path))
    assert os.path.isfile(lib)
    assert os.path.getsize(lib) > 0
    # An ar archive, not merely a path that happens to exist: the magic bytes
    # a mock or a `touch` would not reproduce.
    with open(lib, "rb") as f:
        assert f.read(8) == b"!<arch>\n"


@sdk
def test_sim_so_builds_a_real_hexagon_shared_object(tmp_path):
    rb.build_skel_lib(["scale_fp16"], str(tmp_path))
    so = rb.build_sim_so(str(tmp_path))
    assert os.path.isfile(so)
    assert os.path.getsize(so) > 0
    with open(so, "rb") as f:
        header = f.read(20)
    assert header[:4] == b"\x7fELF"
    e_type = header[16] | (header[17] << 8)
    e_machine = header[18] | (header[19] << 8)
    # ET_DYN (3), not ET_EXEC (2): a build that regressed back to a
    # standalone --force-dynamic executable would still be a valid Hexagon
    # ELF and would still pass a bare "is this an ELF" check, but it could
    # never be dlopen'd by run_main_on_hexagon_sim, which is the entire
    # mechanism this task depends on.
    assert e_type == 3, f"expected ET_DYN (shared object), got e_type={e_type}"
    assert e_machine == 0xA4, f"expected EM_HEXAGON, got e_machine={e_machine}"


@sdk
def test_qurt_sim_configs_reference_real_sdk_artifacts_never_vendored(tmp_path):
    """osam.cfg/q6ss.cfg must land in the build OUTPUT directory (never the
    source tree), and must each be a thin reference to an SDK file that
    genuinely exists on disk -- not a copy of its bytes."""
    osam_cfg, q6ss_cfg = rb.write_qurt_sim_configs(str(tmp_path))
    assert os.path.dirname(osam_cfg) == str(tmp_path)
    assert os.path.dirname(q6ss_cfg) == str(tmp_path)

    osam_target = open(osam_cfg).read().strip()
    assert os.path.isfile(osam_target), (
        f"osam.cfg names {osam_target!r}, which does not exist -- a dangling "
        "reference is as bad as a vendored copy that went stale"
    )
    assert osam_target.endswith("qurt_model.dll")

    q6ss_lines = open(q6ss_cfg).read().splitlines()
    assert len(q6ss_lines) == 2
    for line in q6ss_lines:
        path = line.split()[0]
        assert os.path.isfile(path), f"q6ss.cfg names {path!r}, which does not exist"
    assert q6ss_lines[0].endswith(
        "qtimer.dll --csr_base=0xFC900000 --irq_p=3 --freq=19200000 --cnttid=1"
    )
    assert q6ss_lines[1].endswith("l2vic.dll 32 0xFC910000")


@sdk
def test_sim_qurt_command_shape(tmp_path):
    """Pure assembly, but needs the SDK to locate hexagon-sim -- checked
    ordering matters: runelf.pbn loads run_main_on_hexagon_sim, which then
    dlopen's the .so by (bare) name, and extra_args land after it."""
    so = str(tmp_path / "hexlib_sim.so")
    cmd = rb.sim_qurt_command(str(tmp_path), so, extra_args=("--unmapped",))

    def idx(needle):
        for i, c in enumerate(cmd):
            if needle in c:
                return i
        raise AssertionError(f"{needle!r} not found in {cmd}")

    i_runelf = idx("runelf.pbn")
    i_run_main = idx("run_main_on_hexagon_sim")
    i_so = cmd.index("hexlib_sim.so")
    assert i_runelf < i_run_main < i_so, (
        "runelf.pbn must load run_main_on_hexagon_sim, which must dlopen the "
        ".so, in that order"
    )
    assert cmd[-1] == "--unmapped", "extra_args must come after the .so name"
    assert "--usefs" in cmd and cmd[cmd.index("--usefs") + 1] == str(tmp_path)
    assert "--rtos" in cmd
    assert "--cosim_file" in cmd
    assert "-mv75na_1" in cmd


@sdk
def test_sim_run_reaches_start_and_reports_real_vtcm(tmp_path):
    """THE LOAD-BEARING TEST. Actually boots a real QuRT kernel under
    hexagon-sim, dlopen's hexlib's own .so, and checks the SIMHOST output --
    not merely that build functions returned paths. A skel that still failed
    to acquire VTCM (the exact defect this task exists to fix) would print
    `SIMHOST error=start`, never reach the hwinfo line, and this test would
    fail; a start that reported the wrong arch or the wrong VTCM size would
    fail the regex match below even if `start` itself returned 0."""
    out_dir = str(tmp_path)
    rb.build_skel_lib(["scale_fp16"], out_dir)
    so = rb.build_sim_so(out_dir)
    rb.write_qurt_sim_configs(out_dir)

    # An empty-but-well-formed batch: 0 bufs, 0 tensors, 0 ops. This is
    # enough to drive open -> start -> hwinfo -> invoke -> stop -> close all
    # the way through without needing rpcmem/fd plumbing, which is Task 8's
    # concern, not this one's.
    with open(os.path.join(out_dir, "hexlib_batch.bin"), "wb") as f:
        f.write(wire.pack_batch([], [], []))
    with open(os.path.join(out_dir, "hexlib_in.bin"), "wb") as f:
        f.write(b"")

    cmd = rb.sim_qurt_command(out_dir, so)
    bin_dir = tc.find_toolchain_bin(tc.default_sdk_root())
    env = tc.toolchain_env(bin_dir)
    rc, out, err, timed_out = tc.run(cmd, env, timeout=tc.SIM_TIMEOUT_MAX_S)
    combined = out + err

    assert not timed_out, combined
    assert "SIMHOST error=open" not in combined, combined
    assert "SIMHOST error=start" not in combined, combined

    m = re.search(r"SIMHOST hwinfo arch=(\d+) threads=(\d+) vtcm=(\d+)", combined)
    assert m, f"no SIMHOST hwinfo line recovered -- start did not succeed:\n{combined}"
    assert int(m.group(1)) == 75, f"expected arch=75, got {m.group(0)}"
    assert int(m.group(3)) == 8388608, f"expected vtcm=8388608, got {m.group(0)}"

    assert "SIMHOST invoke rc=0" in combined, combined
    assert "SIMHOST done" in combined, combined
    assert rc == 0, f"process exited {rc}:\n{combined}"


@sdk
def test_an_unmapped_fd_batch_is_still_refused_under_the_new_build(tmp_path):
    """The property task-7b was told to preserve: --unmapped must still
    DELIBERATELY SKIP hexlib_iface_mmap, and the skel must still refuse.
    Only a real op naming a real (registered) buffer can exercise the
    discriminator, so this builds one real "scale" op -- if the new
    QuRT-hosted packaging accidentally made mapping a no-op (e.g. by
    resolving the raw fd as an address the way a shared-address-space bug
    would), this invoke would return OK instead of ERR_UNMAPPED."""
    from hexlib.runtime.genentry import KIND_ID

    out_dir = str(tmp_path)
    rb.build_skel_lib(["scale_fp16"], out_dir)
    so = rb.build_sim_so(out_dir)
    rb.write_qurt_sim_configs(out_dir)

    n = 8
    bufs = [wire.BufDesc(fd=0, size=n * 2 * 2)]
    tensors = [
        wire.TensorDesc(bi=0, offset=0, nbytes=n * 2, dtype="fp16",
                        layout="row_major", ne=(n, 1, 1, 1)),
        wire.TensorDesc(bi=0, offset=n * 2, nbytes=n * 2, dtype="fp16",
                        layout="row_major", ne=(n, 1, 1, 1)),
    ]
    factor_bits = struct.unpack("<i", struct.pack("<f", 0.5))[0]
    ops = [wire.OpDesc(kind=KIND_ID["scale"], params=(factor_bits,),
                       src=(0,), dst=(1,))]
    with open(os.path.join(out_dir, "hexlib_batch.bin"), "wb") as f:
        f.write(wire.pack_batch(bufs, tensors, ops))
    with open(os.path.join(out_dir, "hexlib_in.bin"), "wb") as f:
        f.write(b"\x00" * (n * 2 * 2))

    cmd = rb.sim_qurt_command(out_dir, so, extra_args=("--unmapped",))
    bin_dir = tc.find_toolchain_bin(tc.default_sdk_root())
    env = tc.toolchain_env(bin_dir)
    rc, out, err, timed_out = tc.run(cmd, env, timeout=tc.SIM_TIMEOUT_MAX_S)
    combined = out + err

    assert not timed_out, combined
    assert "note=fd_deliberately_unmapped" in combined, combined
    # hexlib_iface_invoke returns AEE_SUCCESS at the RPC level even for a
    # batch-level failure (skel.c writes the real status into the response
    # header instead) -- so the discriminator is the STATUS field, not the
    # RPC return code. A never-mapped fd must come back ERR_UNMAPPED (7,
    # wire.STATUS), never OK (1): OK would mean the skel resolved an address
    # it was never given, which works only because the simulator shares one
    # address space with the host.
    m = re.search(r"SIMHOST invoke rc=0 rsp_len=\d+ status=(\d+)", combined)
    assert m, f"no successful-RPC SIMHOST invoke line recovered:\n{combined}"
    assert int(m.group(1)) == wire.STATUS["ERR_UNMAPPED"], (
        f"expected status={wire.STATUS['ERR_UNMAPPED']} (ERR_UNMAPPED), "
        f"got status={m.group(1)}:\n{combined}"
    )

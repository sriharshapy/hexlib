# hexlib/device/qdc/test_on_device.py
"""Runs ON THE DEVICE, under the farm's own pytest (see `artifact.py`'s
`requirements.txt`, which asks QDC's runner to `pip install pytest` before
running this file). NOT PART OF HEXLIB'S OWN SUITE -- see
`hexlib/tests/test_qdc_on_device_is_excluded.py` for the mechanism that keeps
`pytest hexlib/tests` from ever collecting this file, and the test that
proves it.

FAIL CLOSED, LOUDLY. A device-farm job on this project's own QDC account once
COMPLETED HAVING RUN ZERO TESTS AND REPORTED PASSING. So, throughout this
file: every expected line is asserted PRESENT, never merely "the bad thing is
absent" (an empty log satisfies "absent" trivially); the binary's OWN exit
code is checked (via the `; echo RC=$?` convention `utils.sh` documents), not
just whatever text happened to reach stdout; and a missing, empty, or
unparseable log is written and then failed on, never silently skipped.

EVERY STRING ASSERTED BELOW WAS READ DIRECTLY OUT OF
`hexlib/runtime/host/main.c`, not guessed or copied from an earlier draft of
this task. Two of the checks below (the unmapped-fd refusal and the
cache-coherency discriminator) need a small addition to `main.c` that does
not exist yet -- see the comment on each for exactly what and why, and
`.superpowers/sdd/2026-08-10-silicon-path-runtime/task-12-report.md` for the
full account. Those two are still written here, in full, on purpose: the
alternative -- leaving the requirement out because today's binary cannot
satisfy it -- is exactly the kind of silent gap this project's own history
(the false-pass job) says not to leave.
"""
import re

from utils import sh, write_qdc_log

DEV = "/data/local/tmp/hexlib"

# `hexlib_run` prints `hexlib: <mode>: cycles_total=%llu` (main.c's
# run_self_test and run_coherency_check). Built as a regex, not a substring,
# because the SUBSTRING IS SATISFIED BY `cycles_total=0` -- a run in which the
# DSP's PCYCLE counter never advanced at all. That is not a pedantic edge
# case: the skel reads the counter inside a user-mode unsigned PD, where
# SYSCFG.PCYCLEEN cannot be set (skel_dispatch.c's hexlib_read_pcycle, and
# include/hexlib/hexlib_harness.h, which sets that bit explicitly for the
# standalone-ELF runtime because the register reads 0 without it). Zero is
# precisely what this job would print if the counter is dead on this silicon,
# and it is the single most important thing this job can report.
_CYCLES_RE = re.compile(r"cycles_total=(\d+)")


def assert_cycles_total_is_a_real_measurement(out, what):
    """Assert `out` carries at least one `cycles_total=` line whose value is a
    decimal integer greater than zero, and return that value.

    PRESENT AND POSITIVE, as two separate failures with two separate
    messages -- "absent" and "zero" are different findings and must not be
    reported as each other. Absence means `hexlib_run` stopped printing it (or
    never got a response); zero means it printed a measurement of nothing,
    which on this platform points straight at PCYCLEEN in the unsigned PD."""
    values = [int(m) for m in _CYCLES_RE.findall(out)]
    assert values, (
        f"{what}: no `cycles_total=` line at all. hexlib_run prints one after "
        f"the PASS line (main.c's run_self_test) -- its ABSENCE is a failure, "
        f"never a success:\n{out}"
    )
    best = max(values)
    assert best > 0, (
        f"{what}: cycles_total={best} -- the DSP measured NOTHING. The kernel "
        f"call is bracketed by PCYCLE on the DSP (skel_dispatch.c), so a real "
        f"call cannot take zero cycles; a zero here means the counter did not "
        f"advance, which is what happens when SYSCFG.PCYCLEEN is clear -- and "
        f"a user-mode unsigned PD cannot set it. THIS IS THE MOST IMPORTANT "
        f"THING THIS JOB CAN REPORT: every cycle figure in stage 1 was "
        f"measured the same way, and if the counter is dead here then none of "
        f"them transfer to silicon:\n{out}"
    )
    return best


def test_binaries_are_present_and_executable():
    sh(f"mkdir -p {DEV}")
    sh(f"cp hexlib_run libhexlib_skel.so {DEV}/")
    sh(f"chmod 755 {DEV}/hexlib_run")
    out = sh(f"ls -l {DEV}")
    assert "hexlib_run" in out, f"hexlib_run did not land in {DEV}:\n{out}"
    assert "libhexlib_skel.so" in out, f"libhexlib_skel.so did not land in {DEV}:\n{out}"


def test_capabilities_report_a_v75_cdsp_with_unsigned_pd():
    """Every substring below is `print_caps()`'s OWN output format
    (`hexlib/runtime/host/main.c`), not the brief's earlier, wrong guesses
    (`ARCH_VER`, `UNSIGNED_PD_SUPPORT = 1`) -- the real lines are lowercase
    and shaped `domain              = CDSP (3)`, `unsigned_pd_support = 1`,
    `arch_ver            = 35957 (0x8c75)`. CDSP is domain 3, measured; ADSP
    (domain 0) is a v73 part with `UNSIGNED_PD_SUPPORT = 0` and must never be
    the thing this printed.

    THE EXIT CODE IS NOW CHECKED, AND UNTIL 2026-08-11 IT WAS NOT. This was
    the ONLY test in this file that ran `hexlib_run` without the `; echo RC=$?`
    convention, which made this file's own stated discipline -- "the binary's
    OWN exit code is checked" (module docstring) -- inapplicable to the one
    mode most likely to fail first on unfamiliar silicon. It was
    unenforceable, not merely unenforced: `print_caps()` returned `void` and
    `main()` returned `HEXLIB_EXIT_OK` unconditionally, so on a device whose
    image has no `libcdsprpc.so` for this ABI, `--caps` printed "could not
    load the FastRPC driver" and exited 0. `print_caps()` now returns
    HEXLIB_EXIT_SESSION_FAILED (2) on either failure branch, and this asserts
    RC=0 -- so a driver that will not load fails HERE, loudly, instead of
    being read as a pass by this test and by every `set -e` wrapper around
    it."""
    out = sh(f"cd {DEV} && ADSP_LIBRARY_PATH={DEV} ./hexlib_run --caps; echo RC=$?")
    write_qdc_log("hexlib_caps.log", out)
    assert "RC=0" in out, (
        f"hexlib_run --caps exited nonzero -- the FastRPC driver did not load "
        f"or the capability query failed (both are exit 2, "
        f"HEXLIB_EXIT_SESSION_FAILED):\n{out}"
    )
    assert "CDSP (3)" in out, f"expected domain CDSP (3), got:\n{out}"
    assert "arch_ver" in out, f"no arch_ver line at all:\n{out}"
    assert "35957" in out and "0x8c75" in out, f"unexpected arch, expected 35957 (0x8c75):\n{out}"
    assert "unsigned_pd_support = 1" in out, (
        f"expected unsigned_pd_support = 1 on CDSP, got:\n{out}"
    )


def test_scale_fp16_runs_on_the_dsp_and_is_correct():
    """`run_self_test()` in main.c prints exactly one line on success:
    `hexlib: --self-test: PASS (4100 values, bit-exact)` -- there is no
    `SELFTEST`/`status=1`/`cycles=` text anywhere in that function; the
    brief's earlier draft invented all three. Checked directly against the
    source before writing this assertion.

    GAP CLOSED 2026-08-11 -- AND THE STALE DOCSTRING IS WHY THE ASSERTION WAS
    MISSING. This said: "KNOWN GAP: `run_self_test()` computes
    `hexlib_batch_rsp_hdr.cycles_total` ... but never prints it, so this file
    cannot read a silicon cycle count off `hexlib_run`'s own stdout today."
    That was superseded by an earlier commit -- `main.c`'s `run_self_test`
    prints `hexlib: --self-test: cycles_total=%llu` right after the PASS line
    -- but the docstring stayed, and because it said the line could not be
    read, nothing here read it. The consequence was concrete: NOTHING ON
    DEVICE asserted the `cycles_total=` line at all, so deleting those two
    `printf` lines from main.c would have kept all five of this file's tests
    green while flipping `hexlib/cli.py`'s post-job check to exit 1 -- the
    device job passing and the gate above it failing, off the same run.

    The assertion below is now the load-bearing one for stage 3's most
    important open question: whether PCYCLE advances at all in a user-mode
    unsigned PD. See `assert_cycles_total_is_a_real_measurement`."""
    out = sh(f"cd {DEV} && ADSP_LIBRARY_PATH={DEV} ./hexlib_run --self-test; echo RC=$?")
    write_qdc_log("hexlib_selftest.log", out)
    assert "RC=0" in out, f"hexlib_run --self-test exited nonzero:\n{out}"
    assert "hexlib: --self-test: PASS" in out, (
        f"the PASS line must be PRESENT -- absence is failure, not success:\n{out}"
    )
    assert "bit-exact)" in out, f"PASS line present but not the bit-exact qualifier:\n{out}"
    assert_cycles_total_is_a_real_measurement(out, "hexlib_run --self-test")
    # A weaker, supplementary check ONLY -- the two asserts above already
    # require the specific success line to be present; this just also rules
    # out a run that printed both the PASS line AND a mismatch report, which
    # would be self-contradictory output worth catching on its own.
    assert "mismatch" not in out.lower()


def test_the_dsp_refuses_an_unmapped_fd_on_silicon_too():
    """The same discriminator that `hexlib/tests/test_dsp_sim.py` proved by
    mutation on the simulator (see `docs/STATE.md`'s Stage 1 entry), exercised
    on real hardware instead of `hexagon-sim`. It should pass trivially here
    -- but if it does NOT, the simulator was hiding something, and that is
    the single most important thing this job can report; hence this is
    asserted explicitly rather than left implicit in a passing self-test.

    SUPERSEDED 2026-08-11 -- THE FLAG NOW EXISTS (cd2582b). Kept because it records why it was added. Formerly: `hexlib_run`'s
    `main()` only ever inspects `argv[1]` (`--caps` / `--self-test` /
    `--batch`) -- there is no `--unmapped` flag today, unlike
    `hexlib/runtime/simhost/simhost.c`'s, which deliberately skips
    `hexlib_iface_mmap` for exactly this test. `--self-test --unmapped`
    therefore runs the ORDINARY self-test right now, ignoring the extra
    argument, and this test will fail (not vacuously pass) until main.c
    grows the small addition described in the task-12 report: build the
    second self-test buffer's fd via the driver-level rpcmem/fastrpc_mmap
    calls `hexlib_host.h` already exposes, WITHOUT the
    `hexlib_iface_mmap` registration call `hexlib_alloc` normally makes, so
    `hexlib_bufs_map`'s table lookup (skel_bufs.c) genuinely has nothing to
    find. That addition is deliberately NOT made here -- it would mean
    editing `hexlib/runtime/*`, out of this task's scope -- so this test
    documents the requirement and fails loudly rather than being silently
    dropped. The strings below are what `main.c`'s EXISTING status-handling
    code already prints once that one addition lands: `hexlib_dispatch_batch`
    (skel_dispatch.c) writes `HEXLIB_DSP_ERR_UNMAPPED` (7) as the batch's
    top-level status, and `run_self_test()`'s existing
    `status != HEXLIB_DSP_OK` branch already prints
    `"hexlib: --self-test: batch status %u, not HEXLIB_DSP_OK"` and returns
    `HEXLIB_EXIT_OP_FAILED` (4) -- no NEW print statement is needed, only the
    skipped registration call.
    """
    out = sh(f"cd {DEV} && ADSP_LIBRARY_PATH={DEV} ./hexlib_run --self-test --unmapped; echo RC=$?")
    write_qdc_log("hexlib_unmapped.log", out)
    assert "RC=4" in out, (
        f"expected HEXLIB_EXIT_OP_FAILED (4) once --unmapped exists; got:\n{out}"
    )
    assert "batch status 7, not HEXLIB_DSP_OK" in out, (
        f"expected HEXLIB_DSP_ERR_UNMAPPED (7) reported by the DSP, got:\n{out}"
    )


def test_cache_coherency_is_independent_of_marshalling_and_of_any_kernel():
    """Design doc §6.1: `buffers.c` maps with `FASTRPC_MAP_FD`, `remote.h`
    documents that flag as putting cache maintenance on US, `rpcmem`
    allocates CACHED memory by default, and the SDK has no CPU-side flush or
    invalidate call at all. A coherency miss on real hardware therefore
    presents EXACTLY like a marshalling bug -- wrong values out of a call
    that otherwise looks correct -- and the device path is the only place
    marshalling is exercised at all, so the two confound each other precisely
    where there is no cheaper way to tell them apart. This check exists so a
    failure says WHICH of the two it is on the FIRST job, not the third.

    SUPERSEDED 2026-08-11 -- THE FLAG NOW EXISTS (cd2582b) AND THIS DESIGN WAS INCOMPLETE; see the correction at the end. Formerly: Unlike the unmapped-fd
    flag above, `--coherency-check` does not exist ANYWHERE today, not even
    in shape (there is no simulator equivalent to mirror, because host and
    DSP share one address space there and a cache-coherency question does
    not exist to ask). This is this file's OWN proposed design for the
    smallest addition that would make the check expressible with capabilities
    `hexlib_run` already has -- described in full in the task-12 report --
    and it is NOT implemented, on purpose (implementing it means editing
    `hexlib/runtime/*`, out of this task's scope). The design, so the strings
    below are not arbitrary:

    1. Build the same two self-test buffers `run_self_test()` already builds
       (`x`, `y`), but before `hexlib_invoke`, the CPU writes a known
       NON-ZERO sentinel (e.g. every fp16 lane set to 1.0) into `y`'s rpcmem
       -- something `run_self_test()` does not do today, since it never
       reads `y` until after invoke.
    2. The batch's `scale` op uses `factor = 0.0`, not `0.125`. `x * 0.0` is
       bit-exact zero in fp16 for any finite, non-NaN `x` -- there is no
       numerically ambiguous case, so a wrong result here cannot be blamed on
       kernel arithmetic.
    3. After invoke, if the response is valid AND its status is
       `HEXLIB_DSP_OK` (proving the batch genuinely ran -- both marshalling
       and dispatch already succeeded), re-read `y`. If it is bit-exact
       all-zero, the DSP's write reached the CPU: coherent. If it still reads
       the sentinel, the DSP wrote zero but the CPU observed its OWN stale
       cache line instead -- conclusively a coherency miss, not a marshalling
       bug (marshalling already succeeded, per the status check) and not a
       kernel bug (the arithmetic is exact and trivial).
    4. Print a line whose presence states the verdict, e.g.
       `"COHERENCY sentinel_overwritten"` (coherent -- the DSP's write was
       observed) vs. `"COHERENCY sentinel_unchanged"` (a coherency miss).

    Until that lands, this test's own `sh()` call fails the moment the flag
    is rejected -- which is the honest state of this discriminator today: not
    yet expressible, not silently skipped.

    FIRST CORRECTION (design doc 6.1, 2026-08-11): THE SENTINEL ALONE DOES
    NOT DISCRIMINATE. If dispatch silently no-ops and still returns
    HEXLIB_DSP_OK, the observable is identical to a coherency miss: status OK,
    sentinel intact.

    SECOND CORRECTION (design doc 6.1, same day) -- AND THE FIRST
    CORRECTION'S OWN FIX WAS ALSO WRONG. It said `cycles_total` separates the
    two, and printed this table, WHICH IS NOW RETRACTED:
      cycles 0,  sentinel intact      -> kernel never ran: a dispatch bug
      cycles >0, sentinel intact      -> ran, write never reached host: COHERENCY
      cycles >0, sentinel overwritten -> healthy, for this direction
    It does not separate them, for three reasons:
      1. ROW 1 IS UNREACHABLE. main.c reads the sentinel only after BOTH the
         batch status and the op's own result->status are HEXLIB_DSP_OK, which
         in skel_dispatch.c happens only if `k->fn(&a)` was called and
         returned OK. PCYCLE brackets exactly that call, so cycles > 0 for any
         real call. Dispatch REFUSING the batch is distinguishable -- by exit
         4 and by NO COHERENCY line at all -- but not by a cycle count.
      2. THE NAMED DEFECT IS INVISIBLE TO IT. "Dispatch no-ops and returns OK"
         means a generated entry or kernel that returns OK WITHOUT WRITING
         `y`. PCYCLE still brackets a real returning call, so cycles > 0 and
         the sentinel is intact -- row 2 fires and prints
         `sentinel_unchanged`, misattributing a dispatch bug to coherency.
      3. IF PCYCLE READS 0 IN THE UNSIGNED PD, EVERY ROW INVERTS: a genuine
         coherency miss would read cycles 0 + sentinel intact, which row 1
         called a dispatch bug.

    WHAT THE OUTPUT ACTUALLY MEANS (the honest table; see design 6.1 and
    run_coherency_check()'s own header comment in main.c):
      no COHERENCY line, exit 4          -> dispatch refused the batch. The
                                            one genuinely diagnostic outcome.
      cycles_total=0, any COHERENCY line -> THE COUNTER IS DEAD, not "the
                                            kernel never ran" (see 1). Settle
                                            this before reading anything else.
      cycles >0, sentinel_overwritten    -> the DSP's write reached the host,
                                            for scale_fp16's write pattern and
                                            this buffer size only.
      cycles >0, sentinel_unchanged      -> NOT DISCRIMINATED: a coherency
                                            miss, OR a kernel/entry that
                                            returned OK without writing `y`,
                                            OR an fd mapped elsewhere. Rule
                                            the no-op out with the ordinary
                                            --self-test (factor 0.125, whose
                                            bit-exact values a no-op cannot
                                            produce) BEFORE spending a job on
                                            cache flags.
      cycles >0, buffer_garbled          -> partial or misdirected write. Its
                                            own outcome, never folded above.

    WHAT WOULD DISCRIMINATE, deliberately not built: a skel-side echo or
    memset op with its own kind id, so the write is performed by the skel and
    not by any generated kernel. Deferred in 6.1; reason 2 above is what that
    deferral costs.

    Two limits stated rather than implied: riding on `scale_fp16` is NOT
    kernel-independent (needs that skel-side echo op), and this covers only
    DSP-write -> host-read. The host-write -> DSP-read direction, which
    every input buffer and the batch blob depend on, is UNTESTED.

    """
    out = sh(
        f"cd {DEV} && ADSP_LIBRARY_PATH={DEV} ./hexlib_run --self-test "
        f"--coherency-check; echo RC=$?"
    )
    write_qdc_log("hexlib_coherency.log", out)
    assert "RC=0" in out, f"hexlib_run --self-test --coherency-check failed:\n{out}"
    assert "COHERENCY sentinel_overwritten" in out, (
        "the CPU must observe the DSP's own write, not a stale sentinel -- "
        f"a coherency miss looks exactly like a marshalling bug, and this line's "
        f"absence is that miss:\n{out}"
    )
    # NECESSARY, NOT SUFFICIENT -- see the second correction above. A positive
    # cycles_total does NOT prove this was a coherency verdict rather than a
    # dispatch one; it rules out exactly one thing, that the DSP measured
    # nothing at all, which would make every other line here unreadable.
    assert_cycles_total_is_a_real_measurement(
        out, "hexlib_run --self-test --coherency-check"
    )

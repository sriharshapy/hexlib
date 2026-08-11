"""hexlib command line.

The backend is a flag, never a code path inside a kernel: `--device sim` (the
default), `--device local`, `--device qdc` all run the same source, the same
harness, and produce the same result table.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import NamedTuple

from hexlib import kerneldir as kd
from hexlib.graph.plan import V75_VTCM_TOTAL_BYTES
from hexlib.result import Err, is_ok
from hexlib.verify import verify

DEVICES = ("sim", "local", "qdc")

# Above this many requested minutes, `--device qdc` refuses to submit without
# an explicit `--yes`. Not a limit on the job itself (job.py's own is
# 1..240) -- only on doing so without a human confirming it. 15 minutes is
# chosen to sit below the step-6 example the silicon-path plan itself uses
# (`--timeout-min 20 --yes`, which is deliberately ABOVE this threshold and
# therefore carries `--yes`), and above job.py's own docstring example of "a
# single, cheap, short-timeout dry run."
_QDC_YES_THRESHOLD_MIN = 15

# The two measurement strings a genuinely successful device run must have
# produced -- read directly out of hexlib/runtime/host/main.c (run_self_test's
# own printf calls), never guessed. A results.xml that parses clean with
# zero failures is NOT enough on its own: this project's own named failure
# mode is a job that completed having run (or measured) nothing at all, and
# a clean JUnit report with no measurements behind it is exactly that shape
# of success again, one level up. See _qdc_submit's post-parse check below.
_SELFTEST_PASS_MARKER = "hexlib: --self-test: PASS"
_CYCLES_TOTAL_MARKER = "cycles_total="

# `cycles_total=0` CONTAINS `cycles_total=`. That is the whole reason this
# regex exists. The presence check below (`marker not in combined`) was the
# only thing standing between "the logs carry a real measurement" and "a
# success value constructible with zero measurements in it" -- and a literal
# ZERO measurement satisfied it. Verified against fabricated local log files:
# `hexlib: --self-test: cycles_total=0` plus a clean `<testsuite tests="5"
# failures="0" errors="0">` printed "measurement lines present" and exited 0.
#
# Zero is not a pedantic edge case here, it is the EXPECTED shape of the
# failure this project is most exposed to. The DSP reads PCYCLE inside a
# user-mode unsigned PD, where SYSCFG.PCYCLEEN cannot be set from
# (skel_dispatch.c's own note, and include/hexlib/hexlib_harness.h's), so
# `cycles_total=0` is precisely what the first silicon job would print if the
# counter never advances there -- the single most important thing that job can
# tell us, and the one thing this check used to swallow.
#
# BUILT FROM THE MARKER RATHER THAN RESPELLING IT, so the presence check and
# the value check cannot drift apart. `(\S*)` deliberately captures whatever
# follows, valid or not, so a malformed value is DISTINGUISHABLE from an
# absent line instead of both silently reading as "no match".
_CYCLES_TOTAL_RE = re.compile(re.escape(_CYCLES_TOTAL_MARKER) + r"(\S*)")
_CYCLES_TOTAL_DIGITS = re.compile(r"[0-9]+")


def _qdc_cycles_total_verdict(combined: str) -> tuple[bool, str]:
    """`(ok, detail)` for the `cycles_total=` measurement in the fetched logs.

    `ok` is True only if at least one `cycles_total=` line carries a value
    that parses as a non-negative decimal integer AND is strictly greater
    than zero. `detail` always says which of the four states was found, so a
    caller's message names the real problem rather than "missing":

      - no `cycles_total=` line anywhere;
      - a line whose value is not a decimal integer at all (truncated log,
        interleaved output, a format change nobody updated this for);
      - every line reporting exactly 0 -- a measurement that measured
        nothing, which is the state this function was added for;
      - at least one positive value: the only pass.

    AT LEAST ONE, not all: a single job's logs legitimately contain several
    `cycles_total=` lines (`--self-test` prints one, `--coherency-check`
    prints another), and there is no requirement that every mode a job ran
    produced a nonzero count -- only that the run genuinely measured
    something. If PCYCLE is dead in the unsigned PD, EVERY line reads 0 and
    no `max` over them can rescue it, so taking the maximum cannot hide the
    failure mode this exists to catch."""
    found = _CYCLES_TOTAL_RE.findall(combined)
    if not found:
        return False, f"no `{_CYCLES_TOTAL_MARKER}` line anywhere in the fetched logs"

    values: list[int] = []
    malformed: list[str] = []
    for raw in found:
        if _CYCLES_TOTAL_DIGITS.fullmatch(raw):
            values.append(int(raw, 10))
        else:
            malformed.append(raw)

    positive = [v for v in values if v > 0]
    if positive:
        return True, f"{_CYCLES_TOTAL_MARKER}{max(positive)}"
    if values:
        return False, (
            f"every `{_CYCLES_TOTAL_MARKER}` line reports 0 "
            f"({len(values)} such line(s)) -- the DSP measured NOTHING, which "
            "is what PCYCLE returns when SYSCFG.PCYCLEEN is clear, and a "
            "user-mode unsigned PD cannot set it"
        )
    return False, (
        f"`{_CYCLES_TOTAL_MARKER}` is present but its value is not a decimal "
        f"integer: {malformed[0]!r}"
    )

# The operator's own account budget, in minutes -- read from the environment,
# exactly the way job.py reads QDC_API_KEY, because it is personal and this
# module has no way to learn it without a network call (which no CLI code
# path may ever make from inside a test, and which this function does not
# make at all, from anywhere). Unset means "unknown," printed as such, never
# guessed at.
_QDC_BUDGET_ENV = "QDC_BUDGET_MIN"

# Only a plain non-negative decimal count of minutes. `QDC_BUDGET_MIN=abc`
# used to be echoed verbatim as "remaining budget: abc minutes" -- a number
# that is not a number, printed as though it were one, and compared against
# nothing at all.
_QDC_BUDGET_RE = re.compile(r"[0-9]+")


class _QdcBudgetError(Exception):
    """`QDC_BUDGET_MIN` is set to something that is not a count of minutes.
    Refused rather than ignored: a budget guard that silently disables itself
    on a typo is worse than no guard, because the operator believes it is
    watching."""


def _qdc_remaining_budget_min() -> int | None:
    """The operator's own recorded remaining minutes, or None if UNSET.

    Read from the environment, exactly the way job.py reads QDC_API_KEY,
    because it is personal and this module has no way to learn it without a
    network call (which no CLI code path may ever make from inside a test,
    and which nothing here makes at all, from anywhere). Never queries QDC:
    there is no such API on this account (job.py's own module docstring:
    `get_job_status` returns `state=None`, `get_jobs_list` lags over 30
    minutes), so the only honest source is whatever the operator has
    recorded for themselves.

    WHAT UNSET MEANS -- A DECISION, NOT AN OVERSIGHT. Unset means UNKNOWN,
    and unknown means this CLI performs no budget comparison and submits
    anyway (loudly saying so). It does NOT mean unlimited, and it must not be
    read as an assurance that the job fits. The alternative -- refusing to
    submit at all without the variable -- was considered and rejected for two
    reasons: (1) the documented stage-3 entry point (`docs/STATE.md`, the
    plan's step 6) does not set it, so making it mandatory would mean no job
    can ever be submitted the way this project's own instructions say to; and
    (2) a mandatory number nobody can verify invites `QDC_BUDGET_MIN=99999`
    under time pressure, which yields a guard that is present, green, and
    meaningless -- strictly worse than a stated "unknown, not checked". The
    `--timeout-min`/`--yes` confirmation threshold still applies either way,
    and it is the guard that does not depend on the operator having recorded
    anything.

    Raises `_QdcBudgetError` if the variable is set but is not a non-negative
    integer.
    """
    raw = os.environ.get(_QDC_BUDGET_ENV)
    if raw is None:
        return None
    text = raw.strip()
    if not _QDC_BUDGET_RE.fullmatch(text):
        raise _QdcBudgetError(
            f"{_QDC_BUDGET_ENV}={raw!r} is not a non-negative whole number of "
            "minutes. It is the only thing standing between a --timeout-min "
            "and a budget it does not fit in, so a value that cannot be "
            "compared is refused rather than printed and ignored. Unset it "
            "to submit with no budget check at all (which will say so)."
        )
    return int(text, 10)


def _qdc_budget_guard(timeout_min: int) -> int:
    """Print the (locally recorded, never queried) remaining budget and
    COMPARE it to `timeout_min`. Returns 0 to proceed, 2 to refuse.

    THE COMPARISON IS THE POINT. This function used to only print, which
    meant `QDC_BUDGET_MIN=3 hexlib test scale_fp16 --device qdc --timeout-min
    240 --yes` printed `remaining budget: 3 minutes` and then submitted a
    240-minute job -- an 80x overspend of non-renewable minutes passing every
    guard, with the figure that would have caught it on screen.

    `timeout_min > budget` is refused, not warned about: `--timeout-min` is
    the ceiling QDC itself will enforce on the job, so a job whose ceiling
    exceeds the stated remaining budget can, on its own, exhaust the account.
    Equality is allowed (spending the last minutes deliberately is a real
    thing to want); exceeding is not.
    """
    try:
        budget = _qdc_remaining_budget_min()
    except _QdcBudgetError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if budget is None:
        print(
            f"remaining budget: unknown ({_QDC_BUDGET_ENV} is not set) -- "
            "NO BUDGET CHECK WAS PERFORMED. Nothing here queries QDC for a "
            "remaining-minutes figure (there is no reliable API for it on "
            f"this account), and unset means unknown, NOT unlimited: set "
            f"{_QDC_BUDGET_ENV} to have --timeout-min actually checked "
            "against it."
        )
        return 0

    print(f"remaining budget: {budget} minutes (from {_QDC_BUDGET_ENV})")
    if timeout_min > budget:
        print(
            f"error: --timeout-min {timeout_min} exceeds the {budget} "
            f"minute(s) recorded in {_QDC_BUDGET_ENV} -- refusing to submit a "
            "job whose own timeout is larger than the budget it has to spend "
            "from. Device minutes are non-renewable. Lower --timeout-min, or "
            f"correct {_QDC_BUDGET_ENV} if it is stale.",
            file=sys.stderr,
        )
        return 2
    return 0


# Stage 3 runs exactly one kernel, and it is not a parameter of anything.
# device/qdc/test_on_device.py hard-codes `./hexlib_run --self-test`, and
# main.c's run_self_test() hard-codes build_scale_batch() -- so the kernel the
# device actually exercises is fixed in C, three layers below the CLI.
_QDC_SUPPORTED_KERNEL = "scale_fp16"


def _qdc_kernel_refusal(kernel: object) -> str | None:
    """None if `kernel` names the one kernel `--device qdc` can genuinely
    run; otherwise the operator-facing reason it is refused.

    WHY REFUSE RATHER THAN IGNORE. `--device qdc` accepted the `<kernel>`
    argument and read it nowhere: `hexlib test add_fp16 --device qdc
    --timeout-min 20 --yes` spent 20 non-renewable minutes running
    `scale_fp16` and returned green, and the operator's notebook then said
    "add_fp16 validated on silicon". Making the argument actually work means
    parameterising main.c's batch builder and the staged on-device script --
    a real change, deliberately not made here. Refusing is the honest
    intermediate state: the command either does what it was asked or says it
    cannot.

    Accepts a bare name (`scale_fp16`, what docs/STATE.md's own entry point
    uses) or a path to the kernel directory (`kernels/scale_fp16`, what
    `--device sim` takes), since one CLI takes both.
    """
    if not isinstance(kernel, str) or not kernel.strip():
        return (
            "--device qdc needs the <kernel> argument to name "
            f"{_QDC_SUPPORTED_KERNEL} explicitly; got {kernel!r}. It is not "
            "optional and it is not ignored -- see _qdc_kernel_refusal."
        )
    name = os.path.basename(os.path.normpath(kernel.strip()))
    if name != _QDC_SUPPORTED_KERNEL:
        return (
            f"--device qdc cannot run {name!r}: stage 3 is "
            f"{_QDC_SUPPORTED_KERNEL}-only today. The staged on-device script "
            "(hexlib/device/qdc/test_on_device.py) hard-codes `./hexlib_run "
            "--self-test`, which runs main.c's fixed build_scale_batch(), so "
            f"submitting this would spend real device minutes measuring "
            f"{_QDC_SUPPORTED_KERNEL} and report the result under {name!r}. "
            f"Run `hexlib test {_QDC_SUPPORTED_KERNEL} --device qdc ...`, or "
            "use --device sim for any other kernel."
        )
    return None


# Extra wall-clock slack, beyond the job's own `--timeout-min`, that `wait()`
# will keep polling for. Covers everything that happens outside the timeout
# QDC enforces on the run itself: queueing for a free SM8650, provisioning,
# and the farm collecting and publishing TestLogs/ afterwards. A POLICY
# CHOICE, NOT A MEASUREMENT -- stage 3 has never run on this account, so
# there is no observed queue time to derive it from; 15 minutes is chosen to
# be comfortably longer than any single step above plausibly takes.
#
# Waiting longer is FREE. Minutes are spent by the job, bounded by its own
# timeout; the CLI blocking on `get_job_log_files` costs nothing. Waiting too
# LITTLE is what costs: `wait()`'s old hard-coded 1800 s cap against a
# `submit()` that accepts 240 minutes meant a legitimate 35-minute job under
# `--timeout-min 60` was abandoned at 30 minutes, the minutes already spent,
# and the results.xml that appeared five minutes later never fetched.
_QDC_WAIT_GRACE_S = 900


def _qdc_wait_cap_s(timeout_min: int) -> int:
    """How long to poll for results.xml, DERIVED FROM THE JOB'S OWN TIMEOUT
    rather than a constant that can silently be smaller than it."""
    return timeout_min * 60 + _QDC_WAIT_GRACE_S


def _qdc_fetch_logs(job_id: int, log_dir: str) -> tuple[list[str], str | None]:
    """`(paths, error_text)` -- downloads whatever logs QDC has and NEVER
    raises. Used on the giving-up path as well as the happy one: a timeout
    that discards the evidence is worse than one that waits, and the fetch
    failing is not a reason to also throw away the fact that the job ran."""
    from hexlib.device.qdc import job

    try:
        return job.fetch(job_id, log_dir), None
    except Exception as e:                      # noqa: BLE001 -- see docstring
        return [], f"{type(e).__name__}: {e}"


def _qdc_submit(args) -> int:
    """The real work, reached only once every guard in `_cmd_test_qdc` has
    already passed: build the device artifacts, stage them together with the
    on-device pytest, and submit to QDC. Isolated into its own function so
    tests can monkeypatch it directly and verify the guards run in the right
    order and print the right things WITHOUT ever touching the SDK, a
    credential, or the network -- none of which any test may require or
    contact.
    """
    from hexlib.device.qdc import artifact, job
    from hexlib.runtime import build as runtime_build

    # RE-CHECKED HERE, not only in _cmd_test_qdc. This is the function that
    # spends the minutes, and it is called directly (by tests today, and by
    # any second caller tomorrow) without going through _cmd_test_qdc's
    # guards at all -- `getattr` with no default so an args object carrying no
    # `kernel` attribute is refused rather than silently submitting for
    # whatever main.c happens to hard-code.
    refusal = _qdc_kernel_refusal(getattr(args, "kernel", None))
    if refusal is not None:
        print(f"error: {refusal}", file=sys.stderr)
        return 2

    build_dir = os.path.join(args.out, "qdc_build")
    try:
        hexlib_run = runtime_build.build_device_binary(build_dir)
    except runtime_build.RuntimeBuildError as e:
        print(f"error: building the device artifacts failed: {e}", file=sys.stderr)
        return 1
    skel_so = os.path.join(build_dir, "libhexlib_skel.so")

    here = os.path.dirname(__file__)
    test_script = os.path.join(here, "device", "qdc", "test_on_device.py")
    utils_py = os.path.join(here, "device", "qdc", "utils.py")

    out_base = os.path.join(args.out, "qdc_job")
    try:
        zip_path = artifact.stage([hexlib_run, skel_so, utils_py], test_script, out_base)
    except artifact.StagingError as e:
        print(f"error: staging the QDC artifact failed: {e}", file=sys.stderr)
        return 1

    try:
        job_id = job.submit(zip_path, timeout_min=args.timeout_min)
    except job.QdcError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"submitted job {job_id} (timeout {args.timeout_min} min)")

    log_dir = os.path.join(args.out, "qdc_logs")
    cap_s = _qdc_wait_cap_s(args.timeout_min)

    # cap_s IS PASSED EXPLICITLY. Omitting it took job.wait's 1800 s default,
    # which is SMALLER than the timeout submit() accepts (240 min) -- see
    # _QDC_WAIT_GRACE_S for the full account of what that cost.
    if not job.wait(job_id, cap_s=cap_s):
        # FETCH THE EVIDENCE BEFORE GIVING UP. The minutes are already spent;
        # whatever the farm has published so far (logcat, a partial
        # TestLogs/, the caps log) is the only thing the operator can
        # diagnose from, and abandoning it leaves them with nothing but "it
        # timed out". This deliberately does NOT then evaluate those logs as a
        # result: a job that never produced results.xml within the cap is a
        # failure regardless of what else it printed.
        paths, fetch_error = _qdc_fetch_logs(job_id, log_dir)
        print(
            f"error: job {job_id} produced no results.xml within the "
            f"{cap_s}s wait cap (derived from --timeout-min "
            f"{args.timeout_min}) -- a job with no results is a failure, "
            "never a pass",
            file=sys.stderr,
        )
        if fetch_error is not None:
            print(
                f"error: job {job_id}: fetching the logs that DO exist also "
                f"failed ({fetch_error}) -- check the job by hand in the QDC "
                "console before spending more minutes",
                file=sys.stderr,
            )
        else:
            print(
                f"fetched {len(paths)} log file(s) that existed at the cap to "
                f"{log_dir} -- the job may still finish; its results.xml was "
                "not there yet",
                file=sys.stderr,
            )
        return 1

    paths, fetch_error = _qdc_fetch_logs(job_id, log_dir)
    if fetch_error is not None:
        print(
            f"error: job {job_id}: results.xml appeared but fetching the log "
            f"files failed ({fetch_error}) -- an unverifiable job is a "
            "failure, never a pass",
            file=sys.stderr,
        )
        return 1
    print(f"fetched {len(paths)} log file(s) to {log_dir}")

    return _qdc_check_results(job_id, paths)


class _QdcResultsError(Exception):
    """Raised by `_qdc_parse_results_xml` for any results.xml that must not
    be treated as a pass -- unparseable, a shape this project does not
    produce, MISSING any of the attributes a JUnit report always carries
    (`_REQUIRED_SUITE_ATTRS`; a missing one is never a zero), or carrying a
    negative count. Caught by `_qdc_check_results`, never allowed to
    propagate past `_qdc_submit`."""


class _JUnitCounts(NamedTuple):
    """Every count `_qdc_check_results` needs, all four of them REQUIRED.

    `skipped` is here because it was for a while parsed NOWHERE, which made a
    collected-but-never-run test indistinguishable from a passing one:
    `<testsuite tests="5" failures="0" errors="0" skipped="5">` plus a good
    self-test log exited 0 and printed "5 test(s), 0 failures, 0 errors" with
    no mention of the skips. On a device a skip overwhelmingly means the test
    could not run at all, which is this project's own named failure mode
    (absence read as success) wearing a different attribute name."""

    tests: int
    failures: int
    errors: int
    skipped: int


# EVERY ONE OF THESE IS REQUIRED ON EVERY <testsuite>, AND A MISSING ONE IS A
# MALFORMED REPORT, NOT A ZERO. `suite.get("failures", "0")` read a report
# with no `failures` attribute at all as a clean pass -- verified:
# `<testsuite tests="5"></testsuite>` plus a good log exited 0 and reported
# "0 failures, 0 errors". That directly contradicted `_QdcResultsError`'s own
# docstring ("missing the attributes a JUnit report always carries"), and it
# is the same absence-read-as-success shape the rest of this file exists to
# refuse: the counts that decide the verdict must be PRESENT, never defaulted.
#
# All four really are always emitted by the only producer this parser is
# pinned to -- pytest's own `--junitxml` (device/qdc/artifact.py's pytest.ini)
# writes `errors`, `failures`, `skipped`, `tests`, `time`, `timestamp`,
# `hostname` and `name` on every `<testsuite>` it emits. Confirmed by
# generating one locally with this repo's own pytest, not assumed from
# memory. Refusing a report that lacks any of them therefore cannot reject a
# report our own device job produced; it rejects a truncated or foreign one,
# which is the safe direction to fail in (the caller reports a parse failure
# as a failure, never a pass).
_REQUIRED_SUITE_ATTRS = ("tests", "failures", "errors", "skipped")


def _qdc_parse_results_xml(path: str) -> _JUnitCounts:
    """Parse a JUnit-style results.xml and return the `_JUnitCounts` summed
    across every `<testsuite>` element. Raises `_QdcResultsError` on anything
    that is not a genuinely parseable report with real counts on it -- a
    truncated or non-XML file, a `<testsuites>`/`<testsuite>` tree with no
    testsuite elements at all, a `<testsuite>` missing any of
    `_REQUIRED_SUITE_ATTRS`, a negative count, or a shape this function does
    not recognize -- so the caller never has to guess whether "zero" means
    "ran zero tests" or "could not even find the count".

    ONLY ONE SHAPE IS ACCEPTED, PINNED TO WHAT THIS PROJECT ACTUALLY
    PRODUCES, NOT GUESSED AT AS A GENERAL JUNIT PARSER. The on-device job's
    own pytest.ini (device/qdc/artifact.py's `_PYTEST_INI`) sets
    `--junitxml=TestLogs/results.xml`, and pytest's `--junitxml` always
    emits exactly one `<testsuite>`, either as the document root or as the
    sole immediate child of a `<testsuites>` wrapper -- it never nests one
    `<testsuite>` inside another. An earlier version of this function
    accepted ANY shape by summing `root.findall(".//testsuite")` -- every
    `<testsuite>` at any depth -- which would silently DOUBLE-COUNT a report
    whose parent `<testsuite>` totals already include a nested child's
    counts. Rather than guess how such a report should be summed, this
    refuses it outright: a parse failure here blocks a false pass (the
    caller reports it as a failure, never a pass -- see
    `_qdc_check_results`), which is the safe direction to fail in.
    """
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise _QdcResultsError(f"{path} did not parse as XML: {e}") from e

    if root.tag == "testsuite":
        suites = [root]
    elif root.tag == "testsuites":
        suites = root.findall("testsuite")   # DIRECT children only.
    else:
        raise _QdcResultsError(
            f"{path} root is <{root.tag}>, not <testsuite> or <testsuites> "
            "-- not a JUnit report shape this project recognizes"
        )
    if not suites:
        raise _QdcResultsError(
            f"{path} contains no <testsuite> element -- not a JUnit report "
            "this project recognizes"
        )

    # Refuse a <testsuite> nested inside another <testsuite> ANYWHERE in the
    # tree, rather than silently summing it -- pytest's own --junitxml never
    # produces this shape (see the docstring above), and summing it would
    # double-count a parent's already-rolled-up totals.
    for suite in suites:
        if suite.findall(".//testsuite"):
            raise _QdcResultsError(
                f"{path} has a <testsuite> nested inside another <testsuite> "
                "-- not the flat shape pytest's --junitxml produces, and "
                "summing nested totals would double-count them; refusing "
                "rather than guessing how to sum it"
            )

    totals = dict.fromkeys(_REQUIRED_SUITE_ATTRS, 0)
    for suite in suites:
        for attr in _REQUIRED_SUITE_ATTRS:
            raw = suite.get(attr)
            if raw is None:
                raise _QdcResultsError(
                    f"{path} has a <testsuite> with no {attr!r} attribute -- "
                    "pytest's own --junitxml always writes all of "
                    f"{', '.join(_REQUIRED_SUITE_ATTRS)}, so an absent one is a "
                    "malformed or truncated report and must NEVER be read as "
                    "zero (that is how a report carrying no failure count at "
                    "all used to be reported as '0 failures, 0 errors')"
                )
            try:
                value = int(raw)
            except ValueError as e:
                raise _QdcResultsError(
                    f"{path} has a non-integer {attr!r} attribute: {e}"
                ) from e
            if value < 0:
                raise _QdcResultsError(
                    f"{path} has {attr}={raw!r}, a NEGATIVE count -- pytest "
                    "cannot produce that, so the report is malformed; a "
                    "negative count must not be summed into a total that then "
                    "compares as 'no failures'"
                )
            totals[attr] += value
    return _JUnitCounts(**totals)


def _qdc_check_results(job_id: int, paths: list[str]) -> int:
    """The parse `job.wait`/`job.fetch` never do. A device job whose
    TestLogs/results.xml merely *exists* proves nothing on its own -- this
    project's own named failure mode is a job that completed having run (or
    measured) NOTHING and still reported passing. Every one of the following
    must hold before this returns 0, each with its own distinct message so a
    real failure is diagnosable from which check tripped:

      - a results.xml was actually fetched, and it PARSES as XML, WITH all
        four of `_REQUIRED_SUITE_ATTRS` actually present on every
        `<testsuite>` (a missing `failures`/`errors` is a malformed report,
        never a zero);
      - it reports `tests > 0` -- zero tests is a failure, never a pass, and
        so is a negative count;
      - `failures == 0` and `errors == 0`;
      - `skipped == 0`. A SKIP IS NOT A PASS. On this device path a skip
        cannot mean "not applicable here": every test in
        device/qdc/test_on_device.py is unconditional, so a skip means the
        farm's pytest collected a test and then did not run it -- the
        device could not be reached, a `skipif` was added upstream, or
        collection half-failed. Reported (and refused) with its own message
        rather than folded into the failure count, because "5 skipped" and
        "5 failed" call for completely different next actions;
      - the fetched logs actually CONTAIN the measurement lines
        `hexlib_run` itself prints on a genuine pass (`cycles_total=` and
        the `--self-test` PASS line, both read directly out of main.c) --
        a clean JUnit report with none of hexlib's own evidence behind it
        is exactly the "success constructible with zero measurements in
        it" shape this check exists to rule out;
      - and the `cycles_total=` line's VALUE is a real integer greater
        than zero. The bullet above was for a while the whole of this
        check, and `cycles_total=0` satisfied it -- the substring is
        there, so a run in which the DSP measured literally nothing
        printed "measurement lines present" and exited 0. See
        `_qdc_cycles_total_verdict` for why zero is the expected shape of
        the failure rather than a pedantic edge case.
    """
    results_path = next(
        (p for p in paths if os.path.basename(p) == "results.xml"), None
    )
    if results_path is None:
        print(
            f"error: job {job_id}: results.xml was not among the fetched log "
            "files -- a job with no results is a failure, never a pass",
            file=sys.stderr,
        )
        return 1

    try:
        counts = _qdc_parse_results_xml(results_path)
    except _QdcResultsError as e:
        print(
            f"error: job {job_id}: could not parse results.xml as a JUnit "
            f"report -- a truncated or unparseable results file is a "
            f"failure, never a pass: {e}",
            file=sys.stderr,
        )
        return 1

    tests, failures, errors, skipped = counts

    # `<= 0`, NOT `== 0`. A negative total is already refused by the parser,
    # so this is belt-and-braces rather than the only guard -- but `tests ==
    # 0` was verified to let `<testsuite tests="-1" failures="0" errors="0">`
    # through, printing "-1 test(s)" and exiting 0, and a comparison that only
    # catches the exact value it was written for is not a bound.
    if tests <= 0:
        print(
            f"error: job {job_id}: results.xml reports {tests} tests -- a job "
            "that ran no tests is a failure, never a pass",
            file=sys.stderr,
        )
        return 1

    if failures != 0 or errors != 0:
        print(
            f"error: job {job_id}: results.xml reports {failures} failure(s) "
            f"and {errors} error(s) across {tests} test(s) "
            f"({skipped} skipped)",
            file=sys.stderr,
        )
        return 1

    # A SKIP IS NOT A PASS -- see this function's docstring.
    if skipped != 0:
        print(
            f"error: job {job_id}: results.xml reports {skipped} skipped "
            f"test(s) out of {tests} -- every test in "
            "hexlib/device/qdc/test_on_device.py is unconditional, so a skip "
            "on device means a test was collected and never actually run. "
            "Collected-but-not-run is not passed: this is the project's own "
            "named failure mode (absence read as success) spelled `skipped=`",
            file=sys.stderr,
        )
        return 1

    combined = ""
    for p in paths:
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                combined += f.read()
        except OSError:
            continue

    missing = [
        marker
        for marker in (_CYCLES_TOTAL_MARKER, _SELFTEST_PASS_MARKER)
        if marker not in combined
    ]
    if missing:
        print(
            f"error: job {job_id}: results.xml reports {tests} test(s) with "
            "no failures, but the fetched logs are missing the expected "
            f"measurement line(s): {', '.join(missing)!r} -- a pass with no "
            "measurements behind it is the exact failure mode this check "
            "exists to rule out",
            file=sys.stderr,
        )
        return 1

    # PRESENCE IS NOT MEASUREMENT. The check above only proved the substring
    # `cycles_total=` appears; this one reads the number after it.
    cycles_ok, cycles_detail = _qdc_cycles_total_verdict(combined)
    if not cycles_ok:
        print(
            f"error: job {job_id}: results.xml reports {tests} test(s) with "
            f"no failures, but the DSP's own cycle measurement is not usable: "
            f"{cycles_detail} -- a pass whose only measurement is zero is "
            "still a pass with no measurements behind it",
            file=sys.stderr,
        )
        return 1

    # The skip count is printed on the PASS line too, not only when it is
    # nonzero: a success line that silently omits a count it checked leaves a
    # reader unable to tell "0 skipped" from "skips were never looked at",
    # which is exactly the state this line was in before.
    print(
        f"job {job_id}: {tests} test(s), 0 failures, 0 errors, "
        f"{skipped} skipped, measurement lines present ({cycles_detail})"
    )
    return 0


def _cmd_test_qdc(args) -> int:
    """`--device qdc`: refuses a kernel stage 3 cannot actually run, refuses
    without an explicit `--timeout-min`, refuses a `--timeout-min` outside
    job.py's own 1..240, always prints the (locally known, never queried)
    remaining budget and refuses a job that does not fit in it, and requires
    `--yes` above `_QDC_YES_THRESHOLD_MIN` -- every one of these guards runs
    before `_qdc_submit` ever touches the SDK, a credential, or the network.

    THE KERNEL CHECK IS FIRST, deliberately. "This command cannot run the
    thing you asked for" is more useful than "you forgot --timeout-min" when
    both are true, and it is the cheapest of the five.
    """
    refusal = _qdc_kernel_refusal(getattr(args, "kernel", None))
    if refusal is not None:
        print(f"error: {refusal}", file=sys.stderr)
        return 2

    if args.timeout_min is None:
        print(
            "error: --device qdc requires --timeout-min (1..240) -- a "
            "runaway job spends real, non-renewable minutes, and there is "
            "no default that could be right for every account.",
            file=sys.stderr,
        )
        return 2

    # RANGE-CHECKED HERE, NOT ONLY IN job.submit. job.py enforces 1..240 too
    # (it is the authority, and these bounds are imported from it rather than
    # respelled), but it does so AFTER _qdc_submit has run a full SDK build of
    # hexlib_run + libhexlib_skel.so and staged a zip -- minutes of local work
    # thrown away to reject an argument that was wrong before any of it
    # started. A lazy import: job.py pulls in nothing but the stdlib at module
    # scope, and never the vendor SDK.
    from hexlib.device.qdc import job as qdc_job

    if not qdc_job.MIN_TIMEOUT_MIN <= args.timeout_min <= qdc_job.MAX_TIMEOUT_MIN:
        print(
            f"error: --timeout-min must be {qdc_job.MIN_TIMEOUT_MIN}.."
            f"{qdc_job.MAX_TIMEOUT_MIN}, got {args.timeout_min} -- QDC itself "
            "refuses anything else, and finding that out only after a full "
            "device build has been run and staged wastes the build.",
            file=sys.stderr,
        )
        return 2

    budget_rc = _qdc_budget_guard(args.timeout_min)
    if budget_rc != 0:
        return budget_rc

    if args.timeout_min > _QDC_YES_THRESHOLD_MIN and not args.yes:
        print(
            f"error: --timeout-min {args.timeout_min} is above the "
            f"{_QDC_YES_THRESHOLD_MIN}-minute confirmation threshold -- pass "
            "--yes to submit anyway. This does not limit the job itself, "
            "only submitting one this size without a human confirming it.",
            file=sys.stderr,
        )
        return 2

    return _qdc_submit(args)


def _cmd_new_kernel(args) -> int:
    try:
        path = kd.scaffold(args.kernels_root, args.name)
    except FileExistsError as e:
        print(str(e) + " — already exists", file=sys.stderr)
        return 1
    print(f"created {path}")
    print("Next: fill in kernel_api.h with the exact contract, write baseline.c,")
    print("then harness.c, then kernel.c. Run: hexlib test " + path)
    return 0


def _cmd_validate(args) -> int:
    problems = kd.validate_dir(args.kernel)
    if problems:
        for p in problems:
            print(f"error: {p}", file=sys.stderr)
        return 1
    print(f"{args.kernel}: valid")
    return 0


def _cmd_test(args) -> int:
    # `--timeout-min` and `--yes` exist ONLY for `--device qdc`; both were
    # accepted and silently ignored for sim/local, so `hexlib test k --device
    # sim --timeout-min 20 --yes` looked like it had asked for something and
    # had it granted. Refused rather than warned about: the two flags are the
    # spend controls for the one backend that spends anything, and a spend
    # control that is accepted where it does nothing teaches an operator that
    # passing it is harmless.
    if args.device != "qdc":
        ignored = [
            flag
            for flag, given in (
                ("--timeout-min", args.timeout_min is not None),
                ("--yes", bool(args.yes)),
            )
            if given
        ]
        if ignored:
            print(
                f"error: {' and '.join(ignored)} appl"
                f"{'y' if len(ignored) > 1 else 'ies'} only to --device qdc, "
                f"not --device {args.device} -- nothing here spends device "
                "minutes, so there is nothing to time out or to confirm. "
                "Refused rather than ignored.",
                file=sys.stderr,
            )
            return 2

    if args.device == "local":
        print(
            "error: --device local is not implemented, no device available -- "
            "the shape exists so a contributor with a phone can wire it up; "
            "this codebase does not pretend it works without one.",
            file=sys.stderr,
        )
        return 2
    if args.device == "qdc":
        return _cmd_test_qdc(args)

    result = verify(args.kernel, args.out)
    if not is_ok(result):
        print(f"error: {result.reason}", file=sys.stderr)
        if result.detail:
            print(result.detail, file=sys.stderr)
        return 1

    table_path = os.path.join(
        args.out, f"{os.path.basename(os.path.normpath(args.kernel))}.result.md"
    )
    if os.path.isfile(table_path):
        with open(table_path, encoding="utf-8") as f:
            print(f.read())
    print(f"result table: {table_path}")
    result_md = os.path.join(args.kernel, "RESULT.md")
    print(f"Wrote {result_md} — commit it with your kernel. CI has no SDK and "
          "cannot regenerate it.")
    print("Paste the same table into your pull request description for reviewers.")
    return 0


def _succeeded(value: object) -> bool:
    """True unless `value` is an `Err`.

    Distinct from `hexlib.result.is_ok`, which tests membership in that
    module's `Ok`/`Err` `Result` type (used by `verify`). The graph passes use
    a different convention -- `Graph | Err`, `Plan | Err` -- where success is
    the value itself, not a wrapper. This checks that convention, for
    whichever value in it `_cmd_plan` is holding at the time (a `Graph` right
    after `build_vision_encoder`, a `Plan` right after `compile_graph`).
    """
    return not isinstance(value, Err)


def _cmd_plan(args) -> int:
    import hexlib.graph.opdefs  # noqa: F401  -- registers the op definitions
    from hexlib.graph.pipeline import compile_graph
    from hexlib.graph.plan import render, to_json
    from hexlib.models.qwen35 import qwen35_at
    from hexlib.models.vit import build_vision_encoder

    if args.model != "qwen35":
        print(f"error: unknown model {args.model!r}; known: qwen35", file=sys.stderr)
        return 2

    graph = build_vision_encoder(qwen35_at(args.image_size))
    if not _succeeded(graph):
        print(f"error: {graph.reason}", file=sys.stderr)
        print(graph.detail, file=sys.stderr)
        return 1

    plan = compile_graph(
        graph,
        budget=args.vtcm_bytes,
        order_policy=args.order_policy,
        alloc_policy=args.alloc_policy,
    )
    if not _succeeded(plan):
        print(f"error: {plan.reason}", file=sys.stderr)
        print(plan.detail, file=sys.stderr)
        return 1

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(to_json(plan))
        print(f"wrote {args.out}")
    if args.print_plan or not args.out:
        print(render(plan))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hexlib")
    sub = p.add_subparsers(dest="cmd", required=True)

    n = sub.add_parser("new-kernel", help="scaffold a conforming kernel directory")
    n.add_argument("name")
    n.add_argument("--kernels-root", default="kernels")
    n.set_defaults(func=_cmd_new_kernel)

    v = sub.add_parser("validate", help="check a kernel directory's structure")
    v.add_argument("kernel")
    v.set_defaults(func=_cmd_validate)

    t = sub.add_parser("test", help="run the gate for a kernel")
    t.add_argument("kernel")
    t.add_argument("--device", choices=DEVICES, default="sim")
    t.add_argument("--out", default="_work")
    t.add_argument(
        "--timeout-min", type=int, default=None,
        help="required for --device qdc (1..240); no default, a runaway job "
             "spends real money",
    )
    t.add_argument(
        "--yes", action="store_true",
        help="confirm a --device qdc submission above the confirmation "
             "threshold (see --timeout-min)",
    )
    t.set_defaults(func=_cmd_test)

    pl = sub.add_parser("plan", help="compile a model's encoder to a VTCM/DMA plan")
    pl.add_argument("model", choices=("qwen35",))
    pl.add_argument("--image-size", type=int, default=256)
    pl.add_argument(
        "--vtcm-bytes",
        type=int,
        default=V75_VTCM_TOTAL_BYTES,
        help="VTCM budget in bytes. The default is the v75 PART TOTAL, which is "
             "not what a process necessarily gets — pass the runtime's number "
             "when planning for real hardware.",
    )
    pl.add_argument("--order-policy", default="min_peak")
    pl.add_argument("--alloc-policy", default="largest_first")
    pl.add_argument("--print", dest="print_plan", action="store_true")
    pl.add_argument("--out", default="")
    pl.set_defaults(func=_cmd_plan)

    try:
        args = p.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

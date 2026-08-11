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


def _qdc_print_remaining_budget() -> None:
    """Printed before ANY submission attempt -- see `_cmd_test_qdc`. Never
    queries QDC: there is no such API on this account (job.py's own module
    docstring: `get_job_status` returns `state=None`, `get_jobs_list` lags
    over 30 minutes), so the only honest source is whatever the operator has
    recorded for themselves."""
    raw = os.environ.get(_QDC_BUDGET_ENV)
    if raw is None:
        print(
            f"remaining budget: unknown ({_QDC_BUDGET_ENV} is not set). "
            "Nothing here queries QDC for a remaining-minutes figure -- "
            "there is no reliable API for it on this account -- so set "
            f"{_QDC_BUDGET_ENV} yourself to have it printed here."
        )
        return
    print(f"remaining budget: {raw} minutes (from {_QDC_BUDGET_ENV})")


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

    if not job.wait(job_id):
        print(
            f"error: job {job_id} produced no results.xml within the wait cap -- "
            "a job with no results is a failure, never a pass",
            file=sys.stderr,
        )
        return 1

    log_dir = os.path.join(args.out, "qdc_logs")
    paths = job.fetch(job_id, log_dir)
    print(f"fetched {len(paths)} log file(s) to {log_dir}")

    return _qdc_check_results(job_id, paths)


class _QdcResultsError(Exception):
    """Raised by `_qdc_parse_results_xml` for any results.xml that must not
    be treated as a pass -- unparseable, or missing the attributes a JUnit
    report always carries. Caught by `_qdc_check_results`, never allowed to
    propagate past `_qdc_submit`."""


def _qdc_parse_results_xml(path: str) -> tuple[int, int, int]:
    """Parse a JUnit-style results.xml and return `(tests, failures,
    errors)` summed across every `<testsuite>` element. Raises
    `_QdcResultsError` on anything that is not a genuinely parseable report
    with real counts on it -- a truncated or non-XML file, a
    `<testsuites>`/`<testsuite>` tree with no testsuite elements at all, or
    a shape this function does not recognize -- so the caller never has to
    guess whether "zero" means "ran zero tests" or "could not even find the
    count".

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

    tests = failures = errors = 0
    for suite in suites:
        try:
            tests += int(suite.get("tests", "0"))
            failures += int(suite.get("failures", "0"))
            errors += int(suite.get("errors", "0"))
        except ValueError as e:
            raise _QdcResultsError(
                f"{path} has a non-integer tests/failures/errors attribute: {e}"
            ) from e
    return tests, failures, errors


def _qdc_check_results(job_id: int, paths: list[str]) -> int:
    """The parse `job.wait`/`job.fetch` never do. A device job whose
    TestLogs/results.xml merely *exists* proves nothing on its own -- this
    project's own named failure mode is a job that completed having run (or
    measured) NOTHING and still reported passing. Every one of the following
    must hold before this returns 0, each with its own distinct message so a
    real failure is diagnosable from which check tripped:

      - a results.xml was actually fetched, and it PARSES as XML;
      - it reports `tests > 0` -- zero tests is a failure, never a pass;
      - `failures == 0` and `errors == 0`;
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
        tests, failures, errors = _qdc_parse_results_xml(results_path)
    except _QdcResultsError as e:
        print(
            f"error: job {job_id}: could not parse results.xml as a JUnit "
            f"report -- a truncated or unparseable results file is a "
            f"failure, never a pass: {e}",
            file=sys.stderr,
        )
        return 1

    if tests == 0:
        print(
            f"error: job {job_id}: results.xml reports 0 tests -- a job "
            "that ran no tests is a failure, never a pass",
            file=sys.stderr,
        )
        return 1

    if failures != 0 or errors != 0:
        print(
            f"error: job {job_id}: results.xml reports {failures} failure(s) "
            f"and {errors} error(s) across {tests} test(s)",
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

    print(
        f"job {job_id}: {tests} test(s), 0 failures, 0 errors, "
        f"measurement lines present ({cycles_detail})"
    )
    return 0


def _cmd_test_qdc(args) -> int:
    """`--device qdc`: refuses without an explicit `--timeout-min`, always
    prints the (locally known, never queried) remaining budget before doing
    anything else, and requires `--yes` above `_QDC_YES_THRESHOLD_MIN` --
    every one of these guards runs before `_qdc_submit` ever touches the SDK,
    a credential, or the network."""
    if args.timeout_min is None:
        print(
            "error: --device qdc requires --timeout-min (1..240) -- a "
            "runaway job spends real, non-renewable minutes, and there is "
            "no default that could be right for every account.",
            file=sys.stderr,
        )
        return 2

    _qdc_print_remaining_budget()

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

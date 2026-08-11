"""hexlib command line.

The backend is a flag, never a code path inside a kernel: `--device sim` (the
default), `--device local`, `--device qdc` all run the same source, the same
harness, and produce the same result table.
"""
from __future__ import annotations

import argparse
import os
import sys

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

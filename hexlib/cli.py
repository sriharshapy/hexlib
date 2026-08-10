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
    if args.device != "sim":
        print(
            f"error: --device {args.device} is not implemented in the simulation "
            "path. The local and qdc backends arrive with the silicon-path plan.",
            file=sys.stderr,
        )
        return 2

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

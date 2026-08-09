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
from hexlib.result import is_ok
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
    print("Paste it into your pull request.")
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

    try:
        args = p.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

# hexlib/runtime/genentry.py
"""Emit each kernel's DSP entry point, and the dispatch table, from RunnerSpec.

WHY GENERATED WHEN harness.c IS HAND-WRITTEN. The harness DECIDES CORRECTNESS, so
a generated one would be a harness nobody read. An argument unpacker decides
nothing — it only marshals — and `RunnerSpec` already declares the mapping for the
file-based path. Generating from the same declaration is what stops the two
transports disagreeing about argument order, which is a class of bug that
produces a plausible wrong answer rather than a failure.

THE ESCAPE HATCH IS REAL. A kernel directory containing its own `dsp_entry.c`
keeps it; nothing is generated for that kernel. So a mapping the declaration
cannot express is written by hand, visibly, rather than by bending the
declaration.

KIND IDS ARE EXPLICIT AND ORDERED. They cross the wire, so they must not depend
on dict iteration order: a silent renumber sends every op to the wrong kernel.
Appending is safe; reordering is not.

WHAT `requires` CAN AND CANNOT VERIFY HERE. `hexlib_args` (see
`hexlib/runtime/skel/hexlib_dsp.h`) carries `dtype[HEXLIB_MAX_BUFS]` per buffer,
filled from the same `DTYPE_ID` table `hexlib/runtime/wire.py` uses to serialize
a tensor's dtype -- so a `("dtype", ...)` requirement is a real, reachable check
against that field. It carries no field at all for a permutation, a shape, or any
other host-side attribute, so a `("perm", ...)` requirement (or anything else not
representable in `ne`/`dtype`/`layout`) CANNOT be checked here: today it is
enforced only on the host, in `RunnerSpec.check_requires`, before the op is ever
put on the wire. Writing an `if` here that always passes would be a check that
protects nothing, so none is emitted for those keys -- only a comment saying so.
"""
from __future__ import annotations

import os

from hexlib.exec.runner import RunnerSpec, Scalar
from hexlib.runtime.wire import DTYPE_ID

KIND_ID: dict[str, int] = {
    "add": 1,
    "cast": 2,
    "layernorm": 3,
    "matmul": 4,
    "matmul_epilogue": 5,
    "patchify": 6,
    "reshape": 7,
    "rope_2d": 8,
    "scale": 9,
    "softmax": 10,
    "transpose": 11,
}

# hexlib_args C types. Keyed by the same wire-dtype strings as
# `hexlib.exec.runner.WIRE_DTYPE` ("int32", not "i32" -- a mismatch here would
# KeyError the first time a kernel declares an int32 input or output, silently
# never today because no current spec uses it).
_CTYPE = {"fp16": "hexlib_hf", "fp32": "float", "int32": "int"}

# C types for values packed into the `a->params` blob, matching
# `Scalar.ctype` / `RunnerSpec._STRUCT_CODE` ('i' -> int, 'f' -> float). Both
# are 4 bytes, so indexing by element (not byte) is safe even when scalars of
# different ctypes are packed back to back in the same blob.
_PARAM_CTYPE = {"int": "int", "float": "float"}


class GenError(Exception):
    pass


def _scalar_expr(sc: Scalar, spec: RunnerSpec, param_index: int) -> str:
    """The C expression for one scalar -- the DSP-side derivation."""
    src = sc.source
    if src.startswith("attr:"):
        ctype = _PARAM_CTYPE[sc.ctype]
        return f"((const {ctype} *) a->params)[{param_index}]"
    if src.startswith("numel:"):
        i = int(src.split(":", 1)[1])
        # From the tensor's OWN extent, not from a number the host asserted.
        return f"(int) (a->ne[{i}][0] * a->ne[{i}][1] * a->ne[{i}][2] * a->ne[{i}][3])"
    if src.startswith("dim:"):
        _, i, axis = src.split(":")
        return f"(int) a->ne[{i}][{axis}]"
    raise GenError(f"unknown scalar source {src!r} in spec for {spec.kind}")


def _requires_check(key: str, want, out_idx: int) -> str:
    """One `requires` clause as C, or an honest comment if it cannot be one.

    Only `("dtype", <wire-dtype>)` maps onto a field `hexlib_args` actually
    carries: `a->dtype[out_idx]`, filled from the same `DTYPE_ID` table the
    host used to serialize the tensor. Everything else (`perm`, and anything
    not representable in `ne`/`dtype`/`layout`) has no wire representation at
    all, so it is documented as unverified rather than given a check that
    cannot fail.
    """
    if key == "dtype":
        want_id = DTYPE_ID[want]
        return (
            f"    /* requires {key} == {want!r}: checked -- a->dtype[{out_idx}] "
            f"mirrors hexlib.runtime.wire.DTYPE_ID, filled in by the host per "
            f"buffer. */\n"
            f"    if (a->dtype[{out_idx}] != {want_id}u) "
            f"return HEXLIB_DSP_ERR_REQUIRES;"
        )
    # HONEST GAP: hexlib_args has no field for this key. buf/ne/dtype/layout are
    # all per-buffer tensor properties; `perm` (and anything else outside that
    # set) is an op-level attribute the wire format never carries down to the
    # kernel entry. So this cannot be verified on the DSP today -- enforcement
    # lives only in RunnerSpec.check_requires, on the host, before the op is
    # ever encoded. HEXLIB_DSP_ERR_REQUIRES is the status this would return if
    # the wire format grows a field for it; until then, no check is emitted,
    # because a check that cannot fail is not a check.
    return (
        f"    /* requires {key} == {want!r}: NOT VERIFIED ON THE DSP -- "
        f"hexlib_args carries no field for {key!r}. Enforced only on the host "
        f"today (RunnerSpec.check_requires). Would return HEXLIB_DSP_ERR_REQUIRES "
        f"if the wire format ever carries this. */"
    )


def emit_entry(name: str, spec: RunnerSpec) -> str:
    """The C adapter from hexlib_args to the kernel's real signature.

    NOTE `spec.kernel_dir` is a PATH ("kernels/scale_fp16"), so the function name
    is its basename. And `spec.inputs` is a tuple of DTYPES, not names -- so each
    input is cast to its own type, which matters for `cast` (fp32 in, fp16 out):
    casting an fp32 buffer to hexlib_hf* would halve every stride silently.
    """
    fn = os.path.basename(spec.kernel_dir)   # "scale_fp16"
    n_in = len(spec.inputs)
    n_buf = n_in + 1                         # inputs + one output
    out_idx = n_in

    args: list[str] = []
    for i, in_dtype in enumerate(spec.inputs):
        args.append(f"(const {_CTYPE[in_dtype]} *) a->buf[{i}]")
    args.append(f"({_CTYPE[spec.out_dtype]} *) a->buf[{out_idx}]")

    param_index = 0
    for sc in spec.scalars:
        expr = _scalar_expr(sc, spec, param_index)
        if sc.source.startswith("attr:"):
            param_index += 1
        args.append(expr)

    checks = [
        f"    if (a->n_buf != {n_buf}) return HEXLIB_DSP_ERR_INVAL_PARAMS;",
    ]
    for i in range(n_buf):
        checks.append(f"    if (!a->buf[{i}]) return HEXLIB_DSP_ERR_INVAL_PARAMS;")

    # `requires` is enforced HERE as well as on the host where it is genuinely
    # checkable -- see `_requires_check` for exactly which keys that is, and the
    # module docstring for why the rest are documented rather than faked.
    for key, want in spec.requires:
        checks.append(_requires_check(key, want, out_idx))

    body = ",\n                ".join(args)
    return f'''/* GENERATED by hexlib/runtime/genentry.py -- do not edit.
 * Source of truth: hexlib/exec/runner.py SPECS[{name!r}].
 * A hand-written dsp_entry.c in this kernel's own directory takes precedence
 * over this generated file.
 */
#include "hexlib_dsp.h"
#include "kernel_api.h"

int {fn}_entry(const hexlib_args *a) {{
{chr(10).join(checks)}
    {fn}({body});
    return HEXLIB_DSP_OK;
}}
'''


def emit_table(specs: dict[str, RunnerSpec]) -> str:
    rows = []
    externs = []
    for name in sorted(specs):
        fn = os.path.basename(specs[name].kernel_dir)
        externs.append(f"extern int {fn}_entry(const hexlib_args *);")
        rows.append(f'    {{ {KIND_ID[name]}u, "{name}", {fn}_entry }},')
    return f'''/* GENERATED by hexlib/runtime/genentry.py -- do not edit.
 *
 * Generated rather than hand-maintained so that adding a kernel touches no
 * shared file and parallel contributions cannot conflict in a central registry.
 */
#include "hexlib_dsp.h"

{chr(10).join(externs)}

const struct hexlib_kernel_entry hexlib_kernel_table[] = {{
{chr(10).join(rows)}
}};

const uint32_t hexlib_kernel_table_len =
    sizeof(hexlib_kernel_table) / sizeof(hexlib_kernel_table[0]);
'''


def generate(repo_root: str, out_dir: str) -> list[str]:
    """Emit entries for every kernel that does not hand-write its own.

    `spec.kernel_dir` is already repo-relative ("kernels/scale_fp16"), so it is
    joined to the REPO root, not to a kernels root -- joining it to `.../kernels`
    would produce `kernels/kernels/scale_fp16` and silently find nothing, which
    would emit an empty dispatch table rather than an error.
    """
    from hexlib.exec.runner import SPECS

    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    used: dict[str, RunnerSpec] = {}
    for name, spec in SPECS.items():
        kdir = os.path.join(repo_root, spec.kernel_dir)
        if not os.path.isdir(kdir):
            continue
        used[name] = spec
        fn = os.path.basename(spec.kernel_dir)
        if os.path.isfile(os.path.join(kdir, "dsp_entry.c")):
            continue  # hand-written wins
        path = os.path.join(out_dir, f"{fn}_entry.c")
        with open(path, "w", encoding="utf-8") as f:
            f.write(emit_entry(name, spec))
        written.append(path)

    # AN EMPTY TABLE IS AN ERROR, NOT AN EMPTY SUCCESS. It would link cleanly and
    # then answer every op with ERR_NO_KERNEL at run time, which reads as "the
    # kernel is broken" rather than "the generator was pointed at the wrong
    # directory". This project has been bitten four times by absence reported as
    # success; a wrong `repo_root` is exactly that shape.
    if not used:
        raise GenError(
            f"no kernel directory found under {repo_root!r} for any of "
            f"{sorted(SPECS)} — kernel_dir is repo-relative "
            f"('kernels/scale_fp16'), so pass the REPO root"
        )

    path = os.path.join(out_dir, "hexlib_kernel_table.c")
    with open(path, "w", encoding="utf-8") as f:
        f.write(emit_table(used))
    written.append(path)
    return written

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

`KIND_ID` BELOW IS THE ONE SOURCE OF TRUTH FOR THOSE IDS, AND THE COPIES ARE
TEST-BOUND. It is hand-maintained (nothing derives it from the op registry,
whatever an earlier draft of the design spec claimed), and there are two other
places the same numbers appear: the generated DSP dispatch table, which
`emit_table` below writes straight from this dict, and one `#define
HEXLIB_KIND_SCALE 9u` in `hexlib/runtime/host/main.c`, which is a hand-copy
because the host binary has no generated header to read. Three tests hold that
together, because a wrong id is not a compile error and not a crash:
`test_host_source.py::test_the_hosts_scale_kind_id_is_the_same_number_the_dsp_
dispatches_on` binds main.c's `#define` to this dict (mutating either fails),
`test_runtime_genentry.py::test_the_shipped_kind_ids_never_move` freezes the 11
shipped values against a renumber, and
`::test_every_kind_a_COMPILED_PLAN_can_contain_has_a_wire_id` checks the table
covers every kind a plan can actually contain (the registry's 13 minus
`fuse.FUSABLE_ACTS`, which fusion absorbs into `matmul_epilogue`).

DTYPES ARE CHECKED PER BUFFER, ALWAYS -- NOT ONLY WHEN `requires` MENTIONS THEM.
`hexlib_args` carries `dtype[HEXLIB_MAX_BUFS]`, filled from the same `DTYPE_ID`
table `hexlib/runtime/wire.py` serializes with, and each buffer is about to be
cast to the C type this spec DECLARES for it. So each cast is guarded by the
matching check: a batch declaring a `scale` input as fp32 would otherwise be read
through `const hexlib_hf *` at half stride -- half the tensor, HEXLIB_DSP_OK, a
plausible wrong answer. That hazard is named in `emit_entry`'s own docstring and
was previously guarded on this side only for whichever single buffer a `requires`
entry happened to mention.

WHAT `requires` CAN AND CANNOT VERIFY HERE. A `("dtype", ...)` requirement lands
on the per-buffer check just described -- and, because the only serializer there
is (`hexlib/exec/dsp.py`) fills that field from `spec.out_dtype`, the check
compares the spec with itself on that path and passes by construction. It is
genuinely reachable from a hand-built batch (main.c, `run_raw`, a future
planner), which is why it is emitted; what it CANNOT do is police the attr a
caller asked for. That is `RunnerSpec.check_requires`'s job, on the host, and
both transports now call it. `hexlib_args` carries no field at all for a
permutation, a shape, or any other op-level attribute, so a `("perm", ...)`
requirement cannot be checked here in any form: writing an `if` that always
passes would be a check that protects nothing, so none is emitted -- only a
comment saying so.
"""
from __future__ import annotations

import os
from typing import Sequence

from hexlib.exec.runner import RunnerSpec, Scalar, WIRE_RAW
from hexlib.runtime.wire import DTYPE_ID, LAYOUT_ID

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
    # KIND_ID IS KEYED BY KERNEL VARIANT, NOT BY OP KIND, from here on. The first
    # eleven happen to coincide because each of those op kinds had at most one
    # kernel; `transpose_hd` is the first that does not. The encoder's 60
    # transposes are two different permutations needing two different kernels
    # (see runner.SPECS), and THE WIRE CARRIES NO ATTRS -- the DSP gets an id and
    # a buffer list, with no perm to branch on. So the variant has to be resolved
    # on the host, by `runner.select`, and then named on the wire by its own id.
    # Appending is safe; reordering is not.
    "transpose_hd": 12,
}

# hexlib_args C types. Keyed by the same wire-dtype strings as
# `hexlib.exec.runner.WIRE_DTYPE` AND `hexlib.runtime.wire.DTYPE_ID` -- all
# three spell int32 "int32". `DTYPE_ID` spelled it "i32" until this was fixed,
# which meant a spec declaring an int32 input was accepted by `RunnerSpec`,
# refused by `pack_batch` as an unknown dtype, and a KeyError here at generate
# time. test_runtime_wire.py binds the three key sets so they cannot drift again.
_CTYPE = {"fp16": "hexlib_hf", "fp32": "float", "int32": "int"}

# BLOCK-QUANTIZED BUFFERS ARE HANDED OVER AS BYTES, deliberately. A q4_0 weight
# is a stream of 18-byte blocks (an fp16 scale then 32 4-bit values) and there is
# no C scalar type for one element of it -- so the entry does not invent one. The
# kernel receives `const unsigned char *` and the block layout is its business,
# which is also why `hexlib_dsp.h` enumerates `q4_0_repacked` as a LAYOUT: the
# guard `_layout_check` emits is what stops an un-repacked weight being read as a
# repacked one. Casting these to `hexlib_hf *` instead would compile fine and
# read the fp16 scale bytes as data.
_RAW_CTYPE = "unsigned char"

# C types for values packed into the `a->params` blob, matching
# `Scalar.ctype` / `RunnerSpec._STRUCT_CODE` ('i' -> int, 'f' -> float). Both
# are 4 bytes, so indexing by element (not byte) is safe even when scalars of
# different ctypes are packed back to back in the same blob.
_PARAM_CTYPE = {"int": "int", "float": "float"}


class GenError(Exception):
    pass


def _comment(text: str) -> str:
    """One block comment, wrapped, at the entry body's indent. Generated code is
    still read by people -- a 400-column comment line is not."""
    import textwrap

    lines = textwrap.wrap(" ".join(text.split()), width=72)
    if len(lines) == 1:
        return f"    /* {lines[0]} */"
    body = "\n".join(f"     * {ln}" for ln in lines[1:])
    return f"    /* {lines[0]}\n{body} */"


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


def _dtype_check(idx: int, dtype: str, role: str) -> str:
    """The guard for ONE buffer, emitted for every buffer the entry casts.

    `a->dtype[idx]` is filled by `skel_dispatch.c` from the tensor's own dtype
    field, which the host serialized through the same `DTYPE_ID` table imported
    here -- so this compares the dtype the BATCH declared with the dtype this
    entry is about to cast the pointer to. Without it, a batch declaring an
    fp32 buffer where the kernel wants fp16 is read (or written) at half
    stride, over half the tensor, and returns HEXLIB_DSP_OK.
    """
    if dtype in WIRE_RAW:
        detail = (
            f"{role} buf[{idx}] is handed over as {_RAW_CTYPE} * -- a stream of "
            f"block-quantized {dtype} data -- so the batch must have declared it "
            f"{dtype} ({DTYPE_ID[dtype]} in hexlib.runtime.wire.DTYPE_ID). A "
            f"dense buffer arriving here would be read as blocks: its values "
            f"decoded as 4-bit fields against scales that are really data."
        )
    else:
        detail = (
            f"{role} buf[{idx}] is cast to {_CTYPE[dtype]} *, so the batch must "
            f"have declared it {dtype} ({DTYPE_ID[dtype]} in "
            f"hexlib.runtime.wire.DTYPE_ID). Casting a wider or narrower dtype "
            f"would silently halve or double every stride."
        )
    return (
        _comment(detail)
        + f"\n    if (a->dtype[{idx}] != {DTYPE_ID[dtype]}u) "
        f"return HEXLIB_DSP_ERR_REQUIRES;"
    )


def _layout_check(idx: int, layout: str, role: str) -> str:
    """The layout guard for ONE buffer, emitted beside its dtype guard.

    `a->layout[idx]` is filled by `skel_dispatch.c` from the tensor's own
    layout field, serialized through the same `LAYOUT_ID` table imported here.
    Nothing checked it before: `main.c` wrote a bare literal `0` with a comment
    for the binding, `grep -c layout hexlib/tests/test_host_source.py` was 0,
    and `--self-test` printed `PASS (4100 values, bit-exact)` regardless of
    what the batch declared.

    THIS IS THE CHECK THAT MAKES THE ENUM WORTH HAVING. `hexlib_dsp.h`'s own
    header says the layout is enumerated rather than ne/nb strides so that
    "un-repacked weights are a plan-time error rather than silent corruption" --
    and `LAYOUT_ID` already carries `q4_0_repacked`, the matmul weight layout.
    Without a guard here that sentence describes an intention, not a mechanism:
    a q4_0-repacked weight buffer handed to a row-major kernel is read as
    row-major fp16 and returns HEXLIB_DSP_OK with a plausible wrong answer, the
    same failure mode the per-buffer dtype check exists to stop.
    """
    return (
        _comment(
            f"{role} buf[{idx}] is addressed as {layout}, so the batch must "
            f"have declared it {layout} ({LAYOUT_ID[layout]} in "
            f"hexlib.runtime.wire.LAYOUT_ID). A differently-laid-out buffer of "
            f"the same dtype and byte count passes every other check here."
        )
        + f"\n    if (a->layout[{idx}] != {LAYOUT_ID[layout]}u) "
        f"return HEXLIB_DSP_ERR_REQUIRES;"
    )


def _requires_check(key: str, want, spec: RunnerSpec, out_idx: int) -> str:
    """One `requires` clause as C, or an honest comment if it cannot be one.

    Only `("dtype", <wire-dtype>)` maps onto a field `hexlib_args` actually
    carries, and it is already covered: `_dtype_check` emits a guard for EVERY
    buffer from the spec's own declared dtypes, so the clause for `out_idx` is
    the same condition this would emit. Rather than emit it twice, this points
    at it.

    WHICH BUFFER A REQUIREMENT IS ABOUT CANNOT BE SAID. This used to assume
    `out_idx` for every key -- correct for `cast`, whose requirement is about
    its output, and silently wrong for any future dtype requirement about an
    INPUT, which would have inspected the output's dtype instead. `requires`
    has no place to name a buffer, so the ambiguous case is refused at generate
    time instead of guessed at: a dtype requirement that is not the spec's own
    declared output dtype is either about an input (unexpressible) or a
    contradiction (it would refuse every batch the host serializer can build,
    since that stamps the output's dtype from `spec.out_dtype`).

    Everything else (`perm`, and anything not representable in
    `ne`/`dtype`/`layout`) has no wire representation at all, so it is
    documented as unverified rather than given a check that cannot fail.
    """
    if key == "dtype":
        if want != spec.out_dtype:
            raise GenError(
                f"{spec.kind}: requires ('dtype', {want!r}) but the spec's "
                f"out_dtype is {spec.out_dtype!r}. `requires` cannot say WHICH "
                f"buffer a dtype requirement is about, and assuming the output "
                f"would emit a check that either inspects the wrong buffer or "
                f"refuses every batch the host can build. Declare the dtype on "
                f"the buffer itself (inputs=/out_dtype=) instead."
            )
        return _comment(
            f"requires {key} == {want!r}: checked above, by the "
            f"a->dtype[{out_idx}] guard emitted for this kernel's declared "
            f"output dtype -- the same condition, from the same DTYPE_ID table. "
            f"NOTE it cannot fail through hexlib/exec/dsp.py, which fills that "
            f"field from spec.out_dtype: only a hand-built batch can violate it. "
            f"The CALLER'S attr is policed on the host, in "
            f"RunnerSpec.check_requires."
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
        ctype = _RAW_CTYPE if in_dtype in WIRE_RAW else _CTYPE[in_dtype]
        args.append(f"(const {ctype} *) a->buf[{i}]")
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

    # EVERY buffer's declared dtype, not just whichever one `requires` mentions
    # -- see `_dtype_check` and the module docstring. After the count and null
    # checks, because `a->dtype[i]` means nothing for a buffer the batch did not
    # supply.
    for i, in_dtype in enumerate(spec.inputs):
        checks.append(_dtype_check(i, in_dtype, "input"))
    checks.append(_dtype_check(out_idx, spec.out_dtype, "output"))

    # AND EVERY BUFFER'S DECLARED LAYOUT, for the same reason and on the same
    # terms -- see `_layout_check`. `spec.buf_layouts()` defaults every buffer to
    # row_major, so this is a no-op for every kernel shipped today and becomes
    # load-bearing the moment a q4_0_repacked weight appears.
    for i, layout in enumerate(spec.buf_layouts()):
        checks.append(_layout_check(i, layout, "input" if i < len(spec.inputs)
                                    else "output"))

    # `requires` is enforced HERE as well as on the host where it is genuinely
    # checkable -- see `_requires_check` for exactly which keys that is, and the
    # module docstring for why the rest are documented rather than faked.
    for key, want in spec.requires:
        checks.append(_requires_check(key, want, spec, out_idx))

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


def generate(repo_root: str, out_dir: str,
             expect: Sequence[str] | None = None) -> list[str]:
    """Emit entries for every kernel that does not hand-write its own.

    `spec.kernel_dir` is already repo-relative ("kernels/scale_fp16"), so it is
    joined to the REPO root, not to a kernels root -- joining it to `.../kernels`
    would produce `kernels/kernels/scale_fp16` and silently find nothing, which
    would emit an empty dispatch table rather than an error.

    `expect` is the set of op kinds whose kernel directory MUST be present,
    defaulting to every kind with a `RunnerSpec`. A tree missing any of them is
    an incomplete checkout, not a smaller build -- see the partial-table comment
    below. Pass a narrower tuple only from a caller that genuinely holds a
    subset and says so (the unit tests in test_runtime_genentry.py, which build
    one-kernel trees in tmp dirs).
    """
    from hexlib.exec.runner import SPECS

    if expect is None:
        expect = tuple(SPECS)
    else:
        unknown = sorted(set(expect) - set(SPECS))
        if unknown:
            raise GenError(
                f"expect names {unknown}, which has no RunnerSpec; `expect` "
                f"narrows a claim about what is on disk, it cannot invent a "
                f"kernel. Known kinds: {sorted(SPECS)}"
            )

    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    used: dict[str, RunnerSpec] = {}
    for name, spec in SPECS.items():
        kdir = os.path.join(repo_root, spec.kernel_dir)
        if not os.path.isdir(kdir):
            continue
        used[name] = spec

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

    # AND A PARTIAL TABLE IS THE SAME BUG ONE STEP DOWN. Finding SOME kernels
    # used to be enough: the table came out missing those rows, no error was
    # raised, and the affected ops answered ERR_NO_KERNEL at run time -- which
    # reads as a broken kernel rather than an incomplete tree. Checked BEFORE
    # anything is written, so a refused run leaves no half-generated table for a
    # build to pick up.
    missing = sorted(set(expect) - set(used))
    if missing:
        raise GenError(
            f"kernel directory missing under {repo_root!r} for {missing} "
            f"(expected {sorted(expect)}, found {sorted(used)}). Generating "
            f"anyway would emit a dispatch table without those rows, which "
            f"links cleanly and then answers ERR_NO_KERNEL at run time."
        )

    for name, spec in used.items():
        kdir = os.path.join(repo_root, spec.kernel_dir)
        fn = os.path.basename(spec.kernel_dir)
        if os.path.isfile(os.path.join(kdir, "dsp_entry.c")):
            continue  # hand-written wins
        path = os.path.join(out_dir, f"{fn}_entry.c")
        with open(path, "w", encoding="utf-8") as f:
            f.write(emit_entry(name, spec))
        written.append(path)

    path = os.path.join(out_dir, "hexlib_kernel_table.c")
    with open(path, "w", encoding="utf-8") as f:
        f.write(emit_table(used))
    written.append(path)
    return written

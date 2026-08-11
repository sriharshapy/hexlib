# hexlib/tests/test_skel_dispatch_source.py
"""The batch walk and the FastRPC entry points, asserted against the source.

Source assertions, not behavioural ones -- the behavioural run is Task 8's. They
exist because the invariants here are easy to satisfy with a log line instead of
a real guard, and a log line passes on the simulator (host and DSP share an
address space there) right up until it fails on silicon. Every check below that
looks for a status constant requires it inside a `return`, an assignment to a
status field, or a genuine call to the shared header-writer -- never merely
"the token appears somewhere in the file", which a FARF-only downgrade would
still satisfy.

Comments are blanked out of both fixtures before any check runs, in both
directions: a mutation cannot satisfy a positive check ("X must be assigned")
by demoting the assignment to a comment, and a mutation cannot trip a negative
check ("X must not appear here") merely by mentioning X in prose -- which
happened during development of this file (a comment in the invoke-before-start
refusal that named `hexlib_dispatch_batch` in prose briefly failed
test_invoke_before_start_is_refused for exactly that reason).

THE SLICER IS SHARED, NOT COPIED. This file carried its own private
`_strip_comments`/`_function_body`/`_brace_block` -- the third copy of the
slicer `hexlib/tests/csource.py` was written to consolidate, and a WEAKER one:
its `_strip_comments` DELETED comment text rather than blanking it, so every
offset in the stripped text was shifted relative to the real file and no
offset could be reported back against the source. Migrated to `csource`
(`code_only` for the fixtures, `function_body`/`block_from` for the slices),
which keeps the same-length blanking property. See csource.py's own module
docstring for the full history, including the payload-check hole that stripping
comments only at the BOUNDARIES left open.
"""
import pathlib
import re

import pytest

from hexlib.tests.csource import block_from as _block_from
from hexlib.tests.csource import code_only as _code_only
from hexlib.tests.csource import function_body as _function_body

DISPATCH = pathlib.Path("hexlib/runtime/skel/skel_dispatch.c")
SKEL = pathlib.Path("hexlib/runtime/skel/skel.c")


@pytest.fixture(scope="module")
def d():
    return _code_only(DISPATCH.read_text())


@pytest.fixture(scope="module")
def s():
    return _code_only(SKEL.read_text())


def test_the_response_is_written_before_any_op_runs(d):
    """The magic and a NON-OK status go down first, so a batch that dies
    partway leaves a readable failure rather than a zero-filled buffer the
    host would have to interpret. Scoped to hexlib_dispatch_batch's own body
    -- not just file order -- so this cannot pass merely because the
    header-writer helper happens to be defined above the dispatcher in the
    file, regardless of what the dispatcher itself does first."""
    body = _function_body(d, "hexlib_dispatch_batch")
    assert body.index("hexlib_write_rsp_hdr") < body.index("hexlib_kernel_table")


def test_status_is_never_left_as_zero(d):
    """HEXLIB_DSP_OK is 1, not 0 (hexlib_dsp.h), so a response buffer that is
    never written cannot read as success. The dispatcher's own default must
    be an actual call passing HEXLIB_DSP_ERR_INTERNAL to the header-writer --
    not a comment or a FARF line naming the constant -- and it must happen
    before the magic/version checks, so every early return after it still
    leaves a specific status rather than reverting to a zeroed buffer."""
    m = re.search(r"hexlib_write_rsp_hdr\s*\([^;]*HEXLIB_DSP_ERR_INTERNAL", d)
    assert m, "no call to hexlib_write_rsp_hdr passes HEXLIB_DSP_ERR_INTERNAL"
    assert m.start() < d.index("HEXLIB_DSP_ERR_BAD_MAGIC")


def test_magic_and_version_are_checked_before_the_offsets_are_used(d):
    body = _function_body(d, "hexlib_dispatch_batch")
    assert body.index("HEXLIB_DSP_ERR_BAD_MAGIC") < body.index("off_bufs")


def test_total_size_is_checked_against_the_actual_length(d):
    """Not just "the tokens appear somewhere" -- an actual `if` comparing
    hdr.total_size to len, whose body reports HEXLIB_DSP_ERR_TRUNCATED."""
    body = _function_body(d, "hexlib_dispatch_batch")
    m = re.search(r"if\s*\(\s*hdr\.total_size\s*!=\s*len\s*\)\s*\{", body)
    assert m, "no guard comparing hdr.total_size against the actual length"
    guard = _block_from(body, m.end() - 1)
    assert "HEXLIB_DSP_ERR_TRUNCATED" in guard


def test_pcycle_brackets_only_the_kernel_call(d):
    """Harness and RPC overhead is roughly constant, so including it
    manufactures ratios out of nothing. Same counter and same placement as
    hexlib.sim. Scoped to hexlib_dispatch_batch's own body: the read helper's
    definition contains the literal text "hexlib_read_pcycle" too (it is the
    function's own name), so an unscoped first/last index() over the whole
    file would anchor "lo" on that definition instead of the first real call,
    and everything from the definition onward -- including the tensor-resolve
    call -- would land "between" the two markers, defeating the very check
    this test exists to make. Slicing the dispatcher's body first removes the
    definition from consideration entirely, and the `()` (no-arg call syntax,
    vs. the definition's `(void)`) requirement in the pattern is a second,
    independent guard against the same confusion."""
    assert "c15:14" in d or "PCYCLE" in d
    body = _function_body(d, "hexlib_dispatch_batch")
    calls = [m.start() for m in re.finditer(r"hexlib_read_pcycle\s*\(\s*\)", body)]
    assert len(calls) >= 2, "expected at least a before/after pair of calls"
    lo, hi = calls[0], calls[-1]
    between = body[lo:hi]
    assert "->fn(" in between, "the kernel call must be inside the bracket"
    assert "hexlib_tensors_resolve" not in between, "resolution must be outside it"
    assert "hexlib_bufs_map" not in between, "buffer mapping must be outside it"
    assert "hexlib_write_rsp_hdr" not in between, "the response write must be outside it"


def test_an_unknown_kind_is_refused(d):
    """Not just "the constant appears" -- the null-kernel-pointer guard must
    itself assign HEXLIB_DSP_ERR_NO_KERNEL to the per-op result AND to the
    batch-level status, which is what actually stops the batch and reports
    the refusal on the wire rather than silently skipping the op."""
    body = _function_body(d, "hexlib_dispatch_batch")
    m = re.search(r"if\s*\(\s*!\s*k\s*\)\s*\{", body)
    assert m, "no null-kernel-pointer guard (`if (!k)`) found"
    guard = _block_from(body, m.end() - 1)
    assert re.search(r"results\[i\]\.status\s*=\s*HEXLIB_DSP_ERR_NO_KERNEL", guard)
    assert re.search(r"batch_status\s*=\s*HEXLIB_DSP_ERR_NO_KERNEL", guard)
    assert "break" in guard, "an unknown kind must stop the batch, not continue it"


def test_vtcm_reclaim_is_reported_not_ignored(d):
    """The release callback in skel_vtcm.c only RECORDS a reclaim request
    (`vtcm_needs_release = 1`); it deliberately does not release memory a
    batch in flight is still using. This dispatcher is what must notice the
    flag, stop at an op boundary, actually give the VTCM back (call
    hexlib_vtcm_release), and report HEXLIB_DSP_ERR_VTCM_RECLAIMED as the
    batch status -- not just mention the flag or the constant somewhere."""
    body = _function_body(d, "hexlib_dispatch_batch")
    m = re.search(r"if\s*\(\s*ctx->vtcm_needs_release\s*\)\s*\{", body)
    assert m, "no check of ctx->vtcm_needs_release inside the dispatcher"
    guard = _block_from(body, m.end() - 1)
    assert "hexlib_vtcm_release(" in guard, "must actually release VTCM, not just stop"
    assert re.search(r"batch_status\s*=\s*HEXLIB_DSP_ERR_VTCM_RECLAIMED", guard)
    assert "break" in guard, "must stop at the op boundary, not continue"


def test_invoke_before_start_is_refused(s):
    """An invoke before start must not run any op, and the response must
    actually carry HEXLIB_DSP_ERR_NOT_STARTED as its status -- not merely a
    FARF line naming the constant while the batch runs anyway. This is the
    task's known loose end: a version that logs the constant and then
    dispatches the batch regardless must fail this test."""
    body = _function_body(s, "hexlib_iface_invoke")
    m = re.search(r"if\s*\(\s*!\s*ctx->started\s*\)\s*\{", body)
    assert m, "hexlib_iface_invoke does not guard on ctx->started"
    guard = _block_from(body, m.end() - 1)
    assert re.search(
        r"hexlib_write_rsp_hdr\s*\([^;]*HEXLIB_DSP_ERR_NOT_STARTED", guard
    ), "the refusal must write NOT_STARTED into the response, not just log it"
    assert "hexlib_dispatch_batch" not in guard, "invoke-before-start must not run any op"


def test_hwinfo_reports_the_acquired_vtcm_size(s):
    """The size on the wire must be READ OUT of the session context that
    skel_vtcm.c filled in from HAP_compute_res, never a constant.

    `"vtcm_size" in s` was satisfied by the qaic-generated OUT-PARAMETER's own
    name in hexlib_iface_hwinfo's signature, and the negative half banned only
    one spelling of one constant -- so `*vtcm_size = (uint64)(8*1024*1024);`
    passed both. The check is now the assignment itself: whatever
    hexlib_iface_hwinfo writes through that pointer must be derived from
    ctx->vtcm_size, which is the only value the acquisition path ever sets."""
    body = _function_body(s, "hexlib_iface_hwinfo")
    m = re.search(r"\*\s*vtcm_size\s*=\s*([^;]+);", body)
    assert m, "hexlib_iface_hwinfo must write something through *vtcm_size"
    rhs = m.group(1)
    assert "ctx->vtcm_size" in rhs, (
        f"hwinfo must report the ACQUIRED size (ctx->vtcm_size, set by "
        f"skel_vtcm.c from HAP_compute_res), not `{rhs.strip()}` -- the part "
        f"total is not the usable budget, in any spelling"
    )
    assert "8388608" not in s, "hwinfo must report what was acquired, not a constant"


# Each qaic entry point, paired with the one thing it must actually DO. The
# entry points are thin by design -- they exist to delegate -- so the call each
# one delegates to IS its whole content, and a body that does not contain it is
# a stub regardless of what it returns.
_IFACE_DELEGATIONS = {
    "hexlib_iface_open": r"\*\s*handle\s*=",
    "hexlib_iface_close": r"\bhexlib_vtcm_free\s*\(",
    "hexlib_iface_start": r"\bhexlib_vtcm_alloc\s*\(",
    "hexlib_iface_stop": r"\bhexlib_vtcm_free\s*\(",
    "hexlib_iface_mmap": r"\bhexlib_bufs_register\s*\(",
    "hexlib_iface_munmap": r"\bhexlib_bufs_unregister\s*\(",
    "hexlib_iface_hwinfo": r"__HEXAGON_ARCH__",
    "hexlib_iface_invoke": r"\bhexlib_dispatch_batch\s*\(",
}


def test_skel_defines_the_iface_symbols_qaic_expects(s):
    """Each symbol must be a real DEFINITION that does its own job -- not
    merely a token present in the file.

    THIS WAS EIGHT `sym in s` CHECKS, AND EACH WAS SATISFIED BY THAT
    FUNCTION'S OWN SIGNATURE. Gutting every body in skel.c to `return
    AEE_SUCCESS;` passed all eight: the names were still there, on the empty
    shells. `function_body` raising on a removed or renamed symbol covers the
    presence half properly; the delegation table above covers the "and it
    still does something" half, one required call per entry point."""
    for sym, required in _IFACE_DELEGATIONS.items():
        body = _function_body(s, sym)   # raises if the definition is gone
        assert re.search(required, body), (
            f"{sym}() is defined but does not {required!r} -- a FastRPC entry "
            f"point that returns AEE_SUCCESS without delegating is a stub, and "
            f"a stub reports success for work that never happened"
        )

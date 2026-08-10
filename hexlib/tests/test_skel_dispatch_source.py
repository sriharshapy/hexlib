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

Comments are stripped from both fixtures before any check runs, in both
directions: a mutation cannot satisfy a positive check ("X must be assigned")
by demoting the assignment to a comment, and a mutation cannot trip a negative
check ("X must not appear here") merely by mentioning X in prose -- which
happened during development of this file (a comment in the invoke-before-start
refusal that named `hexlib_dispatch_batch` in prose briefly failed
test_invoke_before_start_is_refused for exactly that reason).

`_function_body()` is adapted from `hexlib/tests/test_skel_bufs_source.py`
(Task 4), which established the pattern for exactly this reason: whole-file
substring checks can't tell a real guard from a comment, and can't isolate ONE
of several return sites being downgraded while the others stay real.
"""
import pathlib
import re

import pytest

DISPATCH = pathlib.Path("hexlib/runtime/skel/skel_dispatch.c")
SKEL = pathlib.Path("hexlib/runtime/skel/skel.c")


def _strip_comments(text):
    """Remove /* ... */ and // ... comments, replacing each with nothing (not
    whitespace) so a comment can never contribute a stray brace to the
    depth-counting slicer below, and so a name mentioned only in prose can
    never satisfy -- or spuriously trip -- a code-level check."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return text


@pytest.fixture(scope="module")
def d():
    return _strip_comments(DISPATCH.read_text())


@pytest.fixture(scope="module")
def s():
    return _strip_comments(SKEL.read_text())


def _function_body(src, name):
    """Slice the text of a C function from its signature to its matching
    closing brace, by simple brace-depth counting. Good enough for this
    project's straight-line C; not a general C parser."""
    m = re.search(rf"\b{re.escape(name)}\s*\([^;{{]*\)\s*\{{", src)
    assert m, f"could not find the definition of {name}() in the source"
    start = m.end() - 1  # position of the opening brace
    depth = 0
    for i in range(start, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
    raise AssertionError(f"unbalanced braces while slicing {name}()")


def _brace_block(text, open_brace_idx):
    """Given the index of an opening '{', return the text up to and including
    its matching closing '}'."""
    depth = 0
    for i in range(open_brace_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace_idx:i + 1]
    raise AssertionError("unbalanced braces")


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
    guard = _brace_block(body, m.end() - 1)
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
    guard = _brace_block(body, m.end() - 1)
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
    guard = _brace_block(body, m.end() - 1)
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
    guard = _brace_block(body, m.end() - 1)
    assert re.search(
        r"hexlib_write_rsp_hdr\s*\([^;]*HEXLIB_DSP_ERR_NOT_STARTED", guard
    ), "the refusal must write NOT_STARTED into the response, not just log it"
    assert "hexlib_dispatch_batch" not in guard, "invoke-before-start must not run any op"


def test_hwinfo_reports_the_acquired_vtcm_size(s):
    assert "vtcm_size" in s
    assert "8388608" not in s, "hwinfo must report what was acquired, not a constant"


def test_skel_defines_the_iface_symbols_qaic_expects(s):
    for sym in ("hexlib_iface_open", "hexlib_iface_close", "hexlib_iface_start",
                "hexlib_iface_stop", "hexlib_iface_mmap", "hexlib_iface_munmap",
                "hexlib_iface_hwinfo", "hexlib_iface_invoke"):
        assert sym in s, sym

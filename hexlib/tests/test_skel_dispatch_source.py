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

AND SO ARE STRING LITERALS, AS OF THE MERGE-GATE REVIEW THAT FOUND THEM DOING
THE SAME JOB. `csource.code_only` used to hand literals back intact, so this
file's whole promise above was available in a second vehicle: the total_size
guard could log HEXLIB_DSP_ERR_TRUNCATED and fall through (12 passed), and a
`}` inside `FARF(HIGH, "pcycle }")` truncated `function_body`'s slice so the
"no raw register read here" negatives were answered by a fragment that stopped
at the literal (12 passed, with `__asm__("%0 = c15:14")` back in the wrapper).
Both are fixed in `csource`; see its docstring.

FUNCTION SCOPE IS NOT REACHABILITY, WHICH IS THE THIRD THING THIS FILE HAD
WRONG. Moving the raw register read into a new `hexlib_raw_pcycle()` helper
that `hexlib_read_pcycle` calls left this file at 12 passed with no comment or
literal trick at all -- a scoped negative cannot see a rename. Every negative
here is now paired with either a whole-FILE ban (for a construct that must not
exist anywhere) or an exhaustive `csource.calls()` set (for one that must not
be REACHED from a particular block).

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
from hexlib.tests.csource import calls as _calls
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
    # WAS `assert "HEXLIB_DSP_ERR_TRUNCATED" in guard`, which is the bare-token
    # check this file's own docstring says it does not do: a guard that FARFs
    # the constant's name and FALLS THROUGH satisfied it (12 passed), letting a
    # batch whose declared size disagrees with its actual length go on to be
    # walked. Both halves of reporting are now required, in the shapes the
    # docstring promises -- the header-writer call the host reads its status
    # from, and the return that stops the walk.
    assert re.search(
        r"hexlib_write_rsp_hdr\s*\([^;]*HEXLIB_DSP_ERR_TRUNCATED", guard
    ), (
        "the length disagreement must be written into the response header the "
        "host actually reads, not merely logged"
    )
    assert re.search(r"return\s+HEXLIB_DSP_ERR_TRUNCATED\s*;", guard), (
        "and it must RETURN -- a guard that reports and then falls through "
        "walks the batch anyway, which is the whole thing this check is for"
    )


def test_the_cycle_counter_is_read_through_the_sdks_own_api(d):
    """NOT hand-rolled inline asm. `__asm__("%0 = c15:14")` only advances if
    SYSCFG.PCYCLEEN is set, and a user-mode unsigned PD -- which is what the
    skel runs in -- cannot set that bit. This project's own
    include/hexlib/hexlib_harness.h sets it explicitly for the standalone-ELF
    runtime, so the register reading 0 with the bit clear is a fact this repo
    already records. The simulator measures a plausible four-figure number
    either way and therefore cannot discriminate.

    So the read must go through HAP_perf_get_pcycles()
    ($HEXAGON_SDK_ROOT/incs/HAP_perf.h), which issues the identical
    instruction: same mechanism, but a documented SDK API rather than an
    invented one, so a zero on silicon is a reportable platform fact about the
    PD instead of an indistinguishable bug of ours.

    BOTH HALVES ARE ASSERTED, and both are scoped to the wrapper's own body
    (comments and literals already blanked by `code_only`), so neither can be
    satisfied by prose or by a log line: the SDK call must be PRESENT, and the
    raw register read must be ABSENT. A revert to inline asm fails the second
    half even if the first is left behind as dead code.

    AND THE NEGATIVE HALF IS ALSO ASSERTED AT FILE SCOPE, WHICH IS THE ONLY
    SCOPE THAT MEANS ANYTHING FOR IT. A scoped negative asks "is the forbidden
    construct in THIS function", and the answer is no as soon as it is moved
    into a helper this function calls -- proven, and it needed no comment and no
    string literal: `hexlib_raw_pcycle()` holding the `__asm__` while
    `hexlib_read_pcycle` called it (with a dead `if (0)` branch keeping the
    positive half green) left this file at 12 passed, with the wrapper reading a
    register that cannot advance in the PD the skel actually runs in. The claim
    was never really about this function: it is that NOTHING in this
    translation unit reads the counter by hand. So it is checked that way, over
    the whole comment- and literal-blanked file."""
    body = _function_body(d, "hexlib_read_pcycle")
    assert re.search(r"\bHAP_perf_get_pcycles\s*\(\s*\)", body), (
        "hexlib_read_pcycle must read the counter through the SDK's own "
        f"HAP_perf_get_pcycles(), got: {body!r}"
    )
    assert "__asm__" not in body and "asm" not in body, (
        "the cycle counter must not be read by hand-rolled inline asm -- "
        "SYSCFG.PCYCLEEN is unsettable from a user-mode unsigned PD, so a raw "
        f"`c15:14` read may simply return 0 there: {body!r}"
    )
    assert "c15:14" not in body and "C15:14" not in body, (
        f"no raw register read may survive in this wrapper: {body!r}"
    )
    # THE SAME TWO BANS, AT FILE SCOPE. `d` is comment- AND literal-blanked, so
    # this file's own header comment discussing the `__asm__("%0 = c15:14")` it
    # replaced does not trip these, and neither would a FARF quoting it.
    assert "c15:14" not in d and "C15:14" not in d, (
        "no raw c15:14 read may survive anywhere in skel_dispatch.c -- moving "
        "it into a helper the wrapper calls is the same bug with a new name"
    )
    assert "asm" not in d, (
        "no inline asm anywhere in skel_dispatch.c: the counter's one legal "
        "read is HAP_perf_get_pcycles(), and a hand-rolled read one call level "
        "away is still a hand-rolled read"
    )
    # And the wrapper must be the ONLY thing that reads the counter, so the
    # bracketing test below is measuring what it thinks it is.
    assert _calls(body) == {"HAP_perf_get_pcycles"}, (
        f"hexlib_read_pcycle must call the SDK's reader and nothing else -- an "
        f"extra callee here is where a hand-rolled read hides: {_calls(body)!r}"
    )
    # And the include that makes it legal, in code rather than in a comment.
    # (`#include "HAP_perf.h"` is a header-name, not a string literal, so
    # `code_only` leaves it intact on purpose -- see csource.py.)
    assert re.search(r'#\s*include\s+"HAP_perf\.h"', d), (
        "HAP_perf.h must actually be included, not merely referred to"
    )


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
    # WAS `assert "c15:14" in d or "PCYCLE" in d` -- both halves were
    # satisfiable by a comment before the fixtures were switched to
    # `code_only`, and the first half pinned the hand-rolled asm this file now
    # bans outright. The counter's provenance is
    # test_the_cycle_counter_is_read_through_the_sdks_own_api's job above; this
    # test is only about WHERE the pair of reads sits.
    body = _function_body(d, "hexlib_dispatch_batch")
    calls = [m.start() for m in re.finditer(r"hexlib_read_pcycle\s*\(\s*\)", body)]
    assert len(calls) >= 2, "expected at least a before/after pair of calls"
    lo, hi = calls[0], calls[-1]
    between = body[lo:hi]
    assert re.search(r"\bk->fn\s*\(", between), (
        "the kernel call must be inside the bracket -- and it must be the call "
        "through the table's own function pointer, not merely the text `->fn(`"
    )
    # THE EXHAUSTIVE CALLEE SET, NOT THREE NAMED BANS. The three below were
    # named because they were the three things that existed when this was
    # written; anything else that got moved between the reads -- including a
    # one-line helper wrapping the resolve, which is exactly how the same
    # escape was proven against the pcycle wrapper above -- would have gone
    # unnoticed while inflating every measured cycle count on silicon.
    assert _calls(between) == {"hexlib_read_pcycle", "fn"}, (
        f"only the opening pcycle read and the kernel call itself may sit "
        f"inside the bracket; found {_calls(between)!r}"
    )
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
    assert re.search(r"\bbreak\s*;", guard), (
        "an unknown kind must stop the batch, not continue it -- and it must be "
        "an actual `break;` statement"
    )


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
    # A CALL STATEMENT ON THE SESSION CONTEXT, not the token. Deleting the call
    # and leaving its name inside the FARF beside it -- `FARF(HIGH, "hexlib:
    # hexlib_vtcm_release(ctx) deferred ...")` -- satisfied `"hexlib_vtcm_
    # release(" in guard` and left this file at 12 passed, with the batch
    # stopping while still holding the reservation the competing session is
    # blocked on. That is the whole failure this test is named for, and it is
    # invisible to the simulator, where nothing else wants VTCM.
    assert re.search(r"\bhexlib_vtcm_release\s*\(\s*ctx\s*\)\s*;", guard), (
        "the reclaim path must actually call hexlib_vtcm_release(ctx) -- "
        "stopping the batch without giving the memory back leaves the "
        "competing session waiting on a reservation nobody will release"
    )
    assert re.search(r"batch_status\s*=\s*HEXLIB_DSP_ERR_VTCM_RECLAIMED", guard)
    assert re.search(r"\bbreak\s*;", guard), (
        "must stop at the op boundary with an actual `break;`, not continue"
    )
    # And nothing else happens in here: an exhaustive callee set, so the release
    # cannot be swapped for a helper that only logs (see csource.calls()).
    assert _calls(guard) == {"FARF", "hexlib_vtcm_release"}, (
        f"the reclaim path must log and release, and do nothing else at an op "
        f"boundary; found {_calls(guard)!r}"
    )


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
    # AND NOT VIA ANYTHING ELSE EITHER. The named ban above cannot see a
    # one-line helper that dispatches; the exhaustive callee set can.
    assert _calls(guard) == {"FARF", "hexlib_write_rsp_hdr"}, (
        f"the refusal may log and write the response header, and must call "
        f"nothing else -- anything else is a path to running an op; found "
        f"{_calls(guard)!r}"
    )


def test_both_wire_lengths_are_checked_for_a_negative_value(s):
    """`batchLen` and `resultLen` are both `int` on the wire (qaic spells
    `sequence<octet>` as a pointer plus a signed length), and only `resultLen`
    was checked. A negative `batchLen` cast to uint32_t becomes an enormous
    length, which PASSES hexlib_dispatch_batch's `len < sizeof(struct
    hexlib_batch_hdr)` test -- so the dispatcher memcpy()s the full 40-byte
    header out of `batch` before `hdr.total_size != len` can reject anything.
    That is an out-of-bounds read of a buffer the host may have made much
    shorter, and this entry point is the last place the sign is still visible:
    after the cast the information is gone.

    Both guards must be inside hexlib_iface_invoke's own body and must
    precede the cast, so this checks position as well as presence -- a check
    added after the call to hexlib_dispatch_batch would protect nothing."""
    body = _function_body(s, "hexlib_iface_invoke")
    dispatch_pos = body.index("hexlib_dispatch_batch")
    for name in ("batchLen", "resultLen"):
        m = re.search(rf"\b{name}\s*<\s*0\b", body)
        assert m, (
            f"hexlib_iface_invoke does not reject a negative {name} -- cast to "
            f"uint32_t it becomes a huge length that passes every subsequent "
            f"size test"
        )
        assert m.start() < dispatch_pos, (
            f"the negative-{name} guard must run BEFORE hexlib_dispatch_batch "
            f"is called with the cast value, not after"
        )
    # And the refusal must be reported, not merely detected: the batch-length
    # path has a valid response buffer (resultLen was already checked above
    # it), so it must write a real status the host can read off the wire.
    m = re.search(r"if\s*\(\s*batchLen\s*<\s*0\s*\)\s*\{", body)
    assert m, "the negative-batchLen guard must be its own `if` block"
    guard = _block_from(body, m.end() - 1)
    assert re.search(r"hexlib_write_rsp_hdr\s*\([^;]*HEXLIB_DSP_ERR_", guard), (
        "a negative batchLen must be reported in the response header, not "
        "merely logged and dropped"
    )
    assert "hexlib_dispatch_batch" not in guard, (
        "a negative batchLen must not reach the dispatcher at all"
    )
    assert _calls(guard) == {"FARF", "hexlib_write_rsp_hdr"}, (
        f"same as the invoke-before-start refusal: log, write the header, call "
        f"nothing else -- a helper that dispatches would satisfy the named ban "
        f"above; found {_calls(guard)!r}"
    )


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

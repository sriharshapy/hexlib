"""VTCM acquisition. Source assertions; the behaviour is Task 8's hwinfo check.

EVERY CHECK HERE RUNS AGAINST COMMENT-BLANKED, FUNCTION-SCOPED TEXT. Two
proven mutations got through the earlier version of this file. (1) Deleting
the HAP_compute_res_query_VTCM call outright and hardcoding
`vtcm_size = 4*1024*1024` passed, because the constant-ban below only banned
the 8 MiB spelling and the presence check was satisfied by the FARF string
that names the function in its error message. (2) Gutting `release_callback`
and moving `ctx->vtcm_needs_release = 1;` into `hexlib_vtcm_alloc` passed,
because `callback_body = src[:registered_at]` was not a body at all -- it was
the whole file up to the registration call, so anything defined above it
counted. Both are now scoped to the function whose behaviour is claimed.
"""
import pathlib
import re

import pytest

from hexlib.tests.csource import block_after_call as _block_after_call
from hexlib.tests.csource import block_from as _block_from
from hexlib.tests.csource import code_only as _code_only
from hexlib.tests.csource import function_body as _function_body

SRC = pathlib.Path("hexlib/runtime/skel/skel_vtcm.c")


@pytest.fixture(scope="module")
def src():
    """Comment-blanked. Length-preserving, so csource's offsets stay valid."""
    return _code_only(SRC.read_text())


def test_size_comes_from_the_runtime_never_a_constant(src):
    """`STATE.md`: the part total is not the usable budget. VTCM is acquired at
    session start, so the size must come from the runtime.

    THE OUT-PARAMETER, AND NOTHING ELSE, MAY SET `vtcm_size`. `"HAP_compute_
    res_query_VTCM" in src` was satisfied by this file's own FARF error string,
    and the literal bans below only covered 8 MiB -- so deleting the call and
    writing `vtcm_size = 4*1024*1024` passed. The positive check now requires
    the real call with `&vtcm_size` among its arguments, and the negative check
    enumerates every assignment to the local and allows only the `= 0`
    initializer: any other constant, of any magnitude or spelling, fails."""
    alloc = _function_body(src, "hexlib_vtcm_alloc")
    assert re.search(r"HAP_compute_res_query_VTCM\s*\([^;]*&\s*vtcm_size", alloc), (
        "hexlib_vtcm_alloc must ask the runtime for the size, passing "
        "&vtcm_size as the out-parameter -- naming the function in a log "
        "message is not asking it"
    )
    # `(?<![\w>.])` so this sees the LOCAL `vtcm_size`, not `ctx->vtcm_size`.
    for m in re.finditer(r"(?<![\w>.])vtcm_size\s*=\s*([^;=]+);", alloc):
        rhs = m.group(1).strip()
        assert rhs == "0", (
            f"vtcm_size must only ever be set by the runtime query's "
            f"out-parameter (the `= 0` initializer aside); found "
            f"`vtcm_size = {rhs};`"
        )
    # Belt: the 8 MiB part total, in both the decimal and hex forms the v75
    # spec and the address quote it in, must not appear anywhere in the code.
    assert "8388608" not in src
    assert "0x800000" not in src.lower()


def test_the_hardcoded_vtcm_address_appears_nowhere(src):
    assert "0xd9000000" not in src.lower()


def test_a_release_callback_is_registered(src):
    """A competing QNN-HTP or GGML-HTP session can reclaim VTCM mid-run. Not
    registering the callback does not make that stop happening; it makes it
    silent.

    THE CALLBACK'S OWN BODY, AND THE REGISTRATION THAT NAMES IT. This was
    `src[:registered_at]` -- the whole file prefix, not a body -- so gutting
    release_callback and hoisting the flag into hexlib_vtcm_alloc passed.

    The callback must flip the flag ON (not just mention the field, e.g. only
    ever clearing it) and must not release VTCM itself: the batch in flight
    may still be using the memory, so releasing is the dispatcher's job at an
    op boundary (Task 6), not the callback's."""
    alloc = _function_body(src, "hexlib_vtcm_alloc")
    assert re.search(
        r"HAP_compute_res_attr_set_release_callback\s*\([^;]*\brelease_callback\b",
        alloc,
    ), (
        "hexlib_vtcm_alloc must register release_callback itself with the "
        "compute-res attributes -- not merely name the setter"
    )

    callback_body = _function_body(src, "release_callback")
    assert re.search(r"->vtcm_needs_release\s*=\s*1\s*;", callback_body), (
        "release_callback must record the reclaim request by setting "
        "ctx->vtcm_needs_release = 1 -- if some other function sets it, the "
        "reclaim request itself is being dropped on the floor"
    )
    assert "HAP_compute_res_release(" not in callback_body
    assert "HAP_compute_res_release_cached(" not in callback_body


def test_every_hap_failure_path_returns_a_status(src):
    """Each `HAP_compute_res_*` call this file inspects the result of must
    propagate a non-OK status to the caller when that check fails, not just
    mention an error constant somewhere in the file -- the file-wide version
    of this check would also pass for an implementation that calls every
    HAP function, discards every return value, always returns
    HEXLIB_DSP_OK, and happens to reference an error constant once in a dead
    branch or a comment. 'Log and continue' -- FARF the failure and fall
    through to `return HEXLIB_DSP_OK;` -- is exactly the regression class
    this exists to catch; it is the same defect fixed in skel_bufs.c's
    upstream (see ATTRIBUTION.md) and it would hand a kernel a null or
    zero-length VTCM base pointer if reintroduced here."""
    alloc = _function_body(src, "hexlib_vtcm_alloc")

    query_block = _block_after_call(alloc, "HAP_compute_res_query_VTCM")
    assert re.search(r"return\s+HEXLIB_DSP_ERR_\w+\s*;", query_block)

    acquire_block = _block_after_call(alloc, "HAP_compute_res_acquire")
    assert re.search(r"return\s+HEXLIB_DSP_ERR_\w+\s*;", acquire_block)

    ptr_block = _block_after_call(alloc, "HAP_compute_res_attr_get_vtcm_ptr_v2")
    assert re.search(r"return\s+HEXLIB_DSP_ERR_\w+\s*;", ptr_block)


def test_hmx_is_requested_only_when_the_session_asked_for_it(src):
    """No kernel on this branch needs HMX, and `session.c` always passes
    `n_hmx = 0` to `hexlib_iface_start`. Requesting HMX unconditionally here
    risks `HAP_compute_res_acquire` refusing the WHOLE reservation for an
    HMX-availability reason indistinguishable, from its single status code
    alone, from a VTCM-size failure -- the operator would see a VTCM error
    for what was actually an HMX one. `HAP_compute_res_attr_set_hmx_param`
    must therefore be guarded by a real check of `ctx->n_hmx`, not called
    unconditionally in `hexlib_vtcm_alloc`."""
    alloc = _function_body(src, "hexlib_vtcm_alloc")
    guard = re.search(r"if\s*\(\s*ctx->n_hmx\s*>\s*0\s*\)", alloc)
    assert guard, (
        "HAP_compute_res_attr_set_hmx_param must be guarded by ctx->n_hmx > 0"
    )
    guarded_block = _block_from(alloc, guard.end())
    assert "HAP_compute_res_attr_set_hmx_param" in guarded_block, (
        "the HMX request itself must live inside the ctx->n_hmx > 0 guard, "
        "not merely have an unrelated if-block near it"
    )
    # And nowhere else in the function, unguarded -- a duplicate call outside
    # the guard would defeat the whole point.
    outside = alloc.replace(guarded_block, "", 1)
    assert "HAP_compute_res_attr_set_hmx_param" not in outside


def test_no_abort_or_assert_anywhere_in_the_file(src):
    """Fail closed means returning a status, not killing the process --
    upstream aborts on failure; we must not. This is a whole-file negative
    check over CODE ONLY (this file's header discusses upstream's abort, and
    a comment saying so is not an abort), and it legitimately passes for any
    file that simply never spells abort()/assert() -- it does not by itself
    prove a failure is detected or propagated. See
    test_every_hap_failure_path_returns_a_status for that half."""
    assert "abort()" not in src
    assert "assert(" not in src

"""VTCM acquisition. Source assertions; the behaviour is Task 8's hwinfo check."""
import pathlib
import re

import pytest

from hexlib.tests.csource import block_after_call as _block_after_call
from hexlib.tests.csource import function_body as _function_body

SRC = pathlib.Path("hexlib/runtime/skel/skel_vtcm.c")


@pytest.fixture(scope="module")
def src():
    return SRC.read_text()


def test_size_comes_from_the_runtime_never_a_constant(src):
    """`STATE.md`: the part total is not the usable budget. VTCM is acquired at
    session start, so the size must come from the runtime."""
    assert "HAP_compute_res_query_VTCM" in src
    # A call whose result is discarded in favor of the literal 8 MiB budget would
    # still satisfy the check above; catch that by banning the literal itself in
    # both the decimal and hex forms the v75 spec and the address quote it in.
    assert "8388608" not in src
    assert "0x800000" not in src.lower()


def test_the_hardcoded_vtcm_address_appears_nowhere(src):
    assert "0xd9000000" not in src.lower()


def test_a_release_callback_is_registered(src):
    """A competing QNN-HTP or GGML-HTP session can reclaim VTCM mid-run. Not
    registering the callback does not make that stop happening; it makes it
    silent."""
    assert "HAP_compute_res_attr_set_release_callback" in src
    assert "vtcm_needs_release" in src
    # The callback (defined before it is registered, so slicing up to the
    # registration call isolates its body) must actually flip the flag on --
    # not just mention the field somewhere unrelated, e.g. only ever clearing
    # it -- and it must not release VTCM itself: the batch in flight may still
    # be using the memory, so releasing is the dispatcher's job at an op
    # boundary (Task 6), not the callback's.
    registered_at = src.index("HAP_compute_res_attr_set_release_callback")
    callback_body = src[:registered_at]
    assert "vtcm_needs_release = 1" in callback_body
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


def test_no_abort_or_assert_anywhere_in_the_file(src):
    """Fail closed means returning a status, not killing the process --
    upstream aborts on failure; we must not. This is a whole-file negative
    check and legitimately passes for any file that simply never spells
    abort()/assert() -- it does not by itself prove a failure is detected or
    propagated. See test_every_hap_failure_path_returns_a_status for that
    half."""
    assert "abort()" not in src
    assert "assert(" not in src

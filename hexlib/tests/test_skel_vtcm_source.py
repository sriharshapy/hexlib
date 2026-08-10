"""VTCM acquisition. Source assertions; the behaviour is Task 8's hwinfo check."""
import pathlib
import re

import pytest

SRC = pathlib.Path("hexlib/runtime/skel/skel_vtcm.c")


@pytest.fixture(scope="module")
def src():
    return SRC.read_text()


def _function_body(src, name):
    """Slice the text of a C function from its signature to its matching
    closing brace, by simple brace-depth counting. Good enough for this one
    file's straight-line C; not a general C parser.

    Copied from `test_skel_bufs_source.py` (Task 4) rather than reimplemented,
    per the coordinator's note that a third variant of the same helper is not
    wanted."""
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


def _block_after_call(body, call_name):
    """Within a function body, find a call to `call_name` and return the text
    of the nearest brace-delimited block that checks its result -- either the
    call sits inside an `if` condition (`if (call(...) != 0) { ... }`), or an
    `if` immediately follows the call as a separate statement (`x =
    call(...); if (!x) { ... }`). Both shapes occur in this file.

    Asserts an `if (` appears between the call and the block, so a stray
    block that has nothing to do with checking the call's result cannot be
    picked up by accident."""
    m = re.search(rf"\b{re.escape(call_name)}\s*\(", body)
    assert m, f"no call to {call_name}() found in this function"
    call_start = m.start()

    # Walk the call's own parens to find where its argument list ends --
    # none of this file's calls nest parens, but do it properly anyway.
    depth = 0
    call_end = None
    for i in range(m.end() - 1, len(body)):
        if body[i] == "(":
            depth += 1
        elif body[i] == ")":
            depth -= 1
            if depth == 0:
                call_end = i + 1
                break
    assert call_end is not None, f"unbalanced parens in the call to {call_name}()"

    brace_pos = body.find("{", call_end)
    assert brace_pos != -1, f"no block follows the call to {call_name}()"

    window = body[max(0, call_start - 80):brace_pos]
    assert "if" in window and "(" in window, (
        f"{call_name}()'s result does not appear to be checked by an `if` "
        f"before the block that follows it"
    )

    depth = 0
    for i in range(brace_pos, len(body)):
        if body[i] == "{":
            depth += 1
        elif body[i] == "}":
            depth -= 1
            if depth == 0:
                return body[brace_pos:i + 1]
    raise AssertionError(f"unbalanced braces in the block following {call_name}()")


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

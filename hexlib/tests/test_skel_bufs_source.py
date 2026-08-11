# hexlib/tests/test_skel_bufs_source.py
"""The pointer-free invariant, asserted against the source.

These are source assertions, not behavioural ones — the behavioural test is
Task 8's unmapped-fd run on the simulator. They exist because the invariant is
easy to break in a way that PASSES on the simulator: host and DSP share one
address space there, so a skel that trusted the host's `base` would return the
right answer and only fail on silicon. Two independent guards, at two levels.

TWICE REWRITTEN, BOTH TIMES FOR THE SAME REASON. A first version checked bare
substrings anywhere in the file; a second scoped some of them to a function
but still ran every check against comment-BEARING text. Both were defeated by
the same mutation: replace hexlib_bufs_map's unmapped-fd refusal with
`b->base = (uint64_t) b->fd; continue;` -- the shared-address-space bug this
whole file exists to catch -- and all eight tests still passed, because
`"HAP_mmap" in src` was satisfied by a comment, `_returns(src, ...)` was
whole-file and satisfied by an unrelated return in a different function, and
`"nbytes" in src` was satisfied by a FARF format string. So:

  * every fixture and every slice is COMMENT-BLANKED (csource strips by
    default; `code_only` does the whole file), so nothing a mutation leaves
    behind as a comment can satisfy anything here;
  * every check is scoped to the ONE function -- usually the one `if`-block --
    whose behaviour the claim is about, never the file;
  * every presence check is a CALL or an ASSIGNMENT shape, never a bare token,
    so a mention in a log-message format string is not evidence of anything.
"""
import pathlib
import re

import pytest

from hexlib.tests.csource import block_from as _block_from
from hexlib.tests.csource import code_only as _code_only
from hexlib.tests.csource import function_body as _function_body

SRC = pathlib.Path("hexlib/runtime/skel/skel_bufs.c")


@pytest.fixture(scope="module")
def src():
    """Comment-blanked, so every check below is about code. Blanking preserves
    length, so `csource`'s offsets stay valid against this text."""
    return _code_only(SRC.read_text())


def _returns(fragment, constant):
    """A RETURN of the given status constant, inside `fragment` -- which must
    be a function body or (better) the one `if`-block the claim is about.
    NEVER pass the whole file: this was whole-file once, and an unrelated
    `return HEXLIB_DSP_ERR_UNMAPPED;` in hexlib_bufs_unregister then stood in
    for the one hexlib_bufs_map is supposed to have, which is how a mutation
    that deleted the real one passed."""
    return re.search(rf"return\s+{re.escape(constant)}\s*;", fragment) is not None


def test_base_is_cleared_before_any_lookup(src):
    """hexlib_bufs_map must destroy whatever base the host sent before it can
    be read by the fd lookup. This is a behavioural claim about ONE function's
    body, not about the file's layout — checking whole-file text order would
    incidentally constrain where helpers like find_by_fd get defined, which is
    not the invariant. Slice hexlib_bufs_map itself and check order there."""
    body = _function_body(src, "hexlib_bufs_map")
    assert "b->base = 0" in body or "b->base = NULL" in body
    clear = min(
        (body.index(s) for s in ("b->base = 0", "b->base = NULL") if s in body),
        default=-1,
    )
    assert clear != -1
    lookup = re.search(r"\bfind_by_fd\s*\(", body)
    assert lookup, "hexlib_bufs_map does not appear to look the buffer up at all"
    assert clear < lookup.start(), "clear base before looking the buffer up"


def test_lookup_is_by_fd(src):
    """The match must be an actual `==` comparison against the slot's fd,
    inside find_by_fd itself -- not the token `->fd ==` anywhere in the file,
    which any unrelated comparison would satisfy."""
    body = _function_body(src, "find_by_fd")
    assert re.search(r"->fd\s*==", body), (
        "find_by_fd must match slots by comparing the stored fd, never by "
        "anything the host supplied as an address"
    )
    assert not re.search(r"->base\s*==", body), (
        "find_by_fd must never key its lookup off `base` -- that is the host's "
        "value and the whole point of this file is that it is not trusted"
    )


def test_the_dsp_maps_the_fd_itself(src):
    """A REAL CALL, whose result is assigned -- not the token `HAP_mmap`
    anywhere in the file. That check passed for a gutted hexlib_bufs_register
    with both calls deleted, because skel_bufs.c's own comment about
    `HAP_mmap`'s `len` argument spells the name in prose.

    Both spellings are required because both are live: HAP_mmap2 on
    `__HVX_ARCH__ > 73` and HAP_mmap below it (see the file's own comment on
    the `int` vs `size_t` length argument). Deleting either silently removes
    the mapping on one arch."""
    body = _function_body(src, "hexlib_bufs_register")
    assert re.search(r"=\s*HAP_mmap2\s*\(", body), (
        "hexlib_bufs_register must map the fd itself via HAP_mmap2 on the "
        "v75+ branch, assigning the result -- not merely name it"
    )
    assert re.search(r"=\s*HAP_mmap\s*\(", body), (
        "hexlib_bufs_register must map the fd itself via HAP_mmap on the "
        "pre-v75 branch, assigning the result -- not merely name it"
    )


def test_an_unmapped_fd_is_an_error_not_a_zero_base(src):
    """Upstream returns silently with base == 0 when no slot is free, and the
    caller then computes 0 + offset and reads a small bogus address. Fixed.

    EACH STATUS IS PINNED TO THE BLOCK THAT MUST REPORT IT, not to the file.
    This was three whole-file `_returns` calls, and hexlib_bufs_unregister's
    own `return HEXLIB_DSP_ERR_UNMAPPED;` stood in for hexlib_bufs_map's --
    so replacing hexlib_bufs_map's refusal with "FARF and continue, trusting
    the host's fd as an address" passed. That is precisely the
    shared-address-space bug the simulator cannot see."""
    map_body = _function_body(src, "hexlib_bufs_map")
    miss = re.search(r"if\s*\(\s*!\s*m\s*\)\s*\{", map_body)
    assert miss, (
        "hexlib_bufs_map must branch on find_by_fd() having found nothing"
    )
    miss_block = _block_from(map_body, miss.end() - 1)
    assert _returns(miss_block, "HEXLIB_DSP_ERR_UNMAPPED"), (
        "an fd the DSP never mapped must be refused from inside that branch "
        "-- logging and continuing (with or without an address derived from "
        "the fd) is the upstream bug this file exists to have fixed"
    )
    assert "continue" not in miss_block, (
        "the unmapped-fd branch must not continue the loop: the buffer would "
        "be handed to a kernel with whatever base was left in it"
    )

    reg_body = _function_body(src, "hexlib_bufs_register")
    assert _returns(reg_body, "HEXLIB_DSP_ERR_NO_MMAP_SLOT"), (
        "running out of mmap slots must be reported by hexlib_bufs_register "
        "itself, not left as a silent base == 0 fallthrough"
    )
    mmap_failed = re.search(r"if\s*\(\s*va\s*==", reg_body)
    assert mmap_failed, "the result of the mapping call must be checked"
    mmap_failed_block = _block_from(reg_body, mmap_failed.end())
    assert _returns(mmap_failed_block, "HEXLIB_DSP_ERR_MMAP_FAILED"), (
        "a failed mapping must be reported from inside its own check"
    )


def test_no_abort_on_a_failed_mapping(src):
    """Upstream abort()s. Fail closed means returning a status, not killing the
    process and leaving the host to interpret a dead session. Whole-file
    negative, over code only -- a comment discussing upstream's abort() (this
    file's header does) is not an abort()."""
    assert "abort()" not in src


def test_tensor_data_is_computed_from_base_plus_offset(src):
    """The kernel-visible address must be DERIVED, in one assignment, from the
    mapped base and the tensor's own offset. `"base" in src and "offset" in
    src` was satisfied by the file-header comment, and `"->data =" in src` by
    the mutation `t->data = 0;` itself -- the gutting that check was supposed
    to catch."""
    body = _function_body(src, "hexlib_tensors_resolve")
    assert re.search(r"->data\s*=\s*[^;]*->base\s*\+\s*[^;]*->offset", body), (
        "the tensor's data address must be computed as the DSP-side mapped "
        "base plus the tensor's offset, in that one assignment"
    )


def test_resolution_bounds_checks_the_offset(src):
    """A tensor whose offset+nbytes exceeds its buffer must be refused on the
    DSP too. The host checks it, but the host is not the thing being trusted.

    THE GUARD ITSELF, AND ITS OWN RETURN. `_returns(src, TRUNCATED)` was
    whole-file and was satisfied by a different check's return; `"nbytes" in
    src` was satisfied by the word `nbytes` inside a FARF format string. So a
    version that dropped `nbytes` from the bound and deleted this return
    passed."""
    body = _function_body(src, "hexlib_tensors_resolve")
    guard = re.search(r"if\s*\([^;{]*->nbytes[^;{]*\)\s*\{", body)
    assert guard, (
        "hexlib_tensors_resolve must have an `if` whose condition involves the "
        "tensor's own nbytes -- a bound on offset alone is not a bound"
    )
    cond = guard.group(0)
    for token in ("->offset", "->nbytes", "->size", "+", ">"):
        assert token in cond, (
            f"the bound must compare offset + nbytes against the buffer size; "
            f"{token!r} is missing from {cond!r}"
        )
    guard_block = _block_from(body, guard.end() - 1)
    assert _returns(guard_block, "HEXLIB_DSP_ERR_TRUNCATED") or _returns(
        guard_block, "HEXLIB_DSP_ERR_INVAL_PARAMS"
    ), "the out-of-bounds case must return a status from inside its own block"


def test_buffer_index_is_range_checked(src):
    """The out-of-range case must be a real comparison against the buffer
    count, with its own return. `"n_bufs" in src` was satisfied by
    hexlib_tensors_resolve's own PARAMETER NAME, so deleting the guard
    entirely left this test passing."""
    body = _function_body(src, "hexlib_tensors_resolve")
    guard = re.search(r"if\s*\([^;{]*->bi\s*>=\s*n_bufs\s*\)\s*\{", body)
    assert guard, (
        "hexlib_tensors_resolve must refuse a tensor naming a buffer index "
        "at or past n_bufs -- the parameter merely being named is not a check"
    )
    guard_block = _block_from(body, guard.end() - 1)
    assert _returns(guard_block, "HEXLIB_DSP_ERR_INVAL_PARAMS") or _returns(
        guard_block, "HEXLIB_DSP_ERR_UNMAPPED"
    ), "the out-of-range case must actually return an error, not just log one"

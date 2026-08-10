# hexlib/tests/test_skel_bufs_source.py
"""The pointer-free invariant, asserted against the source.

These are source assertions, not behavioural ones — the behavioural test is
Task 8's unmapped-fd run on the simulator. They exist because the invariant is
easy to break in a way that PASSES on the simulator: host and DSP share one
address space there, so a skel that trusted the host's `base` would return the
right answer and only fail on silicon. Two independent guards, at two levels.
"""
import pathlib
import re

import pytest

SRC = pathlib.Path("hexlib/runtime/skel/skel_bufs.c")


@pytest.fixture(scope="module")
def src():
    return SRC.read_text()


def _function_body(src, name):
    """Slice the text of a C function from its signature to its matching
    closing brace, by simple brace-depth counting. Good enough for this one
    file's straight-line C; not a general C parser."""
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


def _returns(src, constant):
    """A RETURN of the given status constant, not just the token anywhere in
    the file (a comment or a FARF log line mentioning it does not count)."""
    return re.search(rf"return\s+{re.escape(constant)}\s*;", src) is not None


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
    assert "->fd ==" in src


def test_the_dsp_maps_the_fd_itself(src):
    assert "HAP_mmap" in src


def test_an_unmapped_fd_is_an_error_not_a_zero_base(src):
    """Upstream returns silently with base == 0 when no slot is free, and the
    caller then computes 0 + offset and reads a small bogus address. Fixed.

    Each status must appear in an actual `return`, not merely somewhere in the
    file (a FARF log line naming the constant is not the same as reporting it
    to the caller) — that is exactly how upstream's silent-fallthrough bug
    could be reintroduced as "log and continue"."""
    assert _returns(src, "HEXLIB_DSP_ERR_UNMAPPED")
    assert _returns(src, "HEXLIB_DSP_ERR_NO_MMAP_SLOT")
    assert _returns(src, "HEXLIB_DSP_ERR_MMAP_FAILED")


def test_no_abort_on_a_failed_mapping(src):
    """Upstream abort()s. Fail closed means returning a status, not killing the
    process and leaving the host to interpret a dead session."""
    assert "abort()" not in src


def test_tensor_data_is_computed_from_base_plus_offset(src):
    assert "base" in src and "offset" in src
    assert "->data =" in src


def test_resolution_bounds_checks_the_offset(src):
    """A tensor whose offset+nbytes exceeds its buffer must be refused on the
    DSP too. The host checks it, but the host is not the thing being trusted.
    Must be an actual return to the caller, not just a logged constant."""
    assert _returns(src, "HEXLIB_DSP_ERR_TRUNCATED") or _returns(
        src, "HEXLIB_DSP_ERR_INVAL_PARAMS"
    )
    assert "nbytes" in src


def test_buffer_index_is_range_checked(src):
    """The out-of-range case must actually return an error, not just log one."""
    assert "n_bufs" in src
    assert _returns(src, "HEXLIB_DSP_ERR_INVAL_PARAMS") or _returns(
        src, "HEXLIB_DSP_ERR_UNMAPPED"
    )

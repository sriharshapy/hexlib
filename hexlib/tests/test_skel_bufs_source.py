# hexlib/tests/test_skel_bufs_source.py
"""The pointer-free invariant, asserted against the source.

These are source assertions, not behavioural ones — the behavioural test is
Task 8's unmapped-fd run on the simulator. They exist because the invariant is
easy to break in a way that PASSES on the simulator: host and DSP share one
address space there, so a skel that trusted the host's `base` would return the
right answer and only fail on silicon. Two independent guards, at two levels.
"""
import pathlib

import pytest

SRC = pathlib.Path("hexlib/runtime/skel/skel_bufs.c")


@pytest.fixture(scope="module")
def src():
    return SRC.read_text()


def test_base_is_cleared_before_any_lookup(src):
    """Upstream's reuse_buf sets b->base = NULL FIRST. That ordering is the
    invariant: whatever the host sent is destroyed before it can be read."""
    assert "b->base = 0" in src or "b->base = NULL" in src
    clear = min(
        (src.index(s) for s in ("b->base = 0", "b->base = NULL") if s in src),
        default=-1,
    )
    assert clear != -1
    assert clear < src.index("->fd =="), "clear base before matching on fd"


def test_lookup_is_by_fd(src):
    assert "->fd ==" in src


def test_the_dsp_maps_the_fd_itself(src):
    assert "HAP_mmap" in src


def test_an_unmapped_fd_is_an_error_not_a_zero_base(src):
    """Upstream returns silently with base == 0 when no slot is free, and the
    caller then computes 0 + offset and reads a small bogus address. Fixed."""
    assert "HEXLIB_DSP_ERR_UNMAPPED" in src
    assert "HEXLIB_DSP_ERR_NO_MMAP_SLOT" in src
    assert "HEXLIB_DSP_ERR_MMAP_FAILED" in src


def test_no_abort_on_a_failed_mapping(src):
    """Upstream abort()s. Fail closed means returning a status, not killing the
    process and leaving the host to interpret a dead session."""
    assert "abort()" not in src


def test_tensor_data_is_computed_from_base_plus_offset(src):
    assert "base" in src and "offset" in src
    assert "->data =" in src


def test_resolution_bounds_checks_the_offset(src):
    """A tensor whose offset+nbytes exceeds its buffer must be refused on the
    DSP too. The host checks it, but the host is not the thing being trusted."""
    assert "HEXLIB_DSP_ERR_TRUNCATED" in src or "HEXLIB_DSP_ERR_INVAL_PARAMS" in src
    assert "nbytes" in src


def test_buffer_index_is_range_checked(src):
    assert "n_bufs" in src

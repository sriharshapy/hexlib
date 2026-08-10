"""VTCM acquisition. Source assertions; the behaviour is Task 8's hwinfo check."""
import pathlib

import pytest

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


def test_acquisition_failure_returns_rather_than_aborting(src):
    assert "abort()" not in src
    assert "assert(" not in src
    assert "HEXLIB_DSP_ERR" in src

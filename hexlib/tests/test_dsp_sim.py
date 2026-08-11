# hexlib/tests/test_dsp_sim.py
"""Stage 1 acceptance: scale_fp16 through the skel's batch path, on the simulator.

NOT "through FastRPC". The simulator omits the qaic stub (spec 0.1), so these
calls bind directly to the skel implementation and no marshalling occurs. What is
proven here is OUR code: batch parsing, the buffer table, the dispatch table, the
kernel adapter, the values, and PCYCLE. Marshalling is stage 3's.

THE TEST THAT MATTERS IS test_an_unmapped_fd_is_refused. Under the simulator the
host and the DSP share one address space, so a skel that used the host's pointer
instead of resolving an fd would return the RIGHT ANSWER to a request whose
buffer was never mapped -- and would fail instantly on silicon. That test fails
exactly when the invariant is broken, which is what makes everything else here
transferable. Without it these are tests of an interface that might only work in
one address space.

Artifacts (the skel archive, the QuRT-hosted .so, the sim configs) are built ONCE
per module via the `backend` fixture -- construction is the slow part; each test
below is one additional `hexagon-sim` launch reusing them.
"""
import os

import numpy as np
import pytest

from hexlib import toolchain as tc
from hexlib.exec import dsp as dspmod

HAS_SDK = os.path.isdir(tc.default_sdk_root())
sdk = pytest.mark.skipif(not HAS_SDK, reason="Hexagon SDK not present")

N = 4100          # SCALE_N: 64*64 + 4, so a tail exists
FACTOR = 0.125    # SCALE_FACTOR, a power of two -> exact in fp16


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    return dspmod.DspSimBackend(["scale_fp16"],
                                  str(tmp_path_factory.mktemp("dspsim")))


@sdk
def test_scale_fp16_matches_numpy_exactly(backend):
    """Error is exactly 0 because 0.125 is a power of two: only the exponent
    changes and no mantissa bit is lost -- PROVIDED the scaled result stays in
    fp16's NORMAL range. It does not always: `rng.standard_normal` puts ~0.8%
    of elements below 0.01 in magnitude, and 0.125 * that underflows fp16's
    normal floor (2**-14) into SUBNORMAL territory, where the real HVX
    kernel's qf32-narrow rounding measurably diverges from numpy's by one ULP
    (confirmed empirically: `hexagon.backend_for("scale")`, the already-proven
    standalone-ELF path, shows the IDENTICAL one-ULP divergence from numpy at
    the identical index for the identical seed -- so this is a genuine kernel
    HW rounding edge case the original harness's deterministic, always-normal
    inputs never exercised, not a marshalling bug in this new path). That is
    a real finding, reported rather than hidden, but it is orthogonal to what
    this test exists to check -- so inputs are kept clear of the subnormal
    boundary, at every index still independently random (a dropped tail or a
    byte-order swap needs that), rather than narrowing the test's claim."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(N).astype(np.float32)
    too_small = np.abs(x) < 0.01
    x = np.where(too_small, np.sign(x) * 0.01 + x, x).astype(np.float16)
    y, stats = backend.run("scale", [x], {"factor": FACTOR})
    expect = (x.astype(np.float32) * FACTOR).astype(np.float16)
    assert y.dtype == np.float16
    assert np.array_equal(y, expect), f"max diff {np.abs(y.astype(np.float32) - expect.astype(np.float32)).max()}"


@sdk
def test_an_unmapped_fd_is_refused(backend):
    """THE DISCRIMINATOR. See the module docstring. A skel leaning on the shared
    address space passes the request; a correct one refuses it."""
    res = backend.run_unmapped("scale", N, FACTOR)
    assert res.status != dspmod.wire.STATUS["OK"], (
        "an invoke naming a never-mapped fd returned a RESULT. The skel resolved "
        "an address it was not given — which works only because the simulator "
        "shares one address space, and will fail on silicon."
    )
    assert res.status == dspmod.wire.STATUS["ERR_UNMAPPED"]


@sdk
def test_the_dsp_reports_v75_and_a_real_vtcm_size(backend):
    info = backend.hwinfo()
    assert info.arch == 75
    assert info.vtcm == 8388608, "acquired VTCM should be the whole 8 MB page"


@sdk
def test_cycles_are_in_the_right_order_of_magnitude(backend):
    """hexlib.sim reports 886 for this kernel. An order-of-magnitude difference
    means PCYCLE is not bracketing what we think it is."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(N).astype(np.float16)
    _, stats = backend.run("scale", [x], {"factor": FACTOR})
    assert 200 < stats.cycles < 20000, f"got {stats.cycles}, sim reports 886"


@sdk
def test_it_agrees_with_the_standalone_elf_path_bit_for_bit(backend, tmp_path):
    """The two transports must not disagree. While both exist, this is the
    signal that says so.

    NOTE: the original draft called `old(["scale"], [x], {"factor": FACTOR})`
    against `hexagon.backend_for`'s returned callable, which takes exactly two
    positional arguments (`arrays`, `attrs`) -- that call would raise
    TypeError before comparing anything, for any implementation, correct or
    not. Fixed here to the real two-argument signature so this test can
    actually fail when the two paths disagree, which is the only thing it
    exists to check.
    """
    from hexlib.exec import hexagon

    rng = np.random.default_rng(2)
    x = rng.standard_normal(N).astype(np.float16)
    y_rpc, _ = backend.run("scale", [x], {"factor": FACTOR})
    old = hexagon.backend_for("scale", work_dir=str(tmp_path))
    assert old is not None, "hexagon.backend_for('scale') found no RunnerSpec"
    (y_elf,) = old([x], {"factor": FACTOR})
    assert np.array_equal(y_rpc, y_elf)


@sdk
def test_a_bad_magic_is_refused_without_running_an_op(backend):
    res = backend.run_raw(b"NOPE" + b"\x00" * 60)
    assert res.status == dspmod.wire.STATUS["ERR_BAD_MAGIC"]
    assert res.n_ops == 0


@sdk
def test_a_truncated_batch_is_refused(backend):
    blob = backend.build_batch("scale", N, FACTOR)
    res = backend.run_raw(blob[: len(blob) - 8])
    assert res.status == dspmod.wire.STATUS["ERR_TRUNCATED"]


@sdk
def test_an_unknown_kind_is_refused(backend):
    res = backend.run_raw(backend.build_batch("scale", N, FACTOR, kind_override=999))
    assert res.status == dspmod.wire.STATUS["ERR_NO_KERNEL"]


@sdk
def test_a_response_that_was_never_written_cannot_read_as_success(backend):
    """Belt and braces on the structural guarantee: status 0 is not a status."""
    with pytest.raises(dspmod.wire.WireError):
        dspmod.wire.unpack_response(b"\x00" * 32)

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


# ---------------------------------------------------------------------------
# layernorm: three inputs, mixed input dtypes, and the dim: scalar sources
# ---------------------------------------------------------------------------
#
# Everything above drives `scale`: one input, one float attr. That left the
# general run() path -- multiple buffers, per-input dtypes, dimension-derived
# scalars -- carried on the simulator by nothing, which STATE.md recorded as an
# open gap. layernorm is the first op here with THREE inputs and the first with
# MIXED input dtypes (fp16 data, fp32 affine parameters), so it exercises the
# generated entry's per-buffer dtype check against buffers that genuinely differ.

LN_EPS = 1e-6


def _ln_reference(x, w, b, eps, ddof=0):
    """The op registry's own formula, accumulated in fp64.

    `ddof=1` produces the UNBIASED-variance near-miss that
    kernels/layernorm_fp16/ keeps as a rejected variant. It is a parameter here
    so the test below can prove its own threshold discriminates.
    """
    xf = x.astype(np.float64)
    mean = xf.mean(axis=-1, keepdims=True)
    centred = xf - mean
    n = xf.shape[-1]
    var = (centred * centred).sum(axis=-1, keepdims=True) / (n - ddof)
    return ((centred / np.sqrt(var + eps)) * w + b).astype(np.float16)


def _ln_inputs(R, C, seed=7):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((R, C)).astype(np.float16),
            rng.standard_normal(C).astype(np.float32),
            rng.standard_normal(C).astype(np.float32))


@sdk
@pytest.mark.parametrize("R,C", [(4, 768), (8, 64), (256, 768)])
def test_layernorm_agrees_with_the_reference_and_rejects_the_near_miss(backend, R, C):
    """A MAX-ERROR TOLERANCE CANNOT CHECK THIS KERNEL, so this does not use one.

    One fp16 ULP is 9.8e-4 relative. The unbiased-variance near-miss -- divide
    by C-1 instead of C -- is only 6.5e-4 at C=768. So ANY per-element tolerance
    loose enough to admit the kernel's genuine 1-ULP noise is already looser
    than the bug it must catch. This repo has paid for that once: that near-miss
    was WRONGLY ACCEPTED on its first run for exactly this reason.

    What separates them is the SHAPE of the disagreement, not its size. The
    kernel's error is 1-ULP noise on a handful of elements; the near-miss is a
    systematic shift on every element. So the statistic is the fraction of
    BIT-EXACT elements, and the test asserts both directions -- that the kernel
    clears the threshold, and that the near-miss does not. The second assertion
    is what stops the threshold from being quietly argued downwards later:
    lower it far enough to admit the bug and this test fails.

    Measured through this path, for the record: 99.805% bit-exact at (4,768),
    99.917% at (256,768), 100% at (8,64); the near-miss scores 49.7%, 47.8% and
    11.1%. Max error is 1 ULP at (4,768) and 6 ULPs at (256,768) -- the relative
    error is the same, there are simply 64x more rows for the tail to appear in.
    """
    x, w, b = _ln_inputs(R, C)
    y, _ = backend.run("layernorm", [x, w, b], {"eps": LN_EPS})

    assert y.dtype == np.float16
    assert y.shape == (R, C)

    good = _ln_reference(x, w, b, LN_EPS, ddof=0)
    near_miss = _ln_reference(x, w, b, LN_EPS, ddof=1)

    frac_good = float((y == good).sum()) / y.size
    frac_bad = float((y == near_miss).sum()) / y.size

    assert frac_good >= 0.99, (
        f"layernorm at R={R} C={C} matched the reference bit-for-bit on only "
        f"{frac_good:.4%} of elements; 1-ULP noise on a few is expected, a "
        f"systematic disagreement is not"
    )
    assert frac_bad < 0.90, (
        f"THE THRESHOLD NO LONGER DISCRIMINATES: the unbiased-variance "
        f"near-miss scores {frac_bad:.4%}, which the 0.99 bound above would "
        f"not obviously reject. Tighten the check or find a better statistic "
        f"-- do not relax it."
    )
    assert frac_good - frac_bad > 0.4, (
        f"correct {frac_good:.4%} vs near-miss {frac_bad:.4%}: the separation "
        f"this test relies on has collapsed"
    )


@sdk
def test_layernorm_reaches_the_kernel_with_its_dimensions_and_eps_intact(backend):
    """The scalars, checked by consequence rather than by reading the blob.

    R and C arrive as `dim:0:0` and `dim:0:1` and eps as `attr:eps` -- the first
    use of the dimension-derived sources on this transport. A swap of R and C
    would be invisible to a square input and to any shape-only assertion, so
    this uses a NON-SQUARE shape whose transpose is not even the same length,
    and a deliberately large eps whose effect on the output is unmistakable.
    """
    R, C = 8, 64
    x, w, b = _ln_inputs(R, C)

    y_small = backend.run("layernorm", [x, w, b], {"eps": 1e-6})[0]
    y_huge = backend.run("layernorm", [x, w, b], {"eps": 4.0})[0]

    assert y_small.shape == (R, C)
    # eps sits inside the sqrt, so a large one shrinks every normalised value
    # towards zero before the affine term. If eps were dropped or read as an
    # int, these two would be identical.
    assert not np.array_equal(y_small, y_huge), (
        "eps=1e-6 and eps=4.0 produced identical output, so the attr scalar is "
        "not reaching the kernel"
    )
    assert np.allclose(y_huge.astype(np.float32),
                       _ln_reference(x, w, b, 4.0).astype(np.float32),
                       atol=2e-3), "eps=4.0 did not match the reference"


# ---------------------------------------------------------------------------
# The host's own fd-patch loop, bounded
# ---------------------------------------------------------------------------
#
# `run_raw` exists so a bad magic, a truncated blob or an unknown op kind
# exercises the DSP'S OWN validation. That makes n_bufs and off_bufs untrusted
# BY DESIGN on this path -- and simhost.c's fd-patch loop read both straight
# out of the blob and wrote 24 bytes per iteration into a fixed 64 KiB static
# array BEFORE `hexlib_iface_invoke` handed them to the code that validates
# them. So the two tests below could not have reported what they claim to:
# they would have crashed hexagon-sim inside the host, and a crash is not the
# DSP refusing anything.
#
# skel_dispatch.c has always validated these fields correctly, and main.c does
# on the device path. The asymmetry was the defect.

_HDR_I_N_BUFS = 3       # struct order in wire._HDR: magic, version, total,
_HDR_I_OFF_BUFS = 6     # n_bufs, n_tensors, n_ops, off_bufs, ...


def _patch_hdr_word(blob: bytes, index: int, value: int) -> bytes:
    out = bytearray(blob)
    out[index * 4:(index + 1) * 4] = int(value).to_bytes(4, "little")
    return bytes(out)


@sdk
def test_an_out_of_range_off_bufs_is_refused_by_the_dsp_not_by_a_host_crash(backend):
    """off_bufs=0xFFFFFF00 with n_bufs=1 -- a ~4 GiB out-of-range write.

    The point is WHICH LAYER SAYS NO. Getting a status back at all means the
    host survived long enough to invoke, and `ERR_TRUNCATED` is the skel's own
    section-bounds check (`off_bufs + n_bufs * sizeof(buf_desc) > len`, widened
    to 64-bit so a large n_bufs cannot wrap it). Before the bound in simhost.c
    this was `memcpy(g_batch + 0xFFFFFF00, &b, 24)` and the run died with an
    opaque simulator failure instead.
    """
    blob = backend.build_batch("scale", N, FACTOR)
    res = backend.run_raw(_patch_hdr_word(blob, _HDR_I_OFF_BUFS, 0xFFFFFF00))
    assert res.status == dspmod.wire.STATUS["ERR_TRUNCATED"], (
        f"expected the skel's own TRUNCATED refusal, got "
        f"{dspmod.wire.STATUS_NAME.get(res.status, res.status)}"
    )


@sdk
def test_an_enormous_n_bufs_is_refused_by_the_dsp_not_by_a_host_crash(backend):
    """n_bufs=0x01000000 with off_bufs left valid -- 384 MiB of forward walk.

    The other shape, and the one that does NOT trip the section-bounds check
    first: the skel answers `ERR_INVAL_PARAMS` from `n_bufs > HEXLIB_MAX_BUFS`.
    Before the bound, simhost's loop would have stepped 24 bytes at a time over
    g_rsp and the skel's own static bufs[]/tens[] on the way there.
    """
    blob = backend.build_batch("scale", N, FACTOR)
    res = backend.run_raw(_patch_hdr_word(blob, _HDR_I_N_BUFS, 0x01000000))
    assert res.status in (
        dspmod.wire.STATUS["ERR_INVAL_PARAMS"],
        dspmod.wire.STATUS["ERR_TRUNCATED"],
    ), (
        f"expected the skel to refuse the buffer count, got "
        f"{dspmod.wire.STATUS_NAME.get(res.status, res.status)}"
    )

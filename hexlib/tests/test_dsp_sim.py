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
def test_an_out_of_range_off_bufs_is_recognised_by_the_host_and_left_unpatched(
    backend,
):
    """off_bufs=0xFFFFFF00 with n_bufs=1, and THE STATUS ALONE DOES NOT PIN IT.

    That was this test's first form and removing the bound left it green, which
    is the only reason the real behaviour here is written down. `size_t` is 32
    bits on this target, so `g_batch + 0xFFFFFF00` is `g_batch - 256`: the
    unbounded loop did not fault at all, it wrote 24 bytes into whatever static
    lives before `g_batch` and then invoked normally, and the skel returned the
    same ERR_TRUNCATED it returns now. Silent corruption of a neighbouring
    object, reported as a clean refusal -- worse than the crash the finding
    described, and invisible to any assertion on the response.

    So this pins the HOST's own recognition, printed by simhost.c, which is the
    thing that actually differs between patched and unpatched. The skel's
    verdict is asserted too: an unpatched batch must still be SENT, because
    refusing to send it here would substitute the host's judgement for the
    DSP's, and `run_raw` exists to observe the DSP's.
    """
    blob = backend.build_batch("scale", N, FACTOR)
    res, out = backend.run_raw_verbose(_patch_hdr_word(blob, _HDR_I_OFF_BUFS, 0xFFFFFF00))
    assert "bufs_out_of_range_not_patched" in out, (
        "simhost patched (or silently wrapped past) a buffer table that does "
        "not lie inside the blob; the loop is unbounded again. stdout:\n"
        + out[-2000:]
    )
    assert res.status == dspmod.wire.STATUS["ERR_TRUNCATED"], (
        f"the unpatched batch must still reach the skel and be refused BY the "
        f"skel; got {dspmod.wire.STATUS_NAME.get(res.status, res.status)}"
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
    res, out = backend.run_raw_verbose(_patch_hdr_word(blob, _HDR_I_N_BUFS, 0x01000000))
    assert "bufs_out_of_range_not_patched" in out, (
        "simhost tried to patch 16.7M buffer descriptors out of a 64 KiB "
        "array. stdout:\n" + out[-2000:]
    )
    assert res.status in (
        dspmod.wire.STATUS["ERR_INVAL_PARAMS"],
        dspmod.wire.STATUS["ERR_TRUNCATED"],
    ), (
        f"expected the skel to refuse the buffer count, got "
        f"{dspmod.wire.STATUS_NAME.get(res.status, res.status)}"
    )


# ===========================================================================
# THE FOUR KERNELS ADDED FOR THE ENCODER, EACH CHECKED AGAINST THE OP REGISTRY
# ===========================================================================
#
# WHY THE REGISTRY IS THE ORACLE HERE rather than a reference written in this
# file. Every one of these ops has a numpy `reference` in
# `hexlib/graph/opdefs/`, and that reference IS the specification -- it is what
# the eager executor runs, what the plan executor was validated against, and
# what the kernel author was told to implement. A second reference written here
# would be a second chance to get the same convention wrong, and for three of
# these four the convention is precisely the dangerous part:
#
#   patchify      `merge` reorders the output ROWS into 2x2 spatial-merge-block
#                 order; ignoring it gives the right shape and byte count.
#   rope_2d       split-half pairing (i, i+D/2), not adjacent pairs. Same shape
#                 either way.
#   transpose_hd  perm(0,2,1) vs perm(1,0,2) -- and with two equal dims, even a
#                 stride confusion returns the right answer.
#
# So these tests compare the C kernel, reached through the real batch wire, with
# the numpy oracle the graph itself uses. Disagreement means one of them is
# wrong, which is the finding either way.
#
# AND WHAT THEY PROVE THAT THE KERNEL GATE DOES NOT. The gate compiles a kernel
# against its own harness and never touches the batch path. `layernorm_fp16`
# gated green for a day while `KIND_ID["layernorm"]` answered ERR_NO_KERNEL on
# the wire, because a kernel is dispatchable only once it has a RunnerSpec. Each
# test below drives the op through `pack_batch` -> the skel -> the generated
# entry -> the kernel, so it fails if the spec's scalars, dtypes, layouts or
# buffer order are wrong even when the kernel itself is perfect.


def _oracle(kind, arrays, attrs):
    """The op registry's own numpy reference for `kind`."""
    import hexlib.graph.opdefs  # noqa: F401  -- registers the op definitions
    from hexlib.graph.ops import REGISTRY

    return REGISTRY.get(kind).reference(tuple(arrays), dict(attrs))[0]


def _frac_bit_exact(got, want):
    return float((np.asarray(got) == np.asarray(want)).sum()) / np.asarray(got).size


@sdk
def test_transpose_hd_dispatches_and_matches_the_registry(backend):
    """perm(0,2,1), the variant `select()` has to route away from its sibling.

    B, T and D are three DIFFERENT numbers, because with T == D a kernel that
    confuses the two strides still returns the right answer -- and the sibling
    kernel keeps a near-miss that is exactly that confusion. Exact comparison:
    this op does no arithmetic, so any difference at all is a bug.

    The routing is the other half of what this checks. `transpose` and
    `transpose_hd` are two kernels behind one op kind and the wire carries no
    perm, so the host resolves the variant and names it with its own KIND_ID.
    Send the wrong id and this returns a correctly-shaped transposed-the-other-
    way answer.
    """
    B, T, D = 3, 8, 5
    rng = np.random.default_rng(11)
    x = rng.standard_normal((B, T, D)).astype(np.float16)

    y, _ = backend.run("transpose", [x], {"perm": (0, 2, 1)})

    assert y.shape == (B, D, T)
    want = _oracle("transpose", [x], {"perm": (0, 2, 1)})
    assert np.array_equal(y, want.astype(np.float16)), (
        "the perm(0,2,1) op did not match the registry -- either the kernel is "
        "wrong or it was dispatched to the perm(1,0,2) kernel"
    )
    # And it must NOT equal the other permutation, which is the failure that
    # would otherwise look like success on a square input.
    if B == D:
        other = np.transpose(x, (1, 0, 2))
        assert not np.array_equal(y, other)


@sdk
def test_softmax_dispatches_and_matches_the_registry(backend):
    """Row softmax over the last axis, through the wire.

    NON-SQUARE ON PURPOSE. The encoder's real shape is (12,256,256), whose last
    two dims are equal -- so a softmax along the wrong axis has the SAME shape
    and the same byte count and nothing downstream could notice. (2,3,64) makes
    the wrong axis a different shape, so `_out_shape` and the byte-count check
    would catch it even before the values were compared.

    Also the first exercise of the `rows:` scalar source on this transport: R is
    the product of the two leading axes (2*3 = 6), which no single `dim:` can
    express, and the DSP computes it from `ne` rather than trusting a number the
    host asserted. A wrong `rows:` gives a kernel that softmaxes over the wrong
    number of rows, which on a 3-D input is a plausible wrong answer.
    """
    rng = np.random.default_rng(12)
    x = (rng.standard_normal((2, 3, 64)) * 3.0).astype(np.float16)

    y, _ = backend.run("softmax", [x], {"axis": -1})

    assert y.shape == (2, 3, 64)
    assert y.dtype == np.float16
    want = _oracle("softmax", [x], {"axis": -1}).astype(np.float16)

    # A tolerance is used here rather than bit-exactness, and the reason is
    # written down: the kernel narrows through Q6_Vhf_equals_Wqf32, whose
    # rounding is NOT IEEE round-to-nearest-even, so a 1-ULP disagreement with
    # numpy is expected and is not a defect. 1 ULP at these magnitudes is ~1e-3
    # relative; the bound below is well inside what a real bug would exceed --
    # kernels/softmax_fp16/harness.c measures its own fp16-accumulation
    # near-miss at 7.7% relative, about 80x this bound.
    err = np.abs(y.astype(np.float32) - want.astype(np.float32))
    assert err.max() < 1e-3, f"max abs error {err.max()}"

    # THE PROPERTY, INDEPENDENT OF THE ORACLE: every row sums to 1. This is what
    # catches a normalisation that divides by the count, or by a stale sum, in a
    # way that comparing against a reference computed the same way would not.
    sums = y.astype(np.float32).sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=2e-3), f"row sums {sums}"


@sdk
def test_rope_2d_dispatches_and_matches_the_registry(backend):
    """The split-half rotation, with three inputs and mixed dtypes.

    T, H and D are all DIFFERENT, so a token/head index swap cannot pass by
    coincidence -- and note the cos/sin tables have NO head axis, so indexing
    them by head instead of by token is a plausible stride slip that the
    kernel's own harness keeps as a near-miss.

    D must be even for the split-half pairing to exist at all, and the kernel
    only vectorises D=64; other D take a correct scalar path. D=64 is used here
    because it is the encoder's own head_dim and the path that actually ships.
    """
    T, H, D = 5, 3, 64
    rng = np.random.default_rng(13)
    x = rng.standard_normal((T, H, D)).astype(np.float16)
    # REAL ROTATION TABLES, AND "REAL" MEANS DUPLICATED ACROSS THE TWO HALVES.
    # A split-half rotation pairs element i with i + D/2 and applies cos[i],
    # sin[i] to both, so the pair is a genuine 2-D rotation only when
    # cos[i + D/2] == cos[i] and sin[i + D/2] == sin[i] -- which is exactly how
    # RoPE tables are built, each frequency written into both halves.
    #
    # This is worth the comment because the first version of this test varied
    # the angle across all D=64 positions. The kernel still MATCHED THE ORACLE
    # (max error 3.9e-3, inside the bound), and the norm assertion below failed
    # anyway -- because with cos[i + 32] != cos[i] the operation being applied
    # is not a rotation and has no reason to preserve anything. The test was
    # wrong, not the kernel. A property assertion is only as good as the inputs
    # that make the property true.
    half = D // 2
    ang_half = (np.arange(T)[:, None] * 0.1 + np.arange(half)[None, :] * 0.02)
    ang = np.concatenate([ang_half, ang_half], axis=1)
    cos = np.cos(ang).astype(np.float32)
    sin = np.sin(ang).astype(np.float32)

    y, _ = backend.run("rope_2d", [x, cos, sin], {})

    assert y.shape == (T, H, D)
    assert y.dtype == np.float16
    want = _oracle("rope_2d", [x, cos, sin], {}).astype(np.float16)
    err = np.abs(y.astype(np.float32) - want.astype(np.float32))
    assert err.max() < 4e-3, f"max abs error {err.max()}"

    # THE PROPERTY: a rotation preserves the norm of each (i, i+D/2) pair. This
    # holds for the split-half convention and FAILS for adjacent pairing, so it
    # is an oracle-independent check on the one thing most likely to be wrong.
    def pair_norms(arr):
        a = arr.astype(np.float32)
        return a[..., :half] ** 2 + a[..., half:] ** 2

    assert np.allclose(pair_norms(y), pair_norms(x), rtol=5e-2, atol=5e-3), (
        "the split-half pair norms changed, so this is not a rotation of the "
        "(i, i+D/2) pairs -- the likely cause is adjacent pairing"
    )


@sdk
def test_patchify_dispatches_and_matches_the_registry(backend):
    """The encoder's first op: rank-4 fp32 in, fp32 out, eight scalars.

    THE ONLY TEST HERE THAT REACHES `dim:0:3`, and the only one with four attr
    params. `patch`, `merge`, `grid_h` and `grid_w` cannot be recovered from the
    shapes, so all four cross the wire -- and `merge` CHANGES THE ANSWER rather
    than describing it, reordering the output rows into 2x2 spatial-merge-block
    order. A dropped `merge` param gives raster order: right shape, right bytes,
    wrong rows.

    A small shape, because patchify at the encoder's own (3,2,256,256) costs
    about 2.0M cycles and this is a dispatch test, not a benchmark.

    THE SHAPE HAD TO BE CHOSEN CAREFULLY AND THE FIRST CHOICE WAS DEGENERATE.
    With grid_h=4, grid_w=2, merge=2 the merge-block order is IDENTICAL to
    raster order: Bw = grid_w/merge = 1, so bw is always 0 and
    `((bh*Bw + bw)*merge + mh)*merge + mw` collapses to the raster index. The
    kernel was correct and the reordering assertion below could not see it
    either way. grid_h=6, grid_w=4 gives Bh=3, Bw=2 -- both greater than one, so
    the two orders genuinely differ -- and they stay DIFFERENT from each other so
    a grid transpose cannot pass either. merge=2 divides both, which the registry
    requires.
    """
    C, T, patch, merge = 3, 2, 3, 2
    grid_h, grid_w = 6, 4
    H, W = grid_h * patch, grid_w * patch
    attrs = {"patch": patch, "merge": merge, "grid_h": grid_h, "grid_w": grid_w,
             "temporal_patch": T}
    rng = np.random.default_rng(14)
    img = rng.standard_normal((C, T, H, W)).astype(np.float32)

    y, _ = backend.run("patchify", [img], attrs)

    assert y.shape == (grid_h * grid_w, C * T * patch * patch)
    assert y.dtype == np.float32
    want = _oracle("patchify", [img], attrs).astype(np.float32)
    # fp32 throughout and no arithmetic at all, so this must be BIT-EXACT.
    assert np.array_equal(y, want), (
        f"patchify disagreed with the registry on "
        f"{(y != want).sum()} of {y.size} elements. Pure data movement in fp32 "
        f"has no rounding to blame."
    )

    # AND THE MERGE REORDERING SPECIFICALLY. Raster order is what a kernel that
    # ignores `merge` produces; it is the same shape, so only the values differ.
    x = img.reshape(C, T, grid_h, patch, grid_w, patch)
    raster = x.transpose(2, 4, 0, 1, 3, 5).reshape(grid_h * grid_w, -1)
    assert not np.array_equal(y, raster), (
        "the output is in raster order, so `merge` was ignored -- the "
        "downstream merger is a pure reshape and needs 2x2 blocks"
    )


@sdk
def test_matmul_dispatches_and_matches_the_reference(backend):
    """A batched fp16 matmul through the DSP batch path.

    Bn > 1 so a wrong batch stride cannot pass, and K is not a multiple of the
    kernel's 64-wide accumulator block so the scalar tail runs.
    """
    rng = np.random.default_rng(5)
    Bn, M, K, N = 3, 8, 70, 128
    a = rng.standard_normal((Bn, M, K)).astype(np.float16)
    b = rng.standard_normal((Bn, K, N)).astype(np.float16)

    y, _ = backend.run("matmul", [a, b], {})
    want = a.astype(np.float32) @ b.astype(np.float32)

    assert y.shape == want.shape, f"{y.shape} != {want.shape}"
    assert np.max(np.abs(y - want)) < 1e-2 * max(1.0, float(np.max(np.abs(want))))


@sdk
def test_matmul_reduces_over_k_and_not_over_a_transposed_operand(backend):
    """Oracle-independent. Build B so that every column is a distinct constant:
    then C[b,m,n] must equal n * sum(A[b,m,:]), which a kernel that read B as
    [N,K] cannot reproduce for a non-square operand."""
    rng = np.random.default_rng(6)
    # N IS A MULTIPLE OF 64 ON PURPOSE. kernels/matmul_fp16/kernel.c loads B
    # rows with an ALIGNED vector read (`(const HVX_Vector *) brow` then
    # `bv[i]`), so a row is only correctly aligned when N % 64 == 0. N=96
    # trips that and this test would fail for a reason that has nothing to do
    # with the property it exists to check. The bug is real and is Task 6;
    # both encoder matmul shapes use N=256 and N=64, so it does not affect
    # dispatch.
    Bn, M, K, N = 2, 4, 32, 128
    a = rng.standard_normal((Bn, M, K)).astype(np.float16)
    b = np.tile(np.arange(N, dtype=np.float16), (Bn, K, 1))

    y, _ = backend.run("matmul", [a, b], {})
    row_sums = a.astype(np.float32).sum(axis=2)              # (Bn, M)
    want = row_sums[:, :, None] * np.arange(N, dtype=np.float32)

    assert y.shape == (Bn, M, N)
    assert np.max(np.abs(y - want)) < 1e-2 * max(1.0, float(np.max(np.abs(want))))


@sdk
def test_matmul_is_correct_when_n_is_not_a_multiple_of_64(backend):
    """N=96 is not a multiple of 64, so B's rows are not 128-byte aligned.

    The vectorised path used to load them with an ALIGNED read, so the first
    nvec64*64 columns were computed from shifted data while the scalar tail
    was correct. Regression guard for that fix; the gate now covers this
    shape too (spec.json's N is 200).
    """
    rng = np.random.default_rng(21)
    Bn, M, K, N = 2, 4, 32, 96
    a = rng.standard_normal((Bn, M, K)).astype(np.float16)
    b = rng.standard_normal((Bn, K, N)).astype(np.float16)

    y, _ = backend.run("matmul", [a, b], {})
    want = a.astype(np.float32) @ b.astype(np.float32)
    assert np.max(np.abs(y - want)) < 1e-2 * max(1.0, float(np.max(np.abs(want))))


# ---------------------------------------------------------------------------
# matmul_epilogue: fp16 activations, a q4_0 weight, an fp32 bias, and a string
# `act` attr that has to cross the wire as an int code.
#
# THE ORACLE TRAP: the registry's own `matmul_epilogue` reference multiplies by
# the FULL-PRECISION weight. The kernel multiplies by the q4_0-QUANTIZED
# weight, whose per-block rounding error dwarfs anything a kernel bug could
# add. So the expected value here is built from `dequantize_q4_0` of the SAME
# bytes handed to the kernel -- the only way left to compare is the
# arithmetic, not the quantization format.
# ---------------------------------------------------------------------------


@sdk
@pytest.mark.parametrize("act", ["none", "gelu_tanh", "gelu_erf"])
def test_matmul_epilogue_dispatches_for_every_activation(backend, act):
    """Bias then activation, against a reference built from the SAME q4_0
    bytes the kernel gets -- see the oracle trap above."""
    from hexlib.exec.quant import dequantize_q4_0, quantize_q4_0
    from hexlib.exec.runner import RawTensor
    from hexlib.graph.ops import get

    rng = np.random.default_rng(11)
    M, K, N = 12, 96, 160
    a = rng.standard_normal((M, K)).astype(np.float16)
    w = rng.standard_normal((K, N)).astype(np.float32)
    bias = rng.standard_normal((N,)).astype(np.float32)

    w_bytes = quantize_q4_0(w)
    w_raw = RawTensor(dtype="q4_0", shape=w.shape, data=w_bytes)

    y, _ = backend.run("matmul_epilogue", [a, w_raw, bias], {"act": act})

    w_eff = dequantize_q4_0(w_bytes, w.shape)
    ref = a.astype(np.float32) @ w_eff + bias
    want = ref if act == "none" else get(act).reference((ref,), {})[0]

    assert y.shape == (M, N)
    assert np.max(np.abs(y - want)) < 1e-2 * max(1.0, float(np.max(np.abs(want))))


@sdk
def test_matmul_epilogue_applies_bias_before_activation(backend):
    """Oracle-independent: with gelu_erf and a large negative bias every output
    is driven to ~0. Adding the bias AFTER the activation cannot produce that --
    the bias would still be visible in the result."""
    from hexlib.exec.quant import quantize_q4_0
    from hexlib.exec.runner import RawTensor

    rng = np.random.default_rng(12)
    M, K, N = 8, 64, 64
    a = rng.standard_normal((M, K)).astype(np.float16)
    w = rng.standard_normal((K, N)).astype(np.float32)
    bias = np.full((N,), -50.0, dtype=np.float32)

    w_raw = RawTensor(dtype="q4_0", shape=w.shape, data=quantize_q4_0(w))
    y, _ = backend.run("matmul_epilogue", [a, w_raw, bias], {"act": "gelu_erf"})
    assert np.max(np.abs(y)) < 1.0, (
        "a large negative bias applied BEFORE gelu must collapse the output; "
        f"max |y| = {float(np.max(np.abs(y)))} means bias came after the "
        "activation"
    )

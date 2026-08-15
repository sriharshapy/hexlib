# hexlib/tests/test_quant_q4_0.py
"""`hexlib.exec.quant` against an INDEPENDENT scalar transcription of
llama.cpp's `quantize_row_q4_0_ref`.

WHY A SECOND IMPLEMENTATION RATHER THAN GOLDEN BYTES. A committed golden blob
would pin the output without saying what it means, and the failure mode here is
not "the bytes changed" — it is "the bytes are a valid q4_0 encoding of the wrong
thing". Three ways to get that, all of which round-trip to plausible values:

  * pairing nibble j with j+1 instead of j with j+16 (a permutation of the right
    values: right norm, right histogram, wrong everywhere);
  * `amax / 8` instead of `max / -8` (flips the sign of every value in blocks
    whose extreme element is positive);
  * `np.round` instead of truncate-after-adding-8.5 (banker's rounding, so it
    disagrees on every exact .5 and nowhere else — invisible on random data).

The scalar reference below is written from `ggml-quants.c:113-146` statement by
statement, in a loop, with no numpy vectorisation, so it shares no code and no
broadcasting with the implementation. Where they agree, they agree for a reason.

The vectorised version is also the one hexlib will use to feed real q4_0 weights
to `matmul_epilogue`, so a disagreement here is a wrong weight matrix on the DSP.
"""
import numpy as np
import pytest

from hexlib.exec.quant import (
    BLOCK_BYTES,
    QK4_0,
    dequantize_q4_0,
    quantize_q4_0,
)
from hexlib.graph import ir


def _scalar_quantize(x):
    """`quantize_row_q4_0_ref`, transcribed. One block at a time, no numpy."""
    flat = np.asarray(x, dtype=np.float32).reshape(-1)
    assert flat.size % QK4_0 == 0
    out = bytearray()
    for i in range(flat.size // QK4_0):
        blk = flat[i * QK4_0 : (i + 1) * QK4_0]
        amax, mx = 0.0, 0.0
        for v in blk:
            v = float(v)
            if amax < abs(v):
                amax, mx = abs(v), v
        d = np.float32(mx / -8.0)
        # Stored as fp16, so everything downstream sees the narrowed value.
        d16 = np.float16(d)
        du = np.float32(d16)
        idv = np.float32(1.0 / du) if du != 0.0 else np.float32(0.0)
        out += bytes(np.array([d16], dtype=np.float16).view(np.uint8))
        for j in range(QK4_0 // 2):
            x0 = np.float32(blk[j]) * idv
            x1 = np.float32(blk[j + QK4_0 // 2]) * idv
            # C: MIN(15, (int8_t)(x + 8.5f)) -- truncation toward zero.
            xi0 = min(15, int(np.trunc(x0 + np.float32(8.5))))
            xi1 = min(15, int(np.trunc(x1 + np.float32(8.5))))
            xi0 = max(0, xi0)
            xi1 = max(0, xi1)
            out.append(xi0 | (xi1 << 4))
    return bytes(out)


def test_the_block_geometry_comes_from_ir():
    """18 bytes per 32 elements, and the constants are ir's, not a second copy.
    A drifted block size is a wrong answer at full speed, not a crash."""
    assert QK4_0 == 32
    assert BLOCK_BYTES == 18
    assert ir.nbytes((1, 32), "q4_0") == BLOCK_BYTES
    assert ir.nbytes((768, 768), "q4_0") == 768 * 768 // 32 * 18


@pytest.mark.parametrize("shape", [(1, 32), (3, 64), (8, 96), (768, 32)])
def test_agrees_with_the_scalar_transcription_byte_for_byte(shape):
    rng = np.random.default_rng(5)
    x = rng.standard_normal(shape).astype(np.float32) * 3.0
    got = quantize_q4_0(x)
    want = _scalar_quantize(x)
    assert len(got) == ir.nbytes(shape, "q4_0")
    assert got == want, (
        f"the vectorised quantizer disagrees with the scalar transcription of "
        f"llama.cpp's own loop on {sum(a != b for a, b in zip(got, want))} of "
        f"{len(want)} bytes"
    )


def test_the_rounding_mode_is_truncate_after_adding_8_point_5():
    """THE THIRD HAZARD, AND IT NEEDS CONSTRUCTED INPUT TO BE VISIBLE AT ALL.

    `trunc(x + 8.5)` and `round(x + 8.0)` agree everywhere except where `x` is
    exactly a half-integer — numpy's `round` is banker's rounding, so it sends
    0.5 to 0 and 1.5 to 2 while truncation-after-8.5 sends them to 1 and 2. On
    random data that difference has measure zero, and swapping one for the other
    passed every other test in this file. Found by mutation, which is the only
    reason this test exists.

    So the input is built to land on exact halves. An extreme of -8.0 makes
    `d = max / -8 = 1.0` (exactly representable in fp16, so the fp16 round trip
    of the scale changes nothing) and therefore `id = 1.0` — which means the
    scaled values ARE the input values, and half-integer inputs hit the boundary
    directly.

    The expected codes are worked out by hand rather than taken from the
    implementation: for value v, `trunc(v + 8.5)` clamped to [0, 15].
    """
    half_ints = np.array([-7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5,
                          0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], dtype=np.float32)
    # -8.0 sets the scale; the rest are exact halves. 32 elements total.
    blk = np.concatenate([
        np.array([-8.0], dtype=np.float32),
        half_ints,
        np.array([-8.0], dtype=np.float32),
        half_ints,
    ])
    assert blk.size == QK4_0

    raw = np.frombuffer(quantize_q4_0(blk.reshape(1, -1)), dtype=np.uint8)
    d = raw[:2].copy().view(np.float16)[0]
    assert d == np.float16(1.0), f"expected an exact scale of 1.0, got {d}"

    qs = raw[2:]
    got = np.concatenate([(qs & 0x0F), (qs >> 4)]).astype(np.int32)

    expect = np.clip(np.trunc(blk + np.float32(8.5)), 0, 15).astype(np.int32)
    assert np.array_equal(got, expect), (
        f"codes {got.tolist()} but truncate-after-8.5 gives {expect.tolist()}. "
        f"numpy's round() would give "
        f"{np.clip(np.round(blk + np.float32(8.0)), 0, 15).astype(int).tolist()}"
    )
    # And the two rounding modes really do differ on this input, so the
    # assertion above is not vacuous.
    bankers = np.clip(np.round(blk + np.float32(8.0)), 0, 15).astype(np.int32)
    assert not np.array_equal(expect, bankers), (
        "this input no longer distinguishes the two rounding modes, so the "
        "assertion above proves nothing -- pick values that hit exact halves"
    )

    # The scalar transcription must agree byte for byte on it too.
    assert quantize_q4_0(blk.reshape(1, -1)) == _scalar_quantize(blk)


def test_blocks_whose_extreme_value_is_POSITIVE_get_a_negative_scale():
    """`d = max / -8` with `max` SIGNED — the second hazard in the docstring.

    A block whose largest-magnitude element is positive must store a NEGATIVE
    scale. `amax / 8` would store a positive one and flip the sign of every
    dequantized value in that block, which is why this is asserted directly on
    the stored bytes rather than only through a round trip.
    """
    x = np.linspace(0.5, 4.0, QK4_0, dtype=np.float32)     # all positive
    raw = np.frombuffer(quantize_q4_0(x.reshape(1, -1)), dtype=np.uint8)
    d = raw[:2].copy().view(np.float16)[0]
    assert d < 0, f"scale {d} should be negative for an all-positive block"

    y = np.linspace(-4.0, -0.5, QK4_0, dtype=np.float32)   # all negative
    raw = np.frombuffer(quantize_q4_0(y.reshape(1, -1)), dtype=np.uint8)
    d = raw[:2].copy().view(np.float16)[0]
    assert d > 0, f"scale {d} should be positive for an all-negative block"


def test_the_low_nibble_is_element_j_and_the_high_nibble_is_element_j_plus_16():
    """THE PAIRING, asserted on the bytes.

    A block built so the two halves are clearly distinguishable: the first 16
    elements near the negative extreme (small quant codes) and the second 16 near
    zero (codes near 8). If the pairing were j with j+1, the low and high nibbles
    of each byte would BOTH come from the first half and both be small.
    """
    x = np.concatenate([
        np.full(16, -1.0, dtype=np.float32),
        np.full(16, 0.0, dtype=np.float32),
    ])
    raw = np.frombuffer(quantize_q4_0(x.reshape(1, -1)), dtype=np.uint8)
    qs = raw[2:]
    lo = qs & 0x0F
    hi = qs >> 4
    # first half is the extreme -> code 0; second half is zero -> code 8
    assert set(lo.tolist()) == {0}, f"low nibbles {sorted(set(lo.tolist()))}"
    assert set(hi.tolist()) == {8}, f"high nibbles {sorted(set(hi.tolist()))}"


def test_a_dequantized_block_recovers_its_extreme_value_exactly():
    """The element that set the scale must come back essentially exact — its
    quant code is 0 and `(0 - 8) * d == max`. This is the one value the format
    represents without error, so it is the sharpest available check on the
    scale."""
    rng = np.random.default_rng(6)
    x = rng.standard_normal((4, 32)).astype(np.float32) * 2.0
    back = dequantize_q4_0(quantize_q4_0(x), x.shape)
    for r in range(x.shape[0]):
        j = int(np.abs(x[r]).argmax())
        # fp16 scale storage is the only loss here.
        assert abs(back[r, j] - x[r, j]) <= abs(x[r, j]) * 1e-3 + 1e-6, (
            f"row {r}: extreme {x[r, j]} came back as {back[r, j]}"
        )


def test_the_round_trip_error_is_what_a_4_bit_format_can_do_and_no_worse():
    """A real bound, not a smoke test. 16 levels spanning [-8d, 7d] means the
    quantization step is |d| and the worst case is about half a step, so the
    error must be under ~amax/8 per element. Asserted as a fraction of each
    block's own amax, because a global bound would be dominated by whichever
    block happened to have the largest values."""
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((32, 64)) * 5.0).astype(np.float32)
    back = dequantize_q4_0(quantize_q4_0(x), x.shape)

    blocks = x.reshape(-1, QK4_0)
    got = back.reshape(-1, QK4_0)
    amax = np.abs(blocks).max(axis=1)
    err = np.abs(got - blocks).max(axis=1)
    assert np.all(err <= amax / 8.0 * 1.02 + 1e-4), (
        f"worst block error {(err / amax).max():.4f} of amax; a 4-bit format "
        f"with 16 levels should stay under 1/8"
    )
    # And it must not be TRIVIALLY good, which would mean the test is measuring
    # nothing -- 4-bit really does lose information.
    assert err.max() > amax.max() / 100.0, (
        "round-trip error is suspiciously small for a 4-bit format; is the "
        "dequantizer reading back a stored copy?"
    )


def test_an_all_zero_block_is_handled_and_round_trips_to_zero():
    """`d == 0` makes `1/d` a division by zero; llama.cpp guards it with
    `id = d ? 1/d : 0`. Every code then lands on 8, and `(8-8)*0 == 0`."""
    x = np.zeros((2, 32), dtype=np.float32)
    data = quantize_q4_0(x)
    raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, BLOCK_BYTES)
    assert np.all(raw[:, 2:] == 0x88), "every quant code should be 8"
    assert np.array_equal(dequantize_q4_0(data, x.shape), x)


def test_a_last_axis_that_is_not_a_multiple_of_the_block_is_refused():
    with pytest.raises(ValueError, match="multiple of 32"):
        quantize_q4_0(np.zeros((4, 33), dtype=np.float32))


def test_dequantize_refuses_a_buffer_of_the_wrong_length():
    with pytest.raises(ValueError, match="bytes"):
        dequantize_q4_0(b"\x00" * 17, (1, 32))

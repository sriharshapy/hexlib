# hexlib/tests/test_quant_q8_0.py
"""q8_0, held to the ggml reference it was transcribed from.

WHY THIS FORMAT EXISTS HERE. Measured against transformers on the shipped
Qwen3.5-0.8B vision weights at 256x256, fp32 arithmetic on both sides so nothing
but the weight format differs:

    q4_0   encoder output cosine 0.867606
    q8_0   encoder output cosine 0.999002

for 1.89x the weight bytes. Mixed precision was measured first and rejected:
the error is DIFFUSE, so keeping the merger, the patch embedding and all of
attention at higher precision while the MLPs stay q4_0 still only reaches 0.913.

WHAT THESE TESTS ARE REALLY FOR. q8_0 sits next to q4_0 in the same file and
looks like its bigger sibling, which makes carrying q4_0's habits across the
obvious mistake. Three of them produce a plausible wrong answer rather than an
error, and each has a test below: the sign of `d`, the rounding mode, and the
element ordering.
"""
import numpy as np
import pytest

from hexlib.exec.quant import (
    QK8_0,
    Q8_0_BLOCK_BYTES,
    dequantize_q8_0,
    quantize_q8_0,
)
from hexlib.graph import ir


def _scalar_quantize(block: np.ndarray) -> bytes:
    """`quantize_row_q8_0_ref` (ggml-quants.c:276-299) transcribed elementwise,
    in Python, with no numpy vectorisation -- so it shares no code path with the
    implementation and a shared bug cannot cancel."""
    import struct

    amax = 0.0
    for v in block:
        amax = max(amax, abs(float(v)))
    d = np.float16(amax / 127.0)
    df = float(np.float32(d))
    idv = (1.0 / df) if df != 0.0 else 0.0
    out = bytearray(struct.pack("<e", d))
    for v in block:
        x0 = float(np.float32(np.float32(v) * np.float32(idv)))
        # roundf: half away from zero.
        r = int(np.sign(x0) * np.floor(abs(x0) + 0.5))
        out.append(struct.pack("<b", max(-128, min(127, r)))[0])
    return bytes(out)


@pytest.mark.parametrize("shape", [(1, 32), (4, 32), (3, 64), (8, 128)])
def test_agrees_with_the_scalar_transcription_byte_for_byte(shape):
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(shape) * 3.0).astype(np.float32)
    got = quantize_q8_0(x)
    want = b"".join(_scalar_quantize(row) for row in x.reshape(-1, QK8_0))
    assert got == want


def test_the_block_is_34_bytes_and_nbytes_agrees():
    """`ir.nbytes` is the authority the wire and the arena both size from; a
    disagreement here is an arena that is short by a fixed fraction."""
    assert Q8_0_BLOCK_BYTES == 34
    assert QK8_0 == 32
    x = np.zeros((4, 64), dtype=np.float32)
    assert len(quantize_q8_0(x)) == ir.nbytes((4, 64), "q8_0")
    assert ir.nbytes((768, 768), "q8_0") / ir.nbytes((768, 768), "q4_0") == pytest.approx(34 / 18)


def test_the_scale_is_POSITIVE_even_when_the_extreme_is_negative():
    """THE q4_0 HABIT THAT MUST NOT CARRY OVER. q4_0 stores `signed_max / -8`,
    so a block whose extreme is positive gets a NEGATIVE scale -- it has its own
    test asserting exactly that. q8_0 stores `amax / 127`, which has no sign at
    all. Carrying q4_0's trick here negates every value in the block."""
    for block in (
        np.linspace(-4.0, -0.5, QK8_0, dtype=np.float32),   # all negative
        np.linspace(0.5, 4.0, QK8_0, dtype=np.float32),     # all positive
    ):
        raw = np.frombuffer(quantize_q8_0(block.reshape(1, -1)), dtype=np.uint8)
        d = raw[:2].copy().view(np.float16)[0]
        assert d > 0, f"q8_0 scale must be positive, got {d}"


def test_the_rounding_is_half_away_from_zero_and_not_bankers():
    """`roundf`, not `np.round`. Built so the scaled values land on exact
    halves: amax = 127 makes d = 1.0 exactly (representable in fp16), so the
    scaled value IS the input and .5 inputs hit the boundary directly.

    The assertion is paired with a proof that the two modes actually differ on
    this input -- otherwise it would pass for a banker's-rounding
    implementation and prove nothing."""
    halves = np.array([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5], dtype=np.float32)
    blk = np.concatenate([
        np.array([127.0], dtype=np.float32),        # sets d = 1.0
        np.tile(halves, 3),
        np.zeros(QK8_0 - 1 - 24, dtype=np.float32),
    ])
    assert blk.size == QK8_0

    raw = np.frombuffer(quantize_q8_0(blk.reshape(1, -1)), dtype=np.uint8)
    assert raw[:2].copy().view(np.float16)[0] == np.float16(1.0)
    got = raw[2:].copy().view(np.int8).astype(np.int64)

    away = (np.sign(blk) * np.floor(np.abs(blk) + 0.5)).astype(np.int64)
    bankers = np.round(blk).astype(np.int64)
    assert np.array_equal(got, away), f"got {got.tolist()}, away-from-zero {away.tolist()}"
    assert not np.array_equal(away, bankers), (
        "this input no longer distinguishes the two rounding modes, so the "
        "assertion above proves nothing"
    )


def test_the_quants_are_in_element_order_with_no_bias_and_no_nibble_pairing():
    """q4_0 packs element j and element j+16 into one byte and biases codes by
    8. q8_0 does neither: byte j is element j, signed. A block that is
    monotonically increasing must come back monotonically increasing."""
    blk = np.linspace(-4.0, 4.0, QK8_0, dtype=np.float32)
    raw = np.frombuffer(quantize_q8_0(blk.reshape(1, -1)), dtype=np.uint8)
    q = raw[2:].copy().view(np.int8).astype(np.int64)
    assert q.size == QK8_0
    assert np.all(np.diff(q) >= 0), f"not monotonic: {q.tolist()}"
    assert q.min() < 0 < q.max(), "codes are signed, not biased into [0, 255]"


def test_the_extreme_element_does_not_wrap_to_minus_128():
    """`amax / 127` keeps |scaled| <= 127 in fp32, but `d` is NARROWED to fp16
    before use and narrowing downward makes the scaled value slightly larger --
    so the extreme can land on 128 and wrap to -128 without a clamp. A wrap
    flips the sign of the single largest element in the block, which is the
    worst possible element to get wrong."""
    rng = np.random.default_rng(7)
    for trial in range(200):
        blk = (rng.standard_normal(QK8_0) * 10 ** rng.uniform(-3, 3)).astype(np.float32)
        raw = np.frombuffer(quantize_q8_0(blk.reshape(1, -1)), dtype=np.uint8)
        q = raw[2:].copy().view(np.int8)
        j = int(np.abs(blk).argmax())
        assert np.sign(int(q[j])) == np.sign(blk[j]) or blk[j] == 0, (
            f"trial {trial}: extreme {blk[j]} quantized to {q[j]} -- sign flipped"
        )


def test_the_round_trip_is_idempotent():
    """`test_encoder_on_sim.py` pre-quantizes its weight feeds so both paths
    multiply identical values, which requires quantize(dequantize(q)) == q. q4_0
    has this property; a q8_0 that lost it would make that test flap rather than
    fail, which is worse."""
    rng = np.random.default_rng(3)
    for seed in range(6):
        w = (rng.standard_normal((8, 64)) * 2.0).astype(np.float32)
        a = quantize_q8_0(w)
        assert quantize_q8_0(dequantize_q8_0(a, w.shape)) == a, f"seed {seed}"


def test_it_is_meaningfully_better_than_q4_0_on_the_same_data():
    """The whole reason this format was added. Not a tolerance -- a COMPARISON,
    so it cannot pass by being loose."""
    from hexlib.exec.quant import dequantize_q4_0, quantize_q4_0

    rng = np.random.default_rng(11)
    w = (rng.standard_normal((32, 128)) * 1.5).astype(np.float32)
    e4 = np.abs(dequantize_q4_0(quantize_q4_0(w), w.shape) - w).max()
    e8 = np.abs(dequantize_q8_0(quantize_q8_0(w), w.shape) - w).max()
    assert e8 < e4 / 4, (
        f"q8_0 max error {e8:.3e} is not meaningfully below q4_0's {e4:.3e}; "
        f"four extra bits should buy roughly 16x"
    )


def test_a_last_axis_that_is_not_a_multiple_of_the_block_is_refused():
    with pytest.raises(ValueError, match="multiple of 32"):
        quantize_q8_0(np.zeros((4, 30), dtype=np.float32))


def test_a_wrong_length_buffer_is_refused_rather_than_reshaped():
    data = quantize_q8_0(np.zeros((4, 32), dtype=np.float32))
    with pytest.raises(ValueError, match="bytes"):
        dequantize_q8_0(data, (4, 64))

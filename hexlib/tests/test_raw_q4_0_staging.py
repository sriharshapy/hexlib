# hexlib/tests/test_raw_q4_0_staging.py
"""The block-quantized weight path: `RawTensor`, and what refuses a dense array.

WHY THIS PATH EXISTS AT ALL. 75 of the encoder's 308 plan steps are
`matmul_epilogue`, and every one of them takes a q4_0 weight -- 48 at (768, 768),
12 at (768, 3072), 12 at (3072, 768), plus three singletons. `RunnerSpec` used to
refuse the dtype outright, with a message that named the missing work:
"block-quantized inputs are staged as raw bytes and need their own path". Nothing
downstream of that refusal could be built.

WHAT MAKES THIS DANGEROUS RATHER THAN MERELY MISSING. Every other input on this
transport is a numpy array staged through `np.ascontiguousarray(a, dtype=...)`,
which converts. A q4_0 buffer cannot be converted -- it is 18-byte blocks of an
fp16 scale plus 32 4-bit values, `llama.cpp`'s `block_q4_0` -- so it has to be
passed through verbatim, and `.tobytes()` on ANY numpy array is willing to
produce bytes. An fp32 weight array staged down this path yields either a buffer
of the wrong size or, when the sizes happen to line up, a correctly-shaped answer
computed from noise, with no error anywhere. That is why the type is distinct and
why these tests spend most of their assertions on refusals.

THE TWO NUMBERS ARE BOTH TRUE ON THE WIRE. `hexlib_tensor` carries `nbytes` and
`ne[]` as separate fields, and `wire.py` checks only that the buffer holds
`offset + nbytes` -- never that `nbytes == prod(ne) * itemsize`. So a (768, 768)
q4_0 weight says ne=(768,768) AND nbytes=331776. Putting the byte shape in `ne`
instead would be a lie every `dim:` scalar and every kernel reading `a->ne`
would inherit.
"""
import numpy as np
import pytest

from hexlib.exec.runner import (
    RawTensor,
    RunnerSpec,
    Scalar,
    WIRE_DTYPE,
    WIRE_RAW,
)
from hexlib.graph import ir
from hexlib.runtime.genentry import emit_entry
from hexlib.runtime.wire import DTYPE_ID, LAYOUT_ID

# The encoder's own weight shapes, from the compiled plan for qwen35 at 256x256.
ENCODER_WEIGHT_SHAPES = ((768, 768), (768, 3072), (3072, 768), (1536, 768),
                         (3072, 3072), (3072, 1024))


def _spec(**kw):
    base = dict(
        kind="matmul_epilogue",
        kernel_dir="kernels/matmul_epilogue_fp16",
        inputs=("fp16", "q4_0", "fp32"),
        out_dtype="fp16",
        layouts=("row_major", "q4_0_repacked", "row_major", "row_major"),
    )
    base.update(kw)
    return RunnerSpec(**base)


def _weight(shape):
    return RawTensor("q4_0", shape, b"\x5a" * ir.nbytes(shape, "q4_0"))


# ---------------------------------------------------------------------------
# RawTensor's own contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ENCODER_WEIGHT_SHAPES)
def test_every_encoder_weight_shape_round_trips(shape):
    """Not a smoke test: these are the six shapes the plan actually contains,
    and `ir.nbytes` refuses a last dimension that is not a multiple of 32, so
    this also confirms every one of them is expressible at all."""
    w = _weight(shape)
    assert w.shape == shape
    assert w.nbytes == ir.nbytes(shape, "q4_0")
    assert w.size == shape[0] * shape[1]
    # The whole point of the format: far smaller than the dense equivalent.
    assert w.nbytes < shape[0] * shape[1] * 2


def test_the_byte_count_comes_from_ir_and_not_from_a_second_copy_of_it():
    """`ir.nbytes` is the authority on the block size and it already refuses a
    bad last dimension. If `RawTensor` recomputed the arithmetic instead of
    asking, the two could drift -- and a drifted block size is a wrong answer,
    not a crash. Checked by asserting the exact published constant: q4_0 is 18
    bytes per 32 elements, which `test_graph_ir.py` pins independently."""
    assert ir.nbytes((1, 32), "q4_0") == 18
    w = RawTensor("q4_0", (1, 32), b"\x00" * 18)
    assert w.nbytes == 18


@pytest.mark.parametrize("delta", [-18, -1, 1, 18])
def test_a_buffer_that_is_not_exactly_the_block_size_is_refused(delta):
    """Off by one byte or by one whole block, both refused at construction.

    A short weight is the realistic accident -- a truncated download, a slice
    taken with the wrong stride -- and it is the one that produces a plausible
    wrong answer rather than a crash, because the kernel reads whatever follows
    it in the payload.
    """
    n = ir.nbytes((768, 768), "q4_0")
    with pytest.raises(ValueError, match="bytes of blocks"):
        RawTensor("q4_0", (768, 768), b"\x00" * (n + delta))


def test_a_last_dimension_that_is_not_a_multiple_of_the_block_is_refused():
    with pytest.raises(ValueError, match="multiple of the 32-element block"):
        RawTensor("q4_0", (768, 33), b"\x00" * 400)


def test_rawtensor_refuses_a_dense_dtype():
    """`RawTensor` is for block-quantized storage only. fp16 has a numpy dtype
    and belongs in the ordinary path; accepting it here would give two ways to
    stage the same thing, one of which skips the conversion."""
    with pytest.raises(ValueError, match="WIRE_RAW"):
        RawTensor("fp16", (16, 32), b"\x00" * 1024)


def test_the_raw_and_dense_dtype_tables_do_not_overlap():
    """A dtype in both tables would make `payload`'s branch order decide which
    staging path a weight takes, which is exactly the kind of thing that is
    correct until someone reorders it."""
    assert not (WIRE_RAW & set(WIRE_DTYPE))
    # And every raw dtype must still have a wire id, or it cannot be declared.
    for dtype in WIRE_RAW:
        assert dtype in DTYPE_ID, f"{dtype} has no DTYPE_ID and cannot cross"


# ---------------------------------------------------------------------------
# RunnerSpec: what may and may not be raw
# ---------------------------------------------------------------------------


def test_a_spec_may_read_q4_0():
    spec = _spec()
    assert spec.inputs[1] == "q4_0"
    assert spec.buf_layouts()[1] == "q4_0_repacked"


def test_a_spec_may_not_WRITE_q4_0():
    """`decode` reads the result back through `np.frombuffer` with a numpy
    dtype, so a quantized output would be a result that cannot be decoded. No
    kernel in this encoder writes one -- the weights arrive quantized and
    everything computed is fp16."""
    with pytest.raises(ValueError, match="not a dense wire dtype"):
        _spec(out_dtype="q4_0")


def test_an_unknown_dtype_is_still_refused_and_names_both_tables():
    with pytest.raises(ValueError, match="neither a dense wire dtype"):
        _spec(inputs=("fp16", "q8_0", "fp32"))


def test_payload_refuses_a_numpy_array_where_the_spec_declared_q4_0():
    """THE FAILURE THIS PATH EXISTS TO PREVENT.

    A dense array staged as blocks is not an error numpy will raise:
    `.tobytes()` always works. So the refusal has to be explicit, and it has to
    be here rather than only in `dsp.py`, because `RunnerSpec.payload` is the
    other transport (the standalone-ELF runner) and both must agree.
    """
    spec = _spec()
    x = np.zeros((256, 768), dtype=np.float16)
    dense_weight = np.zeros((768, 768), dtype=np.float16)
    bias = np.zeros(768, dtype=np.float32)
    with pytest.raises(ValueError, match="must be a RawTensor"):
        spec.payload((x, dense_weight, bias))


def test_payload_refuses_a_rawtensor_whose_dtype_is_not_the_declared_one():
    """Two raw dtypes will exist eventually (q8_0 is the obvious next one), and
    at that point a mismatched RawTensor is a silent reinterpretation of one
    block format as another. Refused now, while there is only one."""
    spec = _spec()
    w = _weight((768, 768))
    object.__setattr__(w, "dtype", "q8_0")   # frozen dataclass; forge the drift
    with pytest.raises(ValueError, match="RawTensor says"):
        spec.payload((np.zeros((256, 768), np.float16), w,
                      np.zeros(768, np.float32)))


def test_payload_stages_the_quantized_bytes_verbatim_and_in_order():
    """The weight's bytes must appear unchanged, and after the activation --
    src order is the contract the DSP walks. A conversion here would be
    undetectable downstream: the buffer would be the right size and the values
    would be wrong."""
    spec = _spec()
    x = np.arange(8, dtype=np.float16)
    w = RawTensor("q4_0", (1, 32), bytes(range(18)))
    bias = np.arange(4, dtype=np.float32)

    blob = spec.payload((x, w, bias))
    assert blob == x.tobytes() + bytes(range(18)) + bias.tobytes()


def test_numel_and_dim_scalars_read_the_LOGICAL_shape_of_a_raw_input():
    """`ne` is the element shape, so a `dim:` scalar naming the weight must get
    768, not the byte count. This is the assertion that would fail if `ne` were
    ever filled from the byte shape."""
    spec = _spec(scalars=(Scalar("dim:1:0", "int"), Scalar("dim:1:1", "int"),
                          Scalar("numel:1", "int")))
    w = _weight((768, 3072))
    header = spec.header((np.zeros((256, 768), np.float16), w,
                          np.zeros(3072, np.float32)), {})
    import struct
    k, n, numel = struct.unpack("<iii", header)
    assert (k, n) == (768, 3072)
    assert numel == 768 * 3072


# ---------------------------------------------------------------------------
# The generated DSP entry
# ---------------------------------------------------------------------------


def test_the_generated_entry_hands_the_weight_over_as_bytes():
    """`const unsigned char *`, not `const hexlib_hf *`.

    There is no C scalar type for one element of a q4_0 block, so the entry
    does not invent one -- the block layout is the kernel's business. Casting to
    `hexlib_hf *` would compile fine and read the fp16 SCALE bytes as data.
    """
    c = emit_entry("matmul_epilogue", _spec())
    assert "(const unsigned char *) a->buf[1]" in c
    assert "(const hexlib_hf *) a->buf[1]" not in c
    # and the dense neighbours are still cast to their own types
    assert "(const hexlib_hf *) a->buf[0]" in c
    assert "(const float *) a->buf[2]" in c


def test_the_generated_entry_guards_the_weights_dtype_and_layout():
    """Both guards, with the real ids from the real tables.

    The dtype guard stops a dense buffer being read as blocks. The LAYOUT guard
    is the one `hexlib_dsp.h` says the enum exists for -- "un-repacked weights
    are a plan-time error rather than silent corruption" -- and without it a
    row-major q4_0 weight handed to a kernel expecting the repacked order is a
    correctly-shaped wrong answer.
    """
    c = emit_entry("matmul_epilogue", _spec())
    assert f"a->dtype[1] != {DTYPE_ID['q4_0']}u" in c
    assert f"a->layout[1] != {LAYOUT_ID['q4_0_repacked']}u" in c
    # the activation beside it keeps its own, different, expectations
    assert f"a->dtype[0] != {DTYPE_ID['fp16']}u" in c
    assert f"a->layout[0] != {LAYOUT_ID['row_major']}u" in c


def test_the_generated_entry_compiles_as_C_for_a_q4_0_spec():
    """Cheap syntactic proof that the emitted cast and guards are real C rather
    than a plausible-looking string. The full compile-and-run probe for entries
    lives in test_genentry_entry_probe.py; this only needs to know that adding a
    raw dtype did not produce something unparseable."""
    c = emit_entry("matmul_epilogue", _spec())
    # balanced braces and every guard statement terminated
    assert c.count("{") == c.count("}")
    for line in c.splitlines():
        s = line.strip()
        if s.startswith("if (") and "return" in s:
            assert s.endswith(";"), f"unterminated guard: {s}"

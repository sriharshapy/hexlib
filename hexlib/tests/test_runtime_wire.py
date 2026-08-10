"""The wire format, and the invariant that makes a simulator pass mean anything.

THE HOST WRITES FDS AND OFFSETS; THE DSP WRITES ADDRESSES. `base` and `data` are
DSP-side scratch. If the host could put an address in either, then under the
simulator -- where host and DSP share one address space -- a skel that used it
would work perfectly and fail instantly on silicon. So the serializer writing
zeros there is asserted, not assumed: it is the field-level half of the guarantee
whose behavioural half is Task 8's unmapped-fd test.
"""
import struct

import pytest

from hexlib.runtime import wire


def test_magic_is_HXLB_little_endian():
    assert wire.BATCH_MAGIC == 0x424C5848
    assert struct.pack("<I", wire.BATCH_MAGIC) == b"HXLB"


def test_ok_is_one_never_zero():
    """A zero-filled response buffer that was never written must not read as OK."""
    assert wire.STATUS["OK"] == 1
    assert 0 not in wire.STATUS.values()


def test_header_says_its_own_total_size():
    blob = wire.pack_batch(
        bufs=[wire.BufDesc(fd=7, size=8192)],
        tensors=[wire.TensorDesc(bi=0, offset=0, nbytes=64, dtype="fp16",
                                 layout="row_major", ne=(32, 1, 1, 1))],
        ops=[wire.OpDesc(kind=3, params=(0,) * 16, src=(0,), dst=(0,))],
    )
    total = struct.unpack_from("<I", blob, 8)[0]
    assert total == len(blob)


def test_host_writes_zero_into_buf_base():
    blob = wire.pack_batch(
        bufs=[wire.BufDesc(fd=7, size=8192)],
        tensors=[],
        ops=[],
    )
    off_bufs = struct.unpack_from("<I", blob, 24)[0]
    base = struct.unpack_from("<Q", blob, off_bufs)[0]
    assert base == 0, "base is DSP-side scratch; a host address must not cross"


def test_host_writes_zero_into_tensor_data():
    blob = wire.pack_batch(
        bufs=[wire.BufDesc(fd=7, size=8192)],
        tensors=[wire.TensorDesc(bi=0, offset=128, nbytes=64, dtype="fp16",
                                 layout="row_major", ne=(32, 1, 1, 1))],
        ops=[],
    )
    off_t = struct.unpack_from("<I", blob, 28)[0]
    # data is the 10th uint32 in hexlib_tensor: bi, offset, nbytes, dtype,
    # layout, ne[0..3], data
    data = struct.unpack_from("<I", blob, off_t + 9 * 4)[0]
    assert data == 0


def test_BufDesc_has_no_way_to_express_an_address():
    """Not 'the host chooses not to'. There is no field."""
    assert not hasattr(wire.BufDesc(fd=1, size=1), "base")
    t = wire.TensorDesc(bi=0, offset=0, nbytes=1, dtype="fp16",
                        layout="row_major", ne=(1, 1, 1, 1))
    assert not hasattr(t, "data")


def test_response_round_trip():
    raw = struct.pack("<IIIIQII", wire.BATCH_MAGIC, 1, 1, 2, 12345, 75, 0)
    raw += struct.pack("<IIQ", 3, 1, 886)
    raw += struct.pack("<IIQ", 4, 1, 1139)
    rsp = wire.unpack_response(raw)
    assert rsp.status == 1 and rsp.arch == 75 and rsp.cycles_total == 12345
    assert [r.cycles for r in rsp.results] == [886, 1139]


def test_response_with_bad_magic_is_refused():
    raw = struct.pack("<IIIIQII", 0xDEADBEEF, 1, 1, 0, 0, 75, 0)
    with pytest.raises(wire.WireError, match="magic"):
        wire.unpack_response(raw)


def test_all_zero_response_is_refused_not_read_as_success():
    with pytest.raises(wire.WireError):
        wire.unpack_response(b"\x00" * 32)


def test_truncated_response_is_refused():
    raw = struct.pack("<IIIIQII", wire.BATCH_MAGIC, 1, 1, 2, 0, 75, 0)
    raw += struct.pack("<IIQ", 3, 1, 886)  # claims 2 results, carries 1
    with pytest.raises(wire.WireError, match="truncated"):
        wire.unpack_response(raw)


def test_too_many_buffers_is_refused_before_the_dsp_sees_it():
    with pytest.raises(wire.WireError, match="HEXLIB_MAX_BUFS"):
        wire.pack_batch(
            bufs=[wire.BufDesc(fd=i, size=64) for i in range(9)],
            tensors=[], ops=[],
        )


def test_tensor_naming_a_nonexistent_buffer_is_refused():
    with pytest.raises(wire.WireError, match="buffer index"):
        wire.pack_batch(
            bufs=[wire.BufDesc(fd=7, size=64)],
            tensors=[wire.TensorDesc(bi=3, offset=0, nbytes=8, dtype="fp16",
                                     layout="row_major", ne=(4, 1, 1, 1))],
            ops=[],
        )


def test_tensor_running_past_its_buffer_is_refused():
    with pytest.raises(wire.WireError, match="past the end"):
        wire.pack_batch(
            bufs=[wire.BufDesc(fd=7, size=64)],
            tensors=[wire.TensorDesc(bi=0, offset=32, nbytes=64, dtype="fp16",
                                     layout="row_major", ne=(32, 1, 1, 1))],
            ops=[],
        )


def test_c_header_agrees_with_python_on_every_constant():
    """One source of truth, checked. A silent disagreement here is a wrong answer
    on the DSP, not a compile error."""
    import pathlib
    src = pathlib.Path("hexlib/runtime/skel/hexlib_dsp.h").read_text()
    assert "0x424C5848u" in src
    assert "#define HEXLIB_MAX_BUFS 8" in src
    assert "#define HEXLIB_MAX_SRC 6" in src
    assert "#define HEXLIB_MAX_DST 4" in src
    assert "#define HEXLIB_MAX_PARAMS 16" in src
    assert "HEXLIB_DSP_OK = 1" in src
    for name, val in wire.STATUS.items():
        assert f"= {val}" in src, f"status {name} missing from the C header"

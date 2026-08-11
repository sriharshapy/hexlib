# hexlib/tests/test_exec_dsp_host.py
"""Everything `DspSimBackend.run()` decides BEFORE and AFTER the one simulator
launch, with the launch itself replaced by a fake.

WHY A SEPARATE FILE FROM test_dsp_sim.py. That file is the acceptance gate: it
builds the skel, the QuRT-hosted `.so` and the sim configs and then launches
`hexagon-sim` once per test -- minutes, and an SDK. None of the decisions
checked here need any of that, and every one of them is a decision whose
failure mode is a CORRECTLY-SHAPED WRONG ANSWER rather than a crash:

  * dispatching an op to a kernel that implements a different permutation
    (`requires`, enforced on the host because `hexlib_args` carries no field
    for a perm at all -- see genentry.py's own note);
  * reading a PREVIOUS call's `hexlib_out.bin` as this call's result;
  * accepting a response that answers for a different op kind, or for a
    different number of ops, than the batch asked about;
  * silently ignoring an extra input array.

Each of those returns plausible values with an OK status, so only a test that
inspects the host's own bookkeeping can catch it. Replacing `run_sim` is what
makes that testable at all: it lets a run be given a deliberately stale, absent,
or mismatched artifact, which a real simulator would never produce on demand.

WHAT THIS FILE DOES NOT PROVE. That the real simulator writes the files this
fake writes -- `simhost.c` does (it writes `hexlib_rsp.bin` on every invoke that
returns, and `hexlib_out.bin` only when the batch status is OK, and returns
`rh.status == HEXLIB_DSP_OK ? 0 : 1`), and test_dsp_sim.py is what checks it end
to end. This file checks what `run()` does with those artifacts.
"""
import os
import struct

import numpy as np
import pytest

from hexlib.exec import dsp as dspmod
from hexlib.runtime import wire
from hexlib.runtime.genentry import KIND_ID

OK = wire.STATUS["OK"]


def _backend(work_dir):
    """A `DspSimBackend` with no artifacts built.

    `__init__` compiles the skel archive, links the QuRT-hosted `.so` and
    writes the sim configs; none of that is reachable from the host-side
    decisions under test. Every attribute `run()` reads is set explicitly here,
    so if `run()` grows a dependency on another one this raises AttributeError
    rather than quietly skipping a check.
    """
    b = object.__new__(dspmod.DspSimBackend)
    b.work_dir = str(work_dir)
    b.sdk_root = os.path.join(str(work_dir), "no-such-sdk")  # run_sim is faked
    os.makedirs(b.work_dir, exist_ok=True)
    return b


def _rsp(results, status=OK, n_ops=None):
    """A batch response blob, packed with `wire.py`'s OWN format strings rather
    than a retyped copy of them -- so this helper cannot drift from the format
    `unpack_response` reads. `n_ops` defaults to `len(results)`; passing it
    explicitly is how the "claims more results than it carries" case is built.
    """
    raw = struct.pack(
        wire._RSP_HDR, wire.BATCH_MAGIC, wire.BATCH_VERSION, status,
        len(results) if n_ops is None else n_ops, 886, 75, 0,
    )
    for kind, st, cycles in results:
        raw += struct.pack(wire._RESULT, kind, st, cycles)
    return raw


def _scale_buffer(x, factor):
    """The whole shared rpcmem buffer as `simhost.c` dumps it: the input
    payload, zero-padded to the 128-byte boundary `run()` aligns the output to,
    then the output region. Built from `dspmod._align_up` rather than a literal
    so it tracks the backend's own alignment rule."""
    y = (x.astype(np.float32) * factor).astype(np.float16)
    in_end = dspmod._align_up(x.nbytes)
    return x.tobytes().ljust(in_end, b"\x00") + y.tobytes(), y


class _FakeSim:
    """One stand-in for `dsp.run_sim`. Records every launch, optionally writes
    an output and/or a response file, and returns whatever `SimHostResult` the
    test asks for -- including the combination `simhost.c` produces when it
    prints an OK invoke line and then dies before writing a file."""

    def __init__(self, work_dir, out=None, rsp=None, status=OK, exit_code=0):
        self.work_dir = str(work_dir)
        self.out = out
        self.rsp = rsp
        self.status = status
        self.exit_code = exit_code
        self.launches = 0

    def __call__(self, work_dir, extra_args=(), sdk_root=None):
        self.launches += 1
        if self.out is not None:
            with open(os.path.join(self.work_dir, dspmod.OUT_NAME), "wb") as f:
                f.write(self.out)
        if self.rsp is not None:
            with open(os.path.join(self.work_dir, dspmod.RSP_NAME), "wb") as f:
                f.write(self.rsp)
        return dspmod.SimHostResult(
            status=self.status, cycles=886, arch=75, vtcm=8388608,
            stdout="SIMHOST fake", exit_code=self.exit_code,
        )


class _NeverLaunches:
    """A `run_sim` replacement that fails if it is ever called. Used by every
    test whose claim is that a request is refused BEFORE the simulator runs:
    asserting only that an exception was raised would also pass if the refusal
    happened afterwards, on the wrong grounds."""

    def __call__(self, *a, **kw):
        raise AssertionError(
            "the simulator was launched for a request that must be refused on "
            "the host, before anything is packed"
        )


# --- F1: `requires` is enforced on the host, because nothing else can ---------


def test_run_refuses_a_perm_the_kernel_does_not_implement(tmp_path, monkeypatch):
    """THE FINDING. `transpose_th_fp16` implements perm (1,0,2). A perm (0,2,1)
    op has a DIFFERENT output shape, which `_out_shape` computes from the
    requested perm -- so the byte count matches, the status is OK, and the
    caller gets an attention layout with the wrong permutation that every
    downstream shape check accepts.

    The DSP cannot catch this: `hexlib_args` has no field carrying a
    permutation (genentry.py emits an honest comment instead of a check that
    could not fail). So the host is the only place it can be refused, and
    `hexlib/exec/hexagon.py` has always done so -- this path did not.
    """
    monkeypatch.setattr(dspmod, "run_sim", _NeverLaunches())
    b = _backend(tmp_path)
    x = np.zeros((4, 3, 2), dtype=np.float16)
    with pytest.raises(ValueError, match=r"perm"):
        b.run("transpose", [x], {"perm": (0, 2, 1)})
    assert not os.listdir(tmp_path), (
        "the batch must not even be written for an op this kernel cannot serve"
    )


def test_run_refuses_a_4d_input_to_the_3d_transpose_kernel(tmp_path, monkeypatch):
    """The variant: the entry passes T,H,D from `ne[0][0..2]` and ignores
    `ne[0][3]`, so a 4-D input moves a fraction of its elements and reports OK.
    A 4-D op cannot have perm (1,0,2) (a permutation names every axis), so the
    same host check refuses it -- which is the point: the check is on the
    ATTRIBUTE the caller supplied, so it covers shapes the kernel never
    considered."""
    monkeypatch.setattr(dspmod, "run_sim", _NeverLaunches())
    b = _backend(tmp_path)
    x = np.zeros((2, 4, 3, 2), dtype=np.float16)
    with pytest.raises(ValueError, match=r"perm"):
        b.run("transpose", [x], {"perm": (1, 0, 2, 3)})


def test_run_refuses_a_cast_to_a_dtype_the_kernel_does_not_produce(tmp_path, monkeypatch):
    """THE SECOND FINDING, AND WHY THE DSP-SIDE CHECK CANNOT COVER IT.
    `cast` is a general op kind; this kernel only does fp32 -> fp16. The
    generated entry does check `a->dtype[out]`, but `run()` stamps that field
    from `spec.out_dtype` -- so on the wire it compares the spec against
    itself and passes by construction, whatever the caller asked for. Only a
    check against the CALLER'S OWN attr can fail here, and that check lives on
    the host."""
    monkeypatch.setattr(dspmod, "run_sim", _NeverLaunches())
    b = _backend(tmp_path)
    x = np.zeros(16, dtype=np.float32)
    with pytest.raises(ValueError, match=r"dtype"):
        b.run("cast", [x], {"dtype": "fp32"})


def test_the_permutation_the_kernel_does_implement_still_runs(tmp_path, monkeypatch):
    """The control. A guard that refused everything would satisfy the two tests
    above, so the accepted case must be shown to reach the simulator and come
    back with the reordered shape."""
    b = _backend(tmp_path)
    x = np.arange(2 * 3 * 4, dtype=np.float16).reshape(2, 3, 4)
    in_end = dspmod._align_up(x.nbytes)
    moved = np.transpose(x, (1, 0, 2))
    buf = x.tobytes().ljust(in_end, b"\x00") + moved.tobytes()
    fake = _FakeSim(tmp_path, out=buf,
                    rsp=_rsp([(KIND_ID["transpose"], OK, 886)]))
    monkeypatch.setattr(dspmod, "run_sim", fake)

    y, stats = b.run("transpose", [x], {"perm": (1, 0, 2)})
    assert fake.launches == 1
    assert y.shape == (3, 2, 4)
    assert np.array_equal(y, moved)
    assert stats.calls == 1 and stats.cycles == 886


def test_scale_round_trips_through_the_faked_launch(tmp_path, monkeypatch):
    """The other control: an op with no `requires` at all is unaffected, and
    the output is sliced at the aligned offset the batch declared."""
    b = _backend(tmp_path)
    x = np.arange(37, dtype=np.float16)
    buf, expect = _scale_buffer(x, 0.5)
    fake = _FakeSim(tmp_path, out=buf, rsp=_rsp([(KIND_ID["scale"], OK, 886)]))
    monkeypatch.setattr(dspmod, "run_sim", fake)

    y, _ = b.run("scale", [x], {"factor": 0.5})
    assert np.array_equal(y, expect)


# --- F3: a stale artifact is not this call's result ---------------------------


def test_a_stale_output_is_not_read_as_this_calls_result(tmp_path, monkeypatch):
    """THE FINDING. One backend, two calls. Call 1 wrote `hexlib_out.bin`. Call
    2's simulator prints its `SIMHOST invoke ... status=1` line and then dies
    before `fopen("hexlib_out.bin","wb")`. Nothing deleted the old file, so
    `run()` sliced call 1's buffer at the same offset and returned it as call
    2's answer -- with `res.exit_code` sitting unread in the result."""
    b = _backend(tmp_path)
    x1 = np.full(37, 4.0, dtype=np.float16)
    stale, stale_y = _scale_buffer(x1, 0.5)
    with open(tmp_path / dspmod.OUT_NAME, "wb") as f:
        f.write(stale)
    with open(tmp_path / dspmod.RSP_NAME, "wb") as f:
        f.write(_rsp([(KIND_ID["scale"], OK, 886)]))

    # Call 2: an OK status line, then death. Writes nothing.
    monkeypatch.setattr(dspmod, "run_sim",
                        _FakeSim(tmp_path, out=None, rsp=None, exit_code=1))
    x2 = np.full(37, 1.0, dtype=np.float16)
    assert stale_y[0] == 2.0  # call 1's values: what must never be returned here
    with pytest.raises(dspmod.DspSimError):
        b.run("scale", [x2], {"factor": 0.5})
    assert not os.path.exists(tmp_path / dspmod.OUT_NAME), (
        "the previous call's output must be GONE before the launch, not merely "
        "unread -- otherwise the next code path that reads it inherits the bug"
    )


def test_a_stale_response_is_not_read_as_this_calls_result(tmp_path, monkeypatch):
    """The same hazard one file over. `hexlib_rsp.bin` is what says WHICH op
    answered and with what status; a leftover one from a previous call would
    vouch for a launch that never wrote anything."""
    b = _backend(tmp_path)
    with open(tmp_path / dspmod.RSP_NAME, "wb") as f:
        f.write(_rsp([(KIND_ID["scale"], OK, 886)]))
    x = np.arange(37, dtype=np.float16)
    buf, _ = _scale_buffer(x, 0.5)
    monkeypatch.setattr(dspmod, "run_sim", _FakeSim(tmp_path, out=buf, rsp=None))
    with pytest.raises(dspmod.DspSimError, match=dspmod.RSP_NAME):
        b.run("scale", [x], {"factor": 0.5})


def test_a_nonzero_exit_code_is_not_a_success(tmp_path, monkeypatch):
    """`simhost.c` returns 0 if and only if the batch status was
    HEXLIB_DSP_OK, so an OK status line together with a nonzero exit means the
    process died after printing it. The result carried `exit_code` and nothing
    read it."""
    b = _backend(tmp_path)
    x = np.arange(37, dtype=np.float16)
    buf, _ = _scale_buffer(x, 0.5)
    monkeypatch.setattr(dspmod, "run_sim", _FakeSim(
        tmp_path, out=buf, rsp=_rsp([(KIND_ID["scale"], OK, 886)]), exit_code=1))
    with pytest.raises(dspmod.DspSimError, match=r"exit"):
        b.run("scale", [x], {"factor": 0.5})


# --- F2: the response must answer for the op that was asked ------------------


def test_a_response_for_a_different_kind_is_refused(tmp_path, monkeypatch):
    """THE RENUMBER SCENARIO, host side. If the DSP's dispatch table and this
    host's `KIND_ID` ever disagree, an op is served by the wrong kernel: the
    element count matches, both pointers are non-null, the status is OK. The
    response carries the kind the DSP actually dispatched (`skel_dispatch.c`
    fills `results[i].kind = op.kind` before the lookup), so comparing it with
    what was packed is the one host-side check that can see the drift."""
    b = _backend(tmp_path)
    x = np.arange(37, dtype=np.float16)
    buf, _ = _scale_buffer(x, 0.5)
    monkeypatch.setattr(dspmod, "run_sim", _FakeSim(
        tmp_path, out=buf, rsp=_rsp([(KIND_ID["transpose"], OK, 886)])))
    with pytest.raises(dspmod.DspSimError, match=r"kind"):
        b.run("scale", [x], {"factor": 0.5})


def test_a_response_for_a_different_number_of_ops_is_refused(tmp_path, monkeypatch):
    b = _backend(tmp_path)
    x = np.arange(37, dtype=np.float16)
    buf, _ = _scale_buffer(x, 0.5)
    monkeypatch.setattr(dspmod, "run_sim", _FakeSim(
        tmp_path, out=buf,
        rsp=_rsp([(KIND_ID["scale"], OK, 886), (KIND_ID["scale"], OK, 12)])))
    with pytest.raises(dspmod.DspSimError, match=r"1 op|n_ops"):
        b.run("scale", [x], {"factor": 0.5})


def test_a_per_op_failure_under_an_ok_batch_status_is_refused(tmp_path, monkeypatch):
    """The batch header's status and the per-op status are two different
    fields. `wire.BatchResponse.ok` already requires both; `run()` only ever
    looked at the one parsed off stdout."""
    b = _backend(tmp_path)
    x = np.arange(37, dtype=np.float16)
    buf, _ = _scale_buffer(x, 0.5)
    monkeypatch.setattr(dspmod, "run_sim", _FakeSim(
        tmp_path, out=buf,
        rsp=_rsp([(KIND_ID["scale"], wire.STATUS["ERR_REQUIRES"], 0)])))
    with pytest.raises(dspmod.DspSimError, match=r"ERR_REQUIRES"):
        b.run("scale", [x], {"factor": 0.5})


# --- Minor: an arity mismatch is refused, not truncated ----------------------


def test_more_input_arrays_than_the_spec_declares_is_refused(tmp_path, monkeypatch):
    """`zip(arrays, spec.inputs)` stops at the shorter one, so an `add` op
    mis-wired with three inputs computed a+b, ignored c, and returned the right
    shape with no error. `RunnerSpec.payload` raises on the other transport for
    exactly this."""
    monkeypatch.setattr(dspmod, "run_sim", _NeverLaunches())
    b = _backend(tmp_path)
    a = np.zeros(8, dtype=np.float16)
    with pytest.raises(dspmod.DspSimError, match=r"3|inputs"):
        b.run("add", [a, a, a], {})


def test_fewer_input_arrays_than_the_spec_declares_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(dspmod, "run_sim", _NeverLaunches())
    b = _backend(tmp_path)
    a = np.zeros(8, dtype=np.float16)
    with pytest.raises(dspmod.DspSimError, match=r"1|inputs"):
        b.run("add", [a], {})

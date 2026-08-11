"""A DSP-runtime backend for the simulator: the skel's batch path, in-process.

THIS IS NOT FASTRPC, AND THE MODULE IS NAMED TO NOT IMPLY IT IS ONE. See
`docs/superpowers/specs/2026-08-10-silicon-path-runtime-design.md` sections 0.1
and 0.1.1: the simulator omits the qaic-generated stub entirely (it would define
the exact same symbol names as the skel's own implementation, so linking both
into one module is a duplicate-symbol error, not merely redundant), so
`simhost.c`'s calls to `hexlib_iface_open/_start/_mmap/_invoke/_stop/_close`
bind DIRECTLY to `skel.c`, as plain intra-module C function calls. No argument
marshalling happens anywhere on this path. An earlier draft of the design spec
claimed the opposite and was corrected in place; do not reintroduce the error
here by naming this class or module after the transport it does not exercise.

WHAT A PASS THROUGH THIS BACKEND PROVES, AND WHAT IT DOES NOT (spec `0.1.1`):

| exercised here                                             | NOT exercised here      |
|--------------------------------------------------------------|--------------------------|
| batch blob parsing and validation (`hexlib_dispatch_batch`)   | qaic argument marshalling|
| the fd->address table and the pointer-free invariant (`skel_bufs.c`) | real ION allocation |
| the kernel dispatch table (`hexlib_kernel_table`)             | `fastrpc_mmap`           |
| kernel numerical correctness, through real argument unpacking | unsigned PD loading      |
| PCYCLE measurement, read from the response header             | the aarch64 host binary  |
| VTCM acquisition, under the QuRT-hosted `.so`                  |                          |

What IS proven is still most of hexlib's own risk: the batch format, the buffer
table, the dispatch table and the generated kernel adapters are all ours.
Marshalling is vendor code neither written nor fixable here, and is stage 3's
concern (a real device), not this one's.

THE DISCRIMINATOR THAT MAKES A SIMULATOR PASS MEAN ANYTHING. Under the
simulator, host and DSP share ONE address space, and the SDK's own simulator
`HAP_mmap` is `return (void*)(uintptr_t)fd;` while `rpcmem_to_fd` is
`return (int)(uintptr_t)po;` -- so the whole pointer -> fd -> map -> base chain
is an IDENTITY FUNCTION here. A skel that leaned on the host's pointer instead
of resolving an fd through its own mmap table would compute the RIGHT ANSWER on
the simulator and fail instantly on silicon, and no comparison of VALUES could
ever catch that. Only a TABLE LOOKUP can: `hexlib_bufs_map` refuses any fd
`hexlib_bufs_register` never populated, whatever that fd numerically equals.
`run_unmapped` below drives exactly that path (`simhost.c`'s `--unmapped` mode,
which deliberately skips registering the buffer) and is the load-bearing test
in `hexlib/tests/test_dsp_sim.py`.

ALIGNMENT. Kernels such as `scale_fp16` cast their buffer pointers straight to
`HVX_Vector *` and dereference them with an ALIGNED vector load/store -- see
`kernels/scale_fp16/kernel_api.h`'s own alignment note. The file-based runner
path (`hexlib/exec/hexagon.py`) gets this for free because its two buffers are
separate, independently 128-byte-aligned C arrays. Here both input and output
share ONE rpcmem buffer (there is exactly one fd), so every tensor's start
offset within it is rounded up to the next 128-byte boundary explicitly
(`_align_up`) -- getting this wrong would not be caught by any test that only
checks a status code, and would show up as silently wrong values or a fault.

FAIL CLOSED, matching `hexlib/sim.py`'s own rule (its docstring explains why: a
device-farm job once ran zero tests and reported passing). A simulator run that
produces no parseable `SIMHOST hwinfo`/`SIMHOST invoke` line raises, and never
returns a default result -- there is no code path here that can manufacture a
`SimHostResult` without both having been read off the process's own stdout.
`HEXLIB_DSP_OK` is 1, never 0, so a response nothing wrote cannot read as
success either (`wire.unpack_response` rejects it directly).

ONE NON-OBVIOUS BUILD DETAIL. Under the QuRT-hosted packaging, `sim_qurt_command`
launches a real QuRT kernel whose relative `fopen` READS resolve through
`--usefs`, but whose WRITES land in the launching process's OWN working
directory instead. `run_sim` below sets the subprocess `cwd` to the work
directory for exactly this reason (`_cwd`, matching
`hexlib/tests/test_runtime_sim_build.py`'s own helper) -- getting this wrong
does not fail loudly, it silently reads a stale `hexlib_out.bin` (or none) from
wherever the caller's own process happened to be running, which presents as a
wrong kernel rather than a wrong working directory.
"""
from __future__ import annotations

import contextlib
import os
import re
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from hexlib import toolchain as tc
from hexlib.exec.runner import RawTensor, RunnerSpec, SPECS, WIRE_DTYPE, WIRE_RAW
from hexlib.runtime import build as rb
from hexlib.runtime import wire
from hexlib.runtime.genentry import KIND_ID

BATCH_NAME = "hexlib_batch.bin"
IN_NAME = "hexlib_in.bin"
RSP_NAME = "hexlib_rsp.bin"
OUT_NAME = "hexlib_out.bin"

# Matches simhost.c's own `#define MAX_BLOB (16 * 1024 * 1024)` -- the fixed
# size of the ONE rpcmem buffer it always allocates, regardless of what any
# particular batch actually needs.
MAX_BLOB = 16 * 1024 * 1024

# HVX vector width for the fp16 lane count kernels like scale_fp16 assume --
# see the module docstring's "ALIGNMENT" paragraph.
HVX_ALIGN = 128

_HWINFO_RE = re.compile(r"SIMHOST hwinfo arch=(\d+) threads=(\d+) vtcm=(\d+)")
_INVOKE_RE = re.compile(
    r"SIMHOST invoke rc=(-?\d+) rsp_len=(\d+) status=(\d+) n_ops=(\d+) cycles=(\d+)"
)


class DspSimError(Exception):
    pass


@dataclass(frozen=True)
class RunnerStats:
    """Deliberately NOT `hexlib.exec.hexagon.RunnerStats` (off limits to
    modify, and shaped around file-based per-call bookkeeping this backend
    does not do). `cycles` is read from the batch response header -- PCYCLE
    around the kernel call only, never the simulator's whole-program count."""

    calls: int = 0
    cycles: int = 0


@dataclass(frozen=True)
class SimHostResult:
    """One `hexagon-sim` launch's outcome, parsed from its `SIMHOST` lines.

    `status`/`cycles` come from the `SIMHOST invoke` line (the batch response
    header: `hexlib_batch_rsp_hdr.status`/`.cycles_total`). `arch`/`vtcm` come
    from the `SIMHOST hwinfo` line. `exit_code` is the simulator PROCESS's own
    exit code (0 only when the batch's own status was `HEXLIB_DSP_OK` -- see
    `simhost.c`'s `main`'s return statement), not a field taken off the wire.
    """

    status: int
    cycles: int
    arch: int
    vtcm: int
    stdout: str
    exit_code: int | None


@contextlib.contextmanager
def _cwd(path: str):
    """`tc.run` takes no `cwd` kwarg. See `sim_qurt_command`'s own docstring
    and `hexlib/tests/test_runtime_sim_build.py`'s identically-named helper:
    under the QuRT-hosted launch, `fopen(..., "wb")` writes land in the
    LAUNCHING process's cwd, never in `--usefs`'s directory, so a caller that
    does not pin its own cwd here would write `hexlib_rsp.bin`/
    `hexlib_out.bin` into wherever pytest happened to be invoked from --
    which, run from the repo root, previously and accidentally committed two
    stray artifacts."""
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def run_sim(work_dir: str, extra_args: Sequence[str] = (),
            sdk_root: str | None = None) -> SimHostResult:
    """Launch the QuRT-hosted `.so` `build_sim_so` already wrote into
    `work_dir` (by its fixed name, `hexlib_sim.so`) under `hexagon-sim`, and
    parse its `SIMHOST` lines.

    FAIL CLOSED. Neither `hwinfo` nor `invoke` info is ever synthesised: a run
    that does not print a parseable line for each raises `DspSimError`. This
    mirrors `hexlib/sim.py`'s own rule for exactly the reason its docstring
    gives -- a farm job once ran zero tests and reported passing, and a
    zero-filled `SimHostResult` returned here on a parse miss would be that
    same failure in a new shape.
    """
    root = sdk_root or tc.default_sdk_root()
    so_path = os.path.join(work_dir, "hexlib_sim.so")
    cmd = rb.sim_qurt_command(work_dir, so_path, sdk_root=root, extra_args=tuple(extra_args))
    bin_dir = tc.find_toolchain_bin(root)
    env = tc.toolchain_env(bin_dir)

    # MUST run with cwd == work_dir -- see the module docstring's "ONE
    # NON-OBVIOUS BUILD DETAIL" and `_cwd`'s own docstring.
    with _cwd(work_dir):
        rc, out, err, timed_out = tc.run(cmd, env, timeout=tc.SIM_TIMEOUT_MAX_S)
    combined = out + err

    if timed_out:
        raise DspSimError(f"hexagon-sim timed out after {tc.SIM_TIMEOUT_MAX_S}s:\n{combined}")

    m_hw = _HWINFO_RE.search(combined)
    if not m_hw:
        raise DspSimError(
            "no SIMHOST hwinfo line recovered -- start did not succeed, or "
            "printed nothing parseable. Fail-closed: this never returns a "
            f"default result.\n{combined}"
        )
    m_inv = _INVOKE_RE.search(combined)
    if not m_inv:
        raise DspSimError(
            "no SIMHOST invoke line recovered -- nothing was computed. "
            f"Fail-closed: this never returns a default result.\n{combined}"
        )

    return SimHostResult(
        status=int(m_inv.group(3)),
        cycles=int(m_inv.group(5)),
        arch=int(m_hw.group(1)),
        vtcm=int(m_hw.group(3)),
        stdout=combined,
        exit_code=rc,
    )


def _align_up(n: int, align: int = HVX_ALIGN) -> int:
    return (n + align - 1) // align * align


def _ne(shape: Sequence[int]) -> tuple[int, int, int, int]:
    if len(shape) > 4:
        raise DspSimError(f"shape {tuple(shape)} has more than 4 dims; hexlib_tensor.ne is fixed at 4")
    padded = list(int(s) for s in shape) + [1] * (4 - len(shape))
    return tuple(padded)  # type: ignore[return-value]


def _out_shape(spec: RunnerSpec, arrays: tuple[np.ndarray, ...],
               attrs: Mapping[str, Any]) -> tuple[int, ...]:
    """Mirrors `hexlib.exec.hexagon._out_shape` (that module is off limits to
    modify or import a private helper from). Elementwise ops keep the first
    input's shape; a permutation reorders it; an explicit `shape` attr wins."""
    shape = tuple(arrays[0].shape)
    perm = attrs.get("perm")
    if perm is not None:
        return tuple(shape[i] for i in perm)
    declared = attrs.get("shape")
    if declared is not None:
        return tuple(declared)
    return shape


def _encode_params(spec: RunnerSpec, arrays: tuple[np.ndarray, ...],
                   attrs: Mapping[str, Any]) -> tuple[int, ...]:
    """Only the ATTR-sourced scalars go into `a->params`, bit for bit and in
    order -- `numel:`/`dim:` scalars are derived on the DSP from the tensor's
    own `ne[]` (see `genentry.py`'s `_scalar_expr`), never carried here. A
    float attr is packed as its raw IEEE-754 bit pattern reinterpreted as
    int32, because `hexlib_args.params` is `const void *` and the generated
    entry casts it to `(const float *)` before indexing -- packing the value
    as a plain `int` would send the wrong bits.
    """
    params: list[int] = []
    for sc in spec.scalars:
        if not sc.source.startswith("attr:"):
            continue
        value = sc.value(arrays, attrs)
        if sc.ctype == "int":
            params.append(int(value))
        else:
            params.append(struct.unpack("<i", struct.pack("<f", float(value)))[0])
    return tuple(params)


def _scale_layout(spec: RunnerSpec, n: int) -> tuple[int, int, int]:
    """(aligned end-of-input offset, output nbytes, total buffer size) for a
    single-input/single-output op of `n` elements -- the shape every kind in
    this backend's required tests actually drives (`scale`). Kept separate
    from the general `run()` layout logic so `build_batch`/`run_unmapped`, which
    build a batch WITHOUT any real array data, do not need one."""
    in_dtype = WIRE_DTYPE[spec.inputs[0]]
    out_dtype = WIRE_DTYPE[spec.out_dtype]
    nbytes_in = n * in_dtype.itemsize
    in_end = _align_up(nbytes_in)
    nbytes_out = n * out_dtype.itemsize
    return in_end, nbytes_out, in_end + nbytes_out


class DspSimBackend:
    """Drives a real kernel through the DSP skel's batch path, on the
    simulator. See the module docstring for exactly what that does and does
    not prove -- in short, our own code (batch parsing, the buffer table, the
    dispatch table, the kernel adapter, PCYCLE) and NOT qaic marshalling,
    which does not run on this path at all.

    Builds the skel archive, the QuRT-hosted `.so`, and the sim configs ONCE
    per instance (construction is the slow part); each `run`/`run_raw`/
    `run_unmapped`/`hwinfo` call is one `hexagon-sim` launch reusing them.
    """

    def __init__(self, kernels: list[str], work_dir: str, sdk_root: str | None = None):
        self.work_dir = work_dir
        self.sdk_root = sdk_root or tc.default_sdk_root()
        os.makedirs(work_dir, exist_ok=True)
        rb.build_skel_lib(kernels, work_dir, sdk_root=self.sdk_root)
        self.so_path = rb.build_sim_so(work_dir, sdk_root=self.sdk_root)
        rb.write_qurt_sim_configs(work_dir, sdk_root=self.sdk_root)

    def _clear_outputs(self) -> None:
        """Delete the previous call's artifacts before this one runs.

        Mirrors `hexlib/exec/hexagon.py`'s own removal of `hexlib_out.bin` and
        its reason: A STALE OUTPUT WOULD BE READ AS THIS CALL'S RESULT if the
        run failed to write one. `simhost.c` writes `hexlib_out.bin` only when
        the batch status is OK and `hexlib_rsp.bin` only once the invoke
        returned, so a launch that prints its `SIMHOST invoke ... status=1`
        line and then dies (or a `hexagon-sim` that never gets that far at all)
        leaves whatever the LAST call wrote sitting in the work directory, at
        the same names and -- for a same-shaped op -- at the same offsets.
        Deleting them here rather than merely not trusting them is deliberate:
        it means any future code path that reads these files inherits the
        guarantee instead of having to re-derive it.

        Every call shape goes through `_write_call`, so this runs for `run`,
        `run_unmapped`, `run_raw` and `hwinfo` alike.
        """
        for name in (OUT_NAME, RSP_NAME):
            path = os.path.join(self.work_dir, name)
            if os.path.exists(path):
                os.remove(path)

    def _write_call(self, blob: bytes, payload: bytes) -> None:
        self._clear_outputs()
        with open(os.path.join(self.work_dir, BATCH_NAME), "wb") as f:
            f.write(blob)
        with open(os.path.join(self.work_dir, IN_NAME), "wb") as f:
            f.write(payload)

    def _read_response(self) -> wire.BatchResponse:
        rsp_path = os.path.join(self.work_dir, RSP_NAME)
        if not os.path.isfile(rsp_path):
            raise DspSimError(f"sim produced no {RSP_NAME}; nothing was computed")
        with open(rsp_path, "rb") as f:
            raw = f.read()
        return wire.unpack_response(raw)

    # -- the acceptance path -------------------------------------------------

    def run(self, kind: str, arrays: Sequence[np.ndarray],
           attrs: Mapping[str, Any]) -> tuple[np.ndarray, RunnerStats]:
        """Invoke `kind` on real array data, through the skel's batch path.

        Builds one buffer holding every input then the output, each tensor's
        start 128-byte aligned (see the module docstring's "ALIGNMENT"), a
        single op naming them by index, and reads the result back out of
        `hexlib_out.bin` at the output tensor's own offset.

        WHAT IS REFUSED HERE, AND WHY IT CANNOT BE REFUSED ANYWHERE ELSE. See
        `spec.check_requires` below: an op kind is not always one kernel, and
        the DSP has no field to check the difference against. Everything this
        method refuses has the same failure mode if it is not refused -- a
        correctly-shaped, OK-status wrong answer -- which is why each check is
        an error and never a fallback.
        """
        spec = SPECS[kind]

        # AN OP KIND IS NOT ALWAYS ONE KERNEL, and the check has to be here.
        # `hexlib/exec/hexagon.py` has always done this, for the reason its own
        # comment gives: the failure mode otherwise is a correctly-shaped wrong
        # answer. This path skipped it, and the DSP cannot make up the
        # difference -- `hexlib_args` carries no field for a perm at all (see
        # genentry.py's `_requires_check`), and for the one key it DOES carry,
        # `dtype`, the generated check compares a value THIS serializer derived
        # from `spec.out_dtype` against the same spec's own requirement, so it
        # passes by construction whatever the caller asked for. Only a check
        # against the CALLER'S OWN attrs can fail, and this is it.
        spec.check_requires(attrs)

        # `zip` stops at the shorter sequence, so an op mis-wired with an extra
        # input array silently dropped it: `add` with three inputs computed
        # a+b, ignored c, and returned the right shape with no error.
        # `RunnerSpec.payload` raises on the other transport for exactly this.
        if len(arrays) != len(spec.inputs):
            raise DspSimError(
                f"{kind} takes {len(spec.inputs)} inputs, got {len(arrays)}; "
                "an extra array would be dropped and a missing one would leave "
                "the op naming a tensor that was never packed"
            )

        # A RAW input passes through untouched. `np.ascontiguousarray(rawtensor)`
        # would build a 0-d object array and every downstream `.nbytes`/`.shape`
        # would then describe a Python pointer rather than the weight. RawTensor
        # carries `.shape`, `.nbytes` and `.size`, so everything below that reads
        # those three works on either kind without knowing which it has.
        arrays = tuple(
            a if dt in WIRE_RAW
            else np.ascontiguousarray(a, dtype=WIRE_DTYPE[dt])
            for a, dt in zip(arrays, spec.inputs)
        )
        for i, (a, dt) in enumerate(zip(arrays, spec.inputs)):
            if dt in WIRE_RAW and not isinstance(a, RawTensor):
                raise DspSimError(
                    f"{kind} input {i} is declared {dt!r}, which is staged as "
                    f"raw quantized bytes, so it must be a RawTensor and not a "
                    f"{type(a).__name__} -- see hexlib/exec/runner.py's RawTensor"
                )
        out_shape = _out_shape(spec, arrays, attrs)
        out_dtype = WIRE_DTYPE[spec.out_dtype]
        out_nbytes = int(np.prod(out_shape)) * out_dtype.itemsize if out_shape else out_dtype.itemsize

        # FROM THE SPEC, NOT THE CONSTANT "row_major". Both tensor loops below
        # used to hard-code it, which meant the DSP's layout guard (added with
        # `genentry._layout_check`) could never see anything but row_major from
        # this serializer -- the same self-comparison the output dtype has, and
        # for the same structural reason. A spec declaring a `q4_0_repacked`
        # weight now actually says so on the wire.
        buf_layouts = spec.buf_layouts()

        payload = bytearray()
        tensors = []
        offset = 0
        for i, (a, dt) in enumerate(zip(arrays, spec.inputs)):
            # `nbytes` is the BYTE count and `ne` is the ELEMENT shape, and for a
            # q4_0 weight those are not related by an item size: (768, 768)
            # elements occupy 331776 bytes of 18-byte blocks. `wire.py` checks
            # only that the buffer holds `offset + nbytes`, never that
            # `nbytes == prod(ne) * itemsize`, so both fields stay true and a
            # kernel reading `a->ne` gets the shape it needs to index blocks with.
            nbytes = a.nbytes
            tensors.append(wire.TensorDesc(
                bi=0, offset=offset, nbytes=nbytes, dtype=dt,
                layout=buf_layouts[i], ne=_ne(tuple(a.shape)),
            ))
            payload += a.data if dt in WIRE_RAW else a.tobytes()
            offset += nbytes
            aligned = _align_up(offset)
            if aligned != offset:
                payload += b"\x00" * (aligned - offset)
                offset = aligned

        out_offset = offset
        # The output tensor's dtype describes the BYTES this buffer will hold,
        # so it comes from the spec, not from any attr -- the buffer was sized
        # from the same place two lines up. That is precisely why a
        # `("dtype", ...)` requirement cannot be validated on the DSP through
        # this serializer (it would be comparing the spec with itself), and why
        # `check_requires` above is the check that actually decides it.
        tensors.append(wire.TensorDesc(
            bi=0, offset=out_offset, nbytes=out_nbytes, dtype=spec.out_dtype,
            layout=buf_layouts[-1], ne=_ne(out_shape),
        ))
        payload += b"\x00" * out_nbytes

        bufs = [wire.BufDesc(fd=0, size=len(payload))]
        src = tuple(range(len(arrays)))
        dst = (len(arrays),)
        params = _encode_params(spec, arrays, attrs)
        kind_id = KIND_ID[kind]
        ops = [wire.OpDesc(kind=kind_id, params=params, src=src, dst=dst)]
        blob = wire.pack_batch(bufs, tensors, ops)

        self._write_call(blob, bytes(payload))
        res = run_sim(self.work_dir, sdk_root=self.sdk_root)
        if res.status != wire.STATUS["OK"]:
            name = wire.STATUS_NAME.get(res.status, res.status)
            raise DspSimError(f"{kind}: DSP invoke returned status {name} ({res.status})")
        # `simhost.c` returns 0 if and only if the batch status was
        # HEXLIB_DSP_OK, so an OK status line together with a nonzero exit means
        # the process died AFTER printing it -- before, for instance, writing
        # the output file. This was captured in the result and never read.
        if res.exit_code not in (0, None):
            raise DspSimError(
                f"{kind}: the batch reported OK but the simulator process "
                f"exited {res.exit_code}, so it did not finish. Anything it had "
                f"written by then is a partial result, not an answer.\n"
                f"{res.stdout}"
            )

        # WHAT CAME BACK MUST ANSWER WHAT WAS ASKED. `skel_dispatch.c` fills
        # `results[i].kind` from the op's own kind BEFORE the table lookup, so
        # this compares the id that reached the DSP with the id packed here. It
        # is the only host-side check that can see the host and the DSP
        # disagreeing about kind ids -- a drift that otherwise dispatches an op
        # to another kernel with a matching buffer count and reports OK. The
        # per-op status is a separate field from the batch header's status
        # (`wire.BatchResponse.ok` requires both) and was never read either.
        rsp = self._read_response()
        if rsp.n_ops != 1 or len(rsp.results) != 1:
            raise DspSimError(
                f"{kind}: the batch carried 1 op but the response reports "
                f"n_ops={rsp.n_ops} with {len(rsp.results)} results"
            )
        result = rsp.results[0]
        if result.kind != kind_id:
            raise DspSimError(
                f"{kind}: asked for kind {kind_id} but the response answers for "
                f"kind {result.kind}. The host and the DSP disagree about kind "
                f"ids, so this op was served by another kernel."
            )
        if not result.ok:
            name = wire.STATUS_NAME.get(result.status, result.status)
            raise DspSimError(
                f"{kind}: the batch status was OK but the op itself returned "
                f"{name} ({result.status})"
            )

        out_path = os.path.join(self.work_dir, OUT_NAME)
        if not os.path.isfile(out_path):
            raise DspSimError(
                f"{kind}: sim reported OK but wrote no {OUT_NAME}; nothing was computed"
            )
        with open(out_path, "rb") as f:
            raw = f.read()
        y_raw = raw[out_offset: out_offset + out_nbytes]
        y = np.frombuffer(y_raw, dtype=out_dtype).reshape(out_shape)

        return y, RunnerStats(calls=1, cycles=res.cycles)

    # -- the discriminator ----------------------------------------------------

    def run_unmapped(self, kind: str, n: int, factor: float) -> wire.BatchResponse:
        """Drive `simhost.c`'s `--unmapped` mode: the payload buffer's fd is
        patched into the batch as usual, but `hexlib_iface_mmap` is
        deliberately never called, so the skel must refuse with
        `HEXLIB_DSP_ERR_UNMAPPED` rather than resolve an address it was never
        given. See the module docstring's own section on why this is the one
        test that makes a simulator pass mean anything on silicon.
        """
        spec = SPECS[kind]
        _, _, total = _scale_layout(spec, n)
        blob = self.build_batch(kind, n, factor)
        self._write_call(blob, b"\x00" * total)
        run_sim(self.work_dir, extra_args=("--unmapped",), sdk_root=self.sdk_root)
        return self._read_response()

    # -- DSP-side validation, bypassing the host-side serializer ---------------

    def run_raw(self, blob: bytes) -> wire.BatchResponse:
        """Send `blob` verbatim as `hexlib_batch.bin`, bypassing
        `wire.pack_batch`'s own host-side checks entirely -- so a bad magic, a
        truncated blob or an unknown op kind exercises the DSP's OWN
        validation in `hexlib_dispatch_batch`, not the Python serializer's."""
        return self.run_raw_verbose(blob)[0]

    def run_raw_verbose(self, blob: bytes) -> tuple[wire.BatchResponse, str]:
        """`run_raw`, plus the simulator's own stdout.

        THE RESPONSE ALONE CANNOT DISTINGUISH SOME MALFORMED BLOBS from the
        host mishandling them, which is why this exists. simhost.c patches the
        real fd into the buffer table before invoking, and refuses to patch a
        table that does not lie inside the blob (see its comment). With
        `off_bufs = 0xFFFFFF00` and a 32-bit `size_t`, the unbounded loop
        computed `g_batch - 256`, corrupted whatever static preceded it, and
        invoked anyway -- and the skel returned the SAME ERR_TRUNCATED it
        returns when the table is correctly left alone. A test asserting only
        on `status` was green either way.

        The host's own `SIMHOST note=...` line is the only place that
        difference is observable, so a caller that needs to tell the two apart
        reads it here rather than inferring it from a status that does not
        carry it.
        """
        self._write_call(blob, b"")
        sim = run_sim(self.work_dir, sdk_root=self.sdk_root)
        return self._read_response(), sim.stdout

    def build_batch(self, kind: str, n: int, factor: float,
                    kind_override: int | None = None) -> bytes:
        """A scale-shaped batch: one input tensor of `n` elements, one
        same-length output tensor, both slices of a single buffer (`fd=0`,
        the placeholder `simhost.c` patches to the real fd). `kind_override`
        lets a caller build an otherwise-well-formed batch naming an
        unregistered op kind, to drive `HEXLIB_DSP_ERR_NO_KERNEL` without
        touching `wire.pack_batch`'s own validation (which does not check
        kind against the dispatch table at all -- that check is the DSP's).
        """
        spec = SPECS[kind]
        in_dtype = WIRE_DTYPE[spec.inputs[0]]
        in_end, nbytes_out, total = _scale_layout(spec, n)
        nbytes_in = n * in_dtype.itemsize

        bufs = [wire.BufDesc(fd=0, size=total)]
        tensors = [
            wire.TensorDesc(bi=0, offset=0, nbytes=nbytes_in, dtype=spec.inputs[0],
                            layout="row_major", ne=(n, 1, 1, 1)),
            wire.TensorDesc(bi=0, offset=in_end, nbytes=nbytes_out, dtype=spec.out_dtype,
                            layout="row_major", ne=(n, 1, 1, 1)),
        ]
        factor_bits = struct.unpack("<i", struct.pack("<f", factor))[0]
        kind_id = KIND_ID[kind] if kind_override is None else int(kind_override)
        ops = [wire.OpDesc(kind=kind_id, params=(factor_bits,), src=(0,), dst=(1,))]
        return wire.pack_batch(bufs, tensors, ops)

    # -- what the DSP says about itself ---------------------------------------

    def hwinfo(self) -> SimHostResult:
        """An empty-but-well-formed batch (0 bufs/tensors/ops) is enough to
        drive open -> start -> hwinfo -> invoke -> stop -> close, with no
        rpcmem/fd plumbing needed -- the same shape
        `test_sim_run_reaches_start_and_reports_real_vtcm` in
        `test_runtime_sim_build.py` already proved reaches a successful
        `start()`."""
        self._write_call(wire.pack_batch([], [], []), b"")
        return run_sim(self.work_dir, sdk_root=self.sdk_root)

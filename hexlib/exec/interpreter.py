"""Replay a `Compiled` on the host, through the plan's own VTCM allocation.

WHAT THIS IS FOR. Until now the M1 pass pipeline was validated structurally:
tests assert that offsets do not overlap, that transfers are well formed, that
the high water is the maximum slot end. Every one of those assertions is
checked by code from the same pass that produced the thing being checked. This
executes the plan instead, so a bad allocation shows up as a wrong number.

It also needs no FastRPC, no DSP skel, and no device. That matters because the
silicon-path runtime is what blocks the on-DSP executor, and it does not block
this.

WHAT IT IS NOT. It is not fast and never will be. Like the eager oracle it
exists to be obviously correct.

PER-OP BACKENDS. `backends` maps an op kind to something that computes it. The
default is the op registry's own reference, so the encoder runs end to end from
the first day and each real kernel replaces exactly one entry. A kind with no
entry is an error, never a silently skipped step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from hexlib.exec.vtcm import COMPUTE_DTYPE, VtcmError, VtcmImage, overlapping_slots
from hexlib.graph.compiled import Compiled
from hexlib.graph.ir import Tensor
from hexlib.graph.ops import REGISTRY, Registry
from hexlib.graph.plan import Step, Transfer
from hexlib.result import Err

Backend = Callable[[tuple[np.ndarray, ...], Mapping[str, Any]], tuple[np.ndarray, ...]]


@dataclass
class ExecReport:
    """What the run did, in the same units the plan predicted it in.

    The plan claims a byte count and a high water. Both are recomputed here from
    what actually happened, so a prediction that does not match execution is a
    visible disagreement rather than an unchecked number.
    """

    outputs: dict[str, np.ndarray]
    bytes_in: int = 0
    bytes_out: int = 0
    steps_run: int = 0
    ops_run: int = 0
    dma_transfers: int = 0
    # Weights the plan streams in chunks rather than making resident. These have
    # a Tiling and deliberately NO Slot: only `buffers * chunk` bytes are ever in
    # VTCM at once, which is why the high water is chunk-sized and not
    # layer-sized. A host interpreter has nothing to overlap a prefetch with, so
    # it reads the whole tensor from host memory -- correct for the values, and
    # NOT evidence of a missing transfer.
    streamed_weights: set[str] = field(default_factory=set)
    # Ops whose input came from host memory with neither a slot nor a tiling to
    # explain it. This IS a plan gap: the plan is supposed to account for
    # everything a kernel reads.
    unstaged_reads: list[str] = field(default_factory=list)
    backend_used: dict[str, str] = field(default_factory=dict)

    @property
    def bytes_moved(self) -> int:
        return self.bytes_in + self.bytes_out


def _registry_backend(kind: str, registry: Registry) -> Backend:
    opdef = registry.get(kind)

    def run(arrays, attrs):
        return opdef.reference(arrays, attrs)

    return run


def run(
    compiled: Compiled,
    feeds: Mapping[str, np.ndarray],
    backends: Mapping[str, Backend] | None = None,
    registry: Registry = REGISTRY,
    check_allocation: bool = True,
) -> ExecReport | Err:
    """Execute `compiled` on `feeds`, returning the declared outputs or an Err."""
    graph, plan = compiled.graph, compiled.plan

    problems = graph.problems()
    if problems:
        return Err("graph is structurally invalid", "\n".join(problems))

    if check_allocation:
        # Independent of the allocator's own invariant check, deliberately.
        overlaps = overlapping_slots(plan.vtcm)
        if overlaps:
            return Err(
                "VTCM allocation aliases two simultaneously-live tensors",
                "\n".join(overlaps[:20]),
            )

    try:
        image = VtcmImage(plan.vtcm_budget, plan.vtcm)
    except VtcmError as e:
        return Err("VTCM image could not be built from the plan", str(e))

    # Host memory. Consts and graph inputs start here; a `dma_out` puts results
    # back here. Stored in the compute dtype -- the rounding to a stored format
    # happens at the VTCM boundary, which is where hardware does it too.
    ddr: dict[str, np.ndarray] = {}
    required = list(graph.inputs) + [t.name for t in graph.tensors.values() if t.const]
    for name in required:
        if name not in feeds:
            return Err(
                "missing feed",
                f"tensor {name!r} is a graph input or a const and was not "
                f"supplied; supplied: {sorted(feeds)[:12]}"
                + (" ..." if len(feeds) > 12 else ""),
            )
        spec = graph.tensor(name)
        try:
            array = np.asarray(feeds[name])
        except (ValueError, TypeError) as e:
            return Err("feed conversion failed", f"{name!r}: {type(e).__name__}: {e}")
        if tuple(array.shape) != spec.shape:
            return Err(
                "feed shape mismatch",
                f"tensor {name!r} is declared {spec.shape} but the feed has "
                f"shape {tuple(array.shape)}",
            )
        ddr[name] = array.astype(COMPUTE_DTYPE, copy=False)

    resolved: dict[str, Backend] = {}
    for op in graph.ops:
        if op.kind in resolved:
            continue
        if backends and op.kind in backends:
            resolved[op.kind] = backends[op.kind]
        else:
            try:
                resolved[op.kind] = _registry_backend(op.kind, registry)
            except KeyError as e:
                return Err("unregistered op kind", f"op {op.id}: {e.args[0]}")

    report = ExecReport(outputs={})
    report.backend_used = {
        kind: ("supplied" if backends and kind in backends else "reference")
        for kind in resolved
    }

    issued: set[int] = set()

    for index, step in enumerate(plan.steps):
        # Issue, then wait, then compute -- the order the hardware does it in.
        # A step's dma_wait names ids from its OWN dma_in and its own tiling
        # (dma.py draws both from one counter), so both must be issued before
        # the wait is checked.
        err = _do_transfers(step.dma_in, "in", graph, ddr, image, report, issued, index)
        if err is not None:
            return err

        if step.tiling is not None:
            err = _do_tiling(step, index, graph, ddr, image, report, issued)
            if err is not None:
                return err

        missing_waits = [i for i in step.dma_wait if i not in issued]
        if missing_waits:
            return Err(
                "plan waits on transfers that were never issued",
                f"step {index}: dma_wait {missing_waits} not among the "
                f"{len(issued)} transfers issued so far",
            )

        if step.op is not None:
            err = _run_op(step, index, graph, ddr, image, resolved, report)
            if err is not None:
                return err
            report.ops_run += 1

        err = _do_transfers(
            step.dma_out, "out", graph, ddr, image, report, issued, index
        )
        if err is not None:
            return err

        report.steps_run += 1

    for name in graph.outputs:
        spec = graph.tensor(name)
        if name in ddr:
            report.outputs[name] = ddr[name]
        elif name in image:
            try:
                report.outputs[name] = image.get(spec)
            except VtcmError as e:
                return Err(f"graph output {name!r} could not be read", str(e))
        else:
            return Err(
                "graph output was never materialized",
                f"{name!r} is neither in host memory nor in a VTCM slot after "
                "the whole plan ran",
            )
    return report


def _do_transfers(
    transfers: tuple[Transfer, ...],
    direction: str,
    graph,
    ddr: dict[str, np.ndarray],
    image: VtcmImage,
    report: ExecReport,
    issued: set[int],
    index: int,
) -> Err | None:
    for t in transfers:
        if t.direction != direction:
            return Err(
                "transfer is in the wrong list",
                f"step {index}: transfer {t.id} for {t.tensor!r} has direction "
                f"{t.direction!r} but appears in dma_{direction}",
            )
        if t.tensor not in graph.tensors:
            return Err(
                "transfer names a tensor the graph does not declare",
                f"step {index}: transfer {t.id} moves {t.tensor!r}",
            )
        spec = graph.tensor(t.tensor)
        issued.add(t.id)
        report.dma_transfers += 1

        # The transfer's own offset must agree with the tensor's slot. Two
        # sources for one address is exactly how a plan drifts, and it is
        # cheaper to catch here than as a wrong activation later.
        if t.tensor in image:
            slot = image.slot(t.tensor)
            if t.vtcm_offset != slot.offset:
                return Err(
                    "transfer and slot disagree about an address",
                    f"step {index}: transfer {t.id} writes {t.tensor!r} at "
                    f"{t.vtcm_offset} but its slot is at {slot.offset}",
                )

        if direction == "in":
            if t.tensor not in ddr:
                return Err(
                    "transfer stages a tensor that is not in host memory",
                    f"step {index}: {t.tensor!r} has not been produced or fed",
                )
            try:
                image.put(spec, ddr[t.tensor])
            except VtcmError as e:
                return Err(f"staging {t.tensor!r} into VTCM failed", str(e))
            report.bytes_in += t.nbytes
        else:
            try:
                ddr[t.tensor] = image.get(spec)
            except VtcmError as e:
                return Err(f"writing back {t.tensor!r} from VTCM failed", str(e))
            report.bytes_out += t.nbytes
    return None


def _do_tiling(
    step: Step,
    index: int,
    graph,
    ddr: dict[str, np.ndarray],
    image: VtcmImage,
    report: ExecReport,
    issued: set[int],
) -> Err | None:
    """Honor a tiled transfer, and check its arithmetic against the tensor.

    A tiling claims that `transfer` repeats `count` times through `buffers`
    rotating slots. The host moves the bytes in one go -- there is nothing to
    overlap with on a host -- but `count * nbytes` still has to account for the
    tensor, and that is a statement about the plan worth checking.
    """
    tiling = step.tiling
    t = tiling.transfer
    if t.tensor not in graph.tensors:
        return Err(
            "tiled transfer names a tensor the graph does not declare",
            f"step {index}: {t.tensor!r}",
        )
    spec = graph.tensor(t.tensor)
    issued.add(t.id)
    report.streamed_weights.add(t.tensor)
    total = tiling.count * t.nbytes
    if total < spec.nbytes:
        return Err(
            "a tiled transfer does not move the whole tensor",
            f"step {index}: {t.tensor!r} is {spec.nbytes} bytes but "
            f"{tiling.count} chunks of {t.nbytes} move only {total}",
        )
    if t.direction == "in":
        if t.tensor not in ddr:
            return Err(
                "tiled transfer stages a tensor that is not in host memory",
                f"step {index}: {t.tensor!r}",
            )
        if t.tensor in image:
            try:
                image.put(spec, ddr[t.tensor])
            except VtcmError as e:
                return Err(f"staging tiled {t.tensor!r} failed", str(e))
        report.bytes_in += total
    else:
        report.bytes_out += total
    report.dma_transfers += tiling.count
    return None


def _read_input(
    name: str,
    graph,
    ddr: dict[str, np.ndarray],
    image: VtcmImage,
    report: ExecReport,
) -> np.ndarray:
    spec: Tensor = graph.tensor(name)
    if name in image:
        try:
            return image.get(spec)
        except VtcmError:
            # Slotted but never written -- fall back to host memory if it is
            # there, and record that the plan did not stage it.
            if name in ddr:
                report.unstaged_reads.append(name)
                return ddr[name]
            raise
    if name in ddr:
        # A chunk-streamed weight has no slot BY DESIGN, so reading it from host
        # memory is the plan working as specified, not a gap.
        if name not in report.streamed_weights:
            report.unstaged_reads.append(name)
        return ddr[name]
    raise VtcmError(f"tensor {name!r} is in neither VTCM nor host memory")


def _run_op(
    step: Step,
    index: int,
    graph,
    ddr: dict[str, np.ndarray],
    image: VtcmImage,
    backends: dict[str, Backend],
    report: ExecReport,
) -> Err | None:
    op = step.op
    try:
        arrays = tuple(
            _read_input(name, graph, ddr, image, report) for name in op.inputs
        )
    except VtcmError as e:
        return Err(f"step {index}: op {op.id} ({op.kind}) could not read an input", str(e))

    try:
        results = backends[op.kind](arrays, op.attrs)
    except Exception as e:  # noqa: BLE001 -- a failing backend is a reportable Err
        return Err(
            "op backend raised",
            f"step {index}: op {op.id} ({op.kind}): {type(e).__name__}: {e}",
        )

    if not isinstance(results, tuple):
        return Err(
            "op backend returned a non-tuple",
            f"step {index}: op {op.id} ({op.kind}) returned "
            f"{type(results).__name__}",
        )
    if len(results) != len(op.outputs):
        return Err(
            "op output count mismatch",
            f"step {index}: op {op.id} ({op.kind}) declares {len(op.outputs)} "
            f"outputs but the backend returned {len(results)}",
        )

    for name, value in zip(op.outputs, results):
        spec = graph.tensor(name)
        value = np.asarray(value)
        if tuple(value.shape) != spec.shape:
            return Err(
                "op result shape mismatch",
                f"step {index}: op {op.id} ({op.kind}) output {name!r} is "
                f"declared {spec.shape} but the backend produced "
                f"{tuple(value.shape)}",
            )
        if name in image:
            try:
                image.put(spec, value)
            except VtcmError as e:
                return Err(f"storing {name!r} into VTCM failed", str(e))
        else:
            ddr[name] = value.astype(COMPUTE_DTYPE, copy=False)
    return None

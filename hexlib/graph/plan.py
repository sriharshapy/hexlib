"""The plan is a value: serializable, inspectable, diffable, committable.

NPU performance work is normally opaque. A plan you can print, diff and commit
is how the next op worth attacking gets identified -- and it is what M2
compares its measured cycles against, so drift between the host model and the
hardware becomes visible rather than assumed.

Fail-closed the same way hexlib v1's Measurements is: a Plan cannot exist
without steps, without an integer high-water mark, or without an integer
predicted cost.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any

from hexlib.graph.ir import Op
from hexlib.result import Err

V75_VTCM_TOTAL_BYTES = 8388608
"""The v75 part total, 8388608 bytes at 0xd9000000.

This is a DEFAULT FOR THE CLI, not a budget. VTCM is acquired at session start
(ggml-hexagon's htp_iface_start takes a max_vmem argument), so a process may
get less. Anything planning for real hardware must pass the runtime's number.
"""


@dataclass(frozen=True)
class Slot:
    tensor: str
    offset: int
    size: int
    first_use: int
    last_use: int

    def __post_init__(self) -> None:
        if self.offset < 0 or self.size <= 0:
            raise ValueError(
                f"slot {self.tensor!r}: offset {self.offset} and size {self.size} "
                "must be non-negative and positive"
            )
        if self.last_use < self.first_use:
            raise ValueError(
                f"slot {self.tensor!r}: last_use {self.last_use} precedes first_use "
                f"{self.first_use}"
            )

    @property
    def end(self) -> int:
        return self.offset + self.size


@dataclass(frozen=True)
class Transfer:
    id: int
    tensor: str
    direction: str
    vtcm_offset: int
    nbytes: int

    def __post_init__(self) -> None:
        if self.direction not in ("in", "out"):
            raise ValueError(
                f"transfer {self.id}: direction must be 'in' or 'out', got "
                f"{self.direction!r}"
            )
        if self.nbytes <= 0:
            raise ValueError(f"transfer {self.id}: nbytes must be positive")
        if self.vtcm_offset < 0:
            raise ValueError(f"transfer {self.id}: vtcm_offset must be non-negative")


@dataclass(frozen=True)
class Tiling:
    """The step's dma_in, op and dma_out repeat `count` times.

    `vtcm_offset` rotates through `buffers` slots, so buffers=2 is the
    double-buffering that lets a DMA overlap the compute it feeds.
    """

    axis: str
    tile_elements: int
    count: int
    buffers: int

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("tiling count must be at least 1")
        if self.buffers < 1:
            raise ValueError("tiling buffers must be at least 1")
        if self.tile_elements < 1:
            raise ValueError("tiling tile_elements must be at least 1")


@dataclass(frozen=True)
class Step:
    op: Op | None
    dma_in: tuple[Transfer, ...]
    dma_wait: tuple[int, ...]
    dma_out: tuple[Transfer, ...]
    tiling: Tiling | None = None


@dataclass(frozen=True)
class Plan:
    steps: tuple[Step, ...]
    vtcm: tuple[Slot, ...]
    vtcm_high_water: int
    predicted_bytes_moved: int
    vtcm_budget: int
    unimplemented: tuple[str, ...]
    target: str = "hexagon-v75"

    def __post_init__(self) -> None:
        if not isinstance(self.target, str) or not self.target:
            raise ValueError(
                "a Plan must name the target it was compiled for; two plans for one "
                "graph are not comparable otherwise"
            )
        if not self.steps:
            raise ValueError(
                "a Plan requires at least one step; a plan that scheduled nothing "
                "is not a plan"
            )
        for field in ("vtcm_high_water", "predicted_bytes_moved", "vtcm_budget"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{field} must be an int, got {type(value).__name__}. A plan whose "
                    "predicted cost cannot be computed is an error, not a plan with a "
                    "zero cost field."
                )
        if self.vtcm_high_water > self.vtcm_budget:
            raise ValueError(
                f"vtcm_high_water {self.vtcm_high_water} exceeds the budget "
                f"{self.vtcm_budget}; an allocation that does not fit is an error, "
                "never a partial plan"
            )


def to_json(plan: Plan) -> str:
    return json.dumps(_plan_to_dict(plan), indent=2, sort_keys=True)


def _plan_to_dict(plan: Plan) -> dict[str, Any]:
    return {
        "steps": [_step_to_dict(s) for s in plan.steps],
        "vtcm": [asdict(s) for s in plan.vtcm],
        "vtcm_high_water": plan.vtcm_high_water,
        "predicted_bytes_moved": plan.predicted_bytes_moved,
        "vtcm_budget": plan.vtcm_budget,
        "unimplemented": list(plan.unimplemented),
        "target": plan.target,
    }


def _step_to_dict(step: Step) -> dict[str, Any]:
    op = None
    if step.op is not None:
        op = {
            "id": step.op.id,
            "kind": step.op.kind,
            "inputs": list(step.op.inputs),
            "outputs": list(step.op.outputs),
            "attrs": dict(step.op.attrs),
        }
    return {
        "op": op,
        "dma_in": [asdict(t) for t in step.dma_in],
        "dma_wait": list(step.dma_wait),
        "dma_out": [asdict(t) for t in step.dma_out],
        "tiling": asdict(step.tiling) if step.tiling else None,
    }


_REQUIRED = (
    "steps",
    "vtcm",
    "vtcm_high_water",
    "predicted_bytes_moved",
    "vtcm_budget",
    "unimplemented",
    "target",
)


def from_json(text: str) -> Plan | Err:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        return Err("plan is not valid JSON", str(e))
    if not isinstance(raw, dict):
        return Err("plan is not a JSON object", f"got {type(raw).__name__}")
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        return Err("plan is missing required fields", ", ".join(missing))
    try:
        return Plan(
            steps=tuple(_step_from_dict(s) for s in raw["steps"]),
            vtcm=tuple(Slot(**s) for s in raw["vtcm"]),
            vtcm_high_water=raw["vtcm_high_water"],
            predicted_bytes_moved=raw["predicted_bytes_moved"],
            vtcm_budget=raw["vtcm_budget"],
            unimplemented=tuple(raw["unimplemented"]),
            target=raw["target"],
        )
    except (TypeError, ValueError) as e:
        return Err("plan failed validation", str(e))


def _step_from_dict(raw: dict[str, Any]) -> Step:
    op = None
    if raw.get("op") is not None:
        o = raw["op"]
        op = Op(
            id=o["id"],
            kind=o["kind"],
            inputs=tuple(o["inputs"]),
            outputs=tuple(o["outputs"]),
            attrs=dict(o["attrs"]),
        )
    return Step(
        op=op,
        dma_in=tuple(Transfer(**t) for t in raw["dma_in"]),
        dma_wait=tuple(raw["dma_wait"]),
        dma_out=tuple(Transfer(**t) for t in raw["dma_out"]),
        tiling=Tiling(**raw["tiling"]) if raw.get("tiling") else None,
    )


def render(plan: Plan) -> str:
    """A human-readable plan: the VTCM map, the DMA schedule, the fusion result."""
    lines: list[str] = []
    lines.append(f"target      {plan.target}")
    lines.append("")
    lines.append("VTCM")
    lines.append(f"  budget      {plan.vtcm_budget:,} bytes")
    lines.append(
        f"  high water  {plan.vtcm_high_water:,} bytes "
        f"({100.0 * plan.vtcm_high_water / plan.vtcm_budget:.1f}%)"
    )
    lines.append(f"  slots       {len(plan.vtcm)}")
    lines.append("")
    lines.append("Predicted traffic")
    lines.append(f"  DDR<->VTCM  {plan.predicted_bytes_moved:,} bytes")
    lines.append("")

    if plan.unimplemented:
        lines.append("Op kinds with no kernel — NOT IMPLEMENTED")
        for kind in plan.unimplemented:
            lines.append(f"  {kind}")
        lines.append("  This plan schedules them but nothing can execute it yet.")
    else:
        lines.append("All op kinds have kernels.")
    lines.append("")

    lines.append(f"Steps ({len(plan.steps)})")
    for i, step in enumerate(plan.steps):
        kind = step.op.kind if step.op else "(dma only)"
        name = step.op.outputs[0] if step.op and step.op.outputs else ""
        tile = ""
        if step.tiling:
            tile = (
                f"  x{step.tiling.count} tiles of {step.tiling.tile_elements} "
                f"on {step.tiling.axis}, {step.tiling.buffers} buffers"
            )
        moved = sum(t.nbytes for t in step.dma_in) + sum(t.nbytes for t in step.dma_out)
        per_iter = f"  {moved:,} B/iter" if moved else ""
        lines.append(f"  {i:4d}  {kind:<18} {name:<24}{per_iter}{tile}")
    return "\n".join(lines)

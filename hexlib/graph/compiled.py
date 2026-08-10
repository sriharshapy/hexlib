"""A `Plan` plus the graph it was compiled from — the unit that can be executed.

WHY THIS TYPE EXISTS. `plan.py` describes a `Plan` as "serializable,
inspectable, diffable, committable", and the project's own notes say M2
"replays a Plan". Both were half true. A `Plan`'s steps carry tensor NAMES
(`plan._step_to_dict`) and nothing else: no shape, no dtype. Fusion also
CREATES ops -- 75 `matmul_epilogue`s at 256x256 -- that exist in no graph the
caller ever saw, because `compile_graph` returned the plan and dropped the
post-fusion graph on the floor. So a committed `plan.json` could be printed and
diffed but could not be replayed by anything, and the gap was invisible because
every test held the graph in a local variable already.

A `Compiled` is the pair, and it round-trips. That makes "committable" true in
the sense the docstring claimed, and it is what an executor -- host or DSP --
needs in order to know that `blk.3.attn.qkv.out` is 256x768 fp16.

The graph carried here is the POST-FUSION, POST-ORDERING graph, not the one the
caller passed in. Its `ops` are in execution order and correspond 1:1 with the
plan's steps that have an op.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from hexlib.graph.ir import Graph, Op, Tensor
from hexlib.graph.plan import Plan
from hexlib.graph.plan import _plan_to_dict, _step_from_dict
from hexlib.result import Err


@dataclass(frozen=True)
class Compiled:
    graph: Graph
    plan: Plan

    def __post_init__(self) -> None:
        if not isinstance(self.graph, Graph):
            raise TypeError(
                f"Compiled.graph must be a Graph, got {type(self.graph).__name__}"
            )
        if not isinstance(self.plan, Plan):
            raise TypeError(
                f"Compiled.plan must be a Plan, got {type(self.plan).__name__}"
            )
        # The pairing is the whole point of the type, so a mismatched pair is
        # rejected at construction rather than at the first missing lookup
        # somewhere inside an executor. Every tensor a step names must be
        # declared, or the pair cannot be replayed.
        missing: list[str] = []
        for step in self.plan.steps:
            if step.op is None:
                continue
            for name in tuple(step.op.inputs) + tuple(step.op.outputs):
                if name not in self.graph.tensors:
                    missing.append(f"op {step.op.id} ({step.op.kind}): {name}")
        if missing:
            raise ValueError(
                "plan references tensors the graph does not declare, so this "
                "pair cannot be executed:\n  " + "\n  ".join(missing[:10])
                + (f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else "")
            )

    def tensor(self, name: str) -> Tensor:
        return self.graph.tensor(name)


def _graph_to_dict(graph: Graph) -> dict[str, Any]:
    return {
        "tensors": [
            {"name": t.name, "dtype": t.dtype, "shape": list(t.shape), "const": t.const}
            for t in graph.tensors.values()
        ],
        "ops": [
            {
                "id": op.id,
                "kind": op.kind,
                "inputs": list(op.inputs),
                "outputs": list(op.outputs),
                "attrs": dict(op.attrs),
            }
            for op in graph.ops
        ],
        "inputs": list(graph.inputs),
        "outputs": list(graph.outputs),
    }


def to_json(compiled: Compiled) -> str:
    return json.dumps(
        {
            "graph": _graph_to_dict(compiled.graph),
            "plan": _plan_to_dict(compiled.plan),
        },
        indent=2,
        sort_keys=True,
    )


def _lists_to_tuples(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _lists_to_tuples(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return tuple(_lists_to_tuples(v) for v in obj)
    return obj


def _graph_from_dict(raw: dict[str, Any]) -> Graph:
    tensors = {}
    for t in raw["tensors"]:
        tensors[t["name"]] = Tensor(
            name=t["name"],
            dtype=t["dtype"],
            shape=tuple(t["shape"]),
            const=bool(t.get("const", False)),
        )
    ops = tuple(
        Op(
            id=o["id"],
            kind=o["kind"],
            inputs=tuple(o["inputs"]),
            outputs=tuple(o["outputs"]),
            attrs=_lists_to_tuples(o.get("attrs", {})),
        )
        for o in raw["ops"]
    )
    return Graph(
        tensors=tensors,
        ops=ops,
        inputs=tuple(raw["inputs"]),
        outputs=tuple(raw["outputs"]),
    )


_REQUIRED = ("graph", "plan")


def from_json(text: str) -> Compiled | Err:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        return Err("compiled artifact is not valid JSON", str(e))
    if not isinstance(raw, dict):
        return Err(
            "compiled artifact is not a JSON object", f"got {type(raw).__name__}"
        )
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        return Err(
            "compiled artifact is missing required fields", ", ".join(missing)
        )
    try:
        plan_raw = raw["plan"]
        plan = Plan(
            steps=tuple(_step_from_dict(s) for s in plan_raw["steps"]),
            vtcm=tuple(_slot(s) for s in plan_raw["vtcm"]),
            vtcm_high_water=plan_raw["vtcm_high_water"],
            predicted_bytes_moved=plan_raw["predicted_bytes_moved"],
            vtcm_budget=plan_raw["vtcm_budget"],
            unimplemented=tuple(plan_raw["unimplemented"]),
            target=plan_raw["target"],
        )
        return Compiled(graph=_graph_from_dict(raw["graph"]), plan=plan)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        return Err(
            "compiled artifact failed validation", f"{type(e).__name__}: {e}"
        )


def _slot(raw: dict[str, Any]):
    from hexlib.graph.plan import Slot

    return Slot(**raw)

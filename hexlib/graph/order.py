"""Pass: choose an execution order.

The encoder is close to a chain -- no window attention, no deepstack side
outputs -- so the scheduler's only real freedom is the q/k/v triple and the
transposes around it. That is small, but it is exactly where peak liveness is
decided, so the policy is swappable and the two are baked off.
"""
from __future__ import annotations

from typing import Callable, Mapping

from hexlib.graph.ir import Graph, Op
from hexlib.result import Err


def _ready_asap(ready: list[Op], live_bytes: Mapping[str, int], graph: Graph) -> Op:
    """First in declaration order. The obvious baseline."""
    return ready[0]


def _ready_min_peak(ready: list[Op], live_bytes: Mapping[str, int], graph: Graph) -> Op:
    """Prefer the op that frees the most bytes net of what it allocates."""

    def net_freed(op: Op) -> int:
        allocated = sum(graph.tensor(n).nbytes for n in op.outputs)
        freed = sum(
            live_bytes[n] for n in op.inputs if live_bytes.get(n, 0) and _last_use(graph, n, op)
        )
        return freed - allocated

    return max(ready, key=lambda op: (net_freed(op), -op.id))


def _last_use(graph: Graph, name: str, op: Op) -> bool:
    later = [o for o in graph.ops if o.id > op.id and name in o.inputs]
    return not later and name not in graph.outputs


ORDER_POLICIES: Mapping[str, Callable[[list[Op], Mapping[str, int], Graph], Op]] = {
    "asap": _ready_asap,
    "min_peak": _ready_min_peak,
}


def order(graph: Graph, policy: str = "min_peak") -> Graph | Err:
    choose = ORDER_POLICIES.get(policy)
    if choose is None:
        return Err(
            "unknown ordering policy",
            f"policy {policy!r} is not registered. Known: "
            f"{', '.join(sorted(ORDER_POLICIES))}",
        )

    available = set(graph.inputs) | {t.name for t in graph.tensors.values() if t.const}
    live_bytes = {name: graph.tensor(name).nbytes for name in available}
    remaining = list(graph.ops)
    scheduled: list[Op] = []

    while remaining:
        ready = [op for op in remaining if all(n in available for n in op.inputs)]
        if not ready:
            stuck = ", ".join(f"{op.id} ({op.kind})" for op in remaining[:5])
            return Err(
                "cycle or unreachable ops in graph",
                f"no op is ready but {len(remaining)} remain; first stuck: {stuck}",
            )
        chosen = choose(ready, live_bytes, graph)
        remaining.remove(chosen)
        scheduled.append(chosen)
        for name in chosen.outputs:
            available.add(name)
            live_bytes[name] = graph.tensor(name).nbytes

    return Graph(
        tensors=graph.tensors,
        ops=tuple(scheduled),
        inputs=graph.inputs,
        outputs=graph.outputs,
    )


def peak_live_bytes(graph: Graph) -> int:
    """Peak simultaneously-live activation bytes under this graph's op order.

    Consts are excluded: they are DDR-resident and streamed, never counted
    against the working set (spec 5.2).
    """
    last_use: dict[str, int] = {}
    for i, op in enumerate(graph.ops):
        for name in op.inputs:
            last_use[name] = i
    for name in graph.outputs:
        last_use[name] = len(graph.ops)

    live: set[str] = {n for n in graph.inputs if not graph.tensor(n).const}
    peak = sum(graph.tensor(n).nbytes for n in live)
    for i, op in enumerate(graph.ops):
        for name in op.outputs:
            if not graph.tensor(name).const:
                live.add(name)
        peak = max(peak, sum(graph.tensor(n).nbytes for n in live))
        for name in set(op.inputs) | set(op.outputs):
            if last_use.get(name, -1) <= i and name in live:
                live.discard(name)
    return peak

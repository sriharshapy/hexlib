"""Pass: live intervals over the ordered ops.

Indices are positions in `graph.ops`, so this runs AFTER `order`. Const
tensors are excluded: they are DDR-resident and streamed in chunks, never
allocated a VTCM interval (spec 5.2 -- "VTCM is a working buffer, not a layer
cache").
"""
from __future__ import annotations

from dataclasses import dataclass

from hexlib.graph.ir import Graph
from hexlib.result import Err


@dataclass(frozen=True)
class Interval:
    tensor: str
    first_use: int
    last_use: int
    nbytes: int

    def overlaps(self, other: "Interval") -> bool:
        return self.first_use < other.last_use and other.first_use < self.last_use

    def covers(self, step: int) -> bool:
        return self.first_use <= step <= self.last_use


def live_intervals(graph: Graph) -> tuple[Interval, ...] | Err:
    problems = graph.problems()
    if problems:
        return Err("graph is structurally invalid", "\n".join(problems))

    first: dict[str, int] = {}
    last: dict[str, int] = {}

    for name in graph.inputs:
        if not graph.tensor(name).const:
            first[name] = 0
            last[name] = 0

    for i, op in enumerate(graph.ops):
        for name in op.outputs:
            if graph.tensor(name).const:
                continue
            first.setdefault(name, i)
            # A dead store still occupies memory when it is written; give it a
            # zero-length interval rather than dropping it, which would
            # understate the high-water mark.
            last[name] = max(last.get(name, i), i)
        for name in op.inputs:
            if graph.tensor(name).const:
                continue
            first.setdefault(name, i)
            last[name] = max(last.get(name, i), i)

    for name in graph.outputs:
        if not graph.tensor(name).const:
            last[name] = len(graph.ops)

    intervals = tuple(
        sorted(
            (
                Interval(
                    tensor=name,
                    first_use=first[name],
                    last_use=last[name],
                    nbytes=graph.tensor(name).nbytes,
                )
                for name in first
            ),
            key=lambda iv: (iv.first_use, iv.tensor),
        )
    )
    return intervals


def live_at(intervals: tuple[Interval, ...], step: int) -> tuple[Interval, ...]:
    return tuple(iv for iv in intervals if iv.covers(step))

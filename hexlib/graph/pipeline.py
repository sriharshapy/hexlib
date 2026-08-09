"""Composes the passes. Plain functions in a list -- no framework.

Each stage's failure is returned, named, with the pass that produced it, so a
bug localises to one pass rather than to "the scheduler".
"""
from __future__ import annotations

from hexlib.graph.dma import insert_transfers, plan_problems
from hexlib.graph.fuse import fuse
from hexlib.graph.ir import Graph
from hexlib.graph.liveness import live_intervals
from hexlib.graph.order import order
from hexlib.graph.ops import REGISTRY
from hexlib.graph.plan import Plan
from hexlib.graph.shapes import infer_shapes
from hexlib.graph.vtcm import allocate, allocation_problems
from hexlib.result import Err

PASSES = ("shapes", "fuse", "order", "liveness", "vtcm", "dma")


def compile_graph(
    graph: Graph,
    budget: int,
    order_policy: str = "min_peak",
    alloc_policy: str = "linear_scan",
    target: str = "hexagon-v75",
) -> Plan | Err:
    """Graph in, Plan out. The budget is a parameter and has no default."""
    if not isinstance(budget, int) or budget <= 0:
        return Err(
            "invalid VTCM budget",
            f"budget must be a positive int, got {budget!r}. VTCM is acquired at "
            "session start, so the budget comes from the runtime -- there is no "
            "safe default to fall back on.",
        )

    stage = infer_shapes(graph)
    if isinstance(stage, Err):
        return _at("shapes", stage)

    stage = fuse(stage)
    if isinstance(stage, Err):
        return _at("fuse", stage)

    stage = order(stage, policy=order_policy)
    if isinstance(stage, Err):
        return _at("order", stage)
    ordered = stage

    intervals = live_intervals(ordered)
    if isinstance(intervals, Err):
        return _at("liveness", intervals)

    allocation = allocate(intervals, budget=budget, policy=alloc_policy)
    if isinstance(allocation, Err):
        return _at("vtcm", allocation)
    slots, high_water = allocation

    problems = allocation_problems(slots, intervals, budget)
    if problems:
        return Err("vtcm: allocation violates its invariants", "\n".join(problems))

    transfers = insert_transfers(ordered, slots, budget)
    if isinstance(transfers, Err):
        return _at("dma", transfers)
    steps, moved = transfers

    problems = plan_problems(steps, ordered, slots, budget=budget)
    if problems:
        return Err("dma: schedule violates its invariants", "\n".join(problems))

    # kernel: None is reported explicitly as "not implemented", never treated
    # as a no-op.
    unimplemented = tuple(
        sorted(
            {
                op.kind
                for op in ordered.ops
                if REGISTRY.get(op.kind).kernel is None
            }
        )
    )

    return Plan(
        steps=steps,
        vtcm=slots,
        vtcm_high_water=high_water,
        predicted_bytes_moved=moved,
        vtcm_budget=budget,
        unimplemented=unimplemented,
        target=target,
    )


def _at(pass_name: str, err: Err) -> Err:
    return Err(f"{pass_name}: {err.reason}", err.detail)

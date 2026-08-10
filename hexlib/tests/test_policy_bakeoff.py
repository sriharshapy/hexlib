"""The M0/M1 bake-off: policies compete on predicted_bytes_moved and vtcm_high_water.

There is no hardware to measure against until M2, so these two numbers are the
metric. Real cycles join the comparison at M2; a policy that wins here and
loses on cycles is a finding about the cost model, and is recorded as one.
"""
from __future__ import annotations

import hexlib.graph.opdefs  # noqa: F401
from hexlib.graph.order import ORDER_POLICIES
from hexlib.graph.pipeline import compile_graph
from hexlib.graph.plan import Plan
from hexlib.graph.vtcm import ALLOC_POLICIES
from hexlib.models.qwen35 import qwen35_at
from hexlib.models.vit import build_vision_encoder

BUDGET = 8388608
REFERENCE_GRAPHS = {"qwen35@256": lambda: build_vision_encoder(qwen35_at(256))}


def test_the_bakeoff_table_is_complete_and_not_empty():
    assert ORDER_POLICIES and ALLOC_POLICIES
    rows = []
    for gname, make in REFERENCE_GRAPHS.items():
        graph = make()
        for op in sorted(ORDER_POLICIES):
            for ap in sorted(ALLOC_POLICIES):
                plan = compile_graph(graph, budget=BUDGET, order_policy=op, alloc_policy=ap)
                assert isinstance(plan, Plan), f"{gname} {op}/{ap}"
                rows.append((gname, op, ap, plan.vtcm_high_water, plan.predicted_bytes_moved))
    # `len(rows) == (number of loop iterations)` is tautological: rows is
    # appended exactly once per iteration of the same three loops, so this
    # can never fail regardless of what compile_graph returns. It is kept
    # as a basic sanity check, but the assertions below are what actually
    # relate the policies to each other -- their absence is why the
    # constant bytes_moved column went unnoticed for as long as it did.
    assert len(rows) == len(REFERENCE_GRAPHS) * len(ORDER_POLICIES) * len(ALLOC_POLICIES)
    print("\ngraph          order      alloc         high_water   bytes_moved")
    for row in rows:
        print(f"{row[0]:<14} {row[1]:<10} {row[2]:<13} {row[3]:>10,} {row[4]:>13,}")

    by_graph: dict[str, list[tuple]] = {}
    for row in rows:
        by_graph.setdefault(row[0], []).append(row)
    for gname, grows in by_graph.items():
        # predicted_bytes_moved is a structural property of the GRAPH: every
        # weight chunk, every resident const, and the graph's own input/
        # output are each moved exactly once, regardless of scheduling
        # (order_policy) or placement (alloc_policy). It being constant
        # across every row of one graph is the expected invariant, not an
        # unexamined coincidence -- assert it so a regression that makes it
        # policy-dependent (a real bug: it would mean some transfer is
        # being placed, or counted, more than once for some policy) is
        # caught here instead of going unnoticed again.
        moved_values = {r[4] for r in grows}
        assert len(moved_values) == 1, (
            f"{gname}: predicted_bytes_moved varies across policy pairs "
            f"({moved_values}); it should depend only on graph structure, "
            "never on order_policy or alloc_policy"
        )
        # vtcm_high_water DOES depend on the policy pair (that is the whole
        # point of baking policies off against each other); if every row
        # reported the same number, the table would have nothing to compare.
        high_waters = {r[3] for r in grows}
        assert len(high_waters) > 1, (
            f"{gname}: vtcm_high_water is identical across every policy pair "
            f"({high_waters}); the bake-off has nothing to compare"
        )


def test_no_policy_pair_produces_an_unplannable_combination():
    # `Plan.__post_init__` already makes vtcm_high_water > vtcm_budget
    # unconstructible, so `plan.vtcm_high_water <= BUDGET` (what this test
    # used to assert) can never fail once `plan` exists -- it never actually
    # checked that every pair PRODUCES a plan in the first place.
    graph = build_vision_encoder(qwen35_at(256))
    for op in ORDER_POLICIES:
        for ap in ALLOC_POLICIES:
            plan = compile_graph(graph, budget=BUDGET, order_policy=op, alloc_policy=ap)
            assert isinstance(plan, Plan), f"{op}/{ap}: {getattr(plan, 'detail', '')}"

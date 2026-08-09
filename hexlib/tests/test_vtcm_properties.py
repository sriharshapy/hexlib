"""Property tests: the allocator's invariants must hold on ANY interval set.

No hypothesis dependency -- a seeded loop over generated cases is enough here
and keeps the dependency list at numpy alone.
"""
from __future__ import annotations

import random

from hexlib.graph.liveness import Interval
from hexlib.graph.vtcm import ALLOC_POLICIES, allocate, allocation_problems
from hexlib.result import Err

CASES = 400
BUDGET = 1 << 22  # 4 MB, generous enough that overflow is rare but possible


def _random_intervals(rng: random.Random) -> tuple[Interval, ...]:
    n_ops = rng.randint(1, 40)
    count = rng.randint(1, 30)
    out = []
    for i in range(count):
        first = rng.randint(0, n_ops)
        last = min(n_ops, first + rng.randint(0, 10))
        out.append(
            Interval(
                tensor=f"t{i}",
                first_use=first,
                last_use=last,
                nbytes=rng.choice([1, 64, 127, 128, 1024, 65536, 262144]),
            )
        )
    return tuple(out)


def test_every_policy_satisfies_every_invariant_on_random_intervals():
    assert ALLOC_POLICIES, "policy table is empty; this test would pass vacuously"
    rng = random.Random(20260809)
    checked = 0
    for _ in range(CASES):
        intervals = _random_intervals(rng)
        for policy in ALLOC_POLICIES:
            result = allocate(intervals, budget=BUDGET, policy=policy)
            if isinstance(result, Err):
                # Overflow is a legitimate outcome; a WRONG allocation is not.
                assert "does not fit" in result.detail or "invariant" in result.reason
                continue
            slots, high_water = result
            problems = allocation_problems(slots, intervals, BUDGET)
            assert problems == [], f"{policy}: {problems}\nintervals={intervals}"
            assert high_water <= BUDGET
            assert {s.tensor for s in slots} == {iv.tensor for iv in intervals}
            checked += 1
    assert checked > 100, f"only {checked} allocations actually succeeded; the "
    "generator is producing overflow-only cases and proves nothing"


def test_a_deliberately_broken_allocator_is_caught_by_the_invariants():
    # The property test above is only evidence if the invariants can fail.
    from hexlib.graph.plan import Slot

    rng = random.Random(1)
    caught = 0
    for _ in range(200):
        intervals = _random_intervals(rng)
        # Everything at offset zero: correct only when nothing overlaps.
        bad = tuple(
            Slot(
                tensor=iv.tensor,
                offset=0,
                size=max(iv.nbytes, 1),
                first_use=iv.first_use,
                last_use=iv.last_use,
            )
            for iv in intervals
        )
        if allocation_problems(bad, intervals, BUDGET):
            caught += 1
    assert caught > 50, (
        f"only {caught}/200 broken allocations were caught; the invariants are "
        "not discriminating and the property test above proves nothing"
    )


def test_disjoint_intervals_always_reuse_space():
    rng = random.Random(7)
    for _ in range(100):
        size = rng.choice([128, 4096, 65536])
        count = rng.randint(2, 10)
        intervals = tuple(
            Interval(tensor=f"t{i}", first_use=2 * i, last_use=2 * i + 1, nbytes=size)
            for i in range(count)
        )
        for policy in ALLOC_POLICIES:
            slots, high_water = allocate(intervals, budget=BUDGET, policy=policy)
            assert high_water == size, (
                f"{policy} used {high_water} for {count} non-overlapping tensors of "
                f"{size} bytes; they should all share one offset"
            )

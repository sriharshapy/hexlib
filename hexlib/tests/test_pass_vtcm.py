from __future__ import annotations

from hexlib.graph.liveness import Interval
from hexlib.graph.vtcm import ALLOC_POLICIES, allocate, allocation_problems
from hexlib.result import Err


def _ivs(*specs):
    return tuple(
        Interval(tensor=n, first_use=f, last_use=l, nbytes=b) for n, f, l, b in specs
    )


def test_disjoint_intervals_may_share_an_offset():
    ivs = _ivs(("a", 0, 1, 1024), ("b", 2, 3, 1024))
    slots, high_water = allocate(ivs, budget=4096)
    offsets = {s.tensor: s.offset for s in slots}
    assert offsets["a"] == offsets["b"] == 0
    assert high_water == 1024


def test_overlapping_intervals_get_disjoint_ranges():
    ivs = _ivs(("a", 0, 3, 1024), ("b", 1, 2, 1024))
    slots, high_water = allocate(ivs, budget=4096)
    by_name = {s.tensor: s for s in slots}
    assert not (by_name["a"].offset < by_name["b"].end and by_name["b"].offset < by_name["a"].end)
    assert high_water == 2048


def test_high_water_is_the_maximum_end_offset():
    # Slots are rounded up to VTCM_ALIGN (128 B) before placement, so this is
    # align(1000) + align(500) = 1024 + 512, not 1000 + 500. The alignment is
    # not incidental -- an unaligned slot costs a split HVX load in every
    # kernel that touches it.
    ivs = _ivs(("a", 0, 3, 1000), ("b", 1, 2, 500))
    _, high_water = allocate(ivs, budget=4096)
    assert high_water == 1536


def test_overflow_is_an_err_naming_the_tensor_the_high_water_and_the_budget():
    ivs = _ivs(("a", 0, 3, 4096), ("big", 1, 2, 8192))
    out = allocate(ivs, budget=8192)
    assert isinstance(out, Err)
    assert "big" in out.detail
    assert "8192" in out.detail


def test_overflow_returns_no_partial_plan():
    out = allocate(_ivs(("big", 0, 1, 99999)), budget=1024)
    assert isinstance(out, Err)
    assert not hasattr(out, "slots")


def test_every_interval_gets_exactly_one_slot():
    ivs = _ivs(("a", 0, 3, 64), ("b", 1, 2, 64), ("c", 2, 4, 64))
    slots, _ = allocate(ivs, budget=4096)
    assert sorted(s.tensor for s in slots) == ["a", "b", "c"]


def test_offsets_are_aligned_to_128_bytes():
    # HVX loads want 128-byte alignment; an unaligned slot silently costs
    # a split load in every kernel that touches it.
    ivs = _ivs(("a", 0, 3, 100), ("b", 1, 2, 100))
    slots, _ = allocate(ivs, budget=4096)
    for s in slots:
        assert s.offset % 128 == 0, f"{s.tensor} at {s.offset}"


def test_an_empty_interval_list_is_an_err_not_an_empty_plan():
    # A allocator with nothing to allocate reporting success is the hazard
    # CONTRIBUTING.md names.
    out = allocate((), budget=4096)
    assert isinstance(out, Err)


def test_a_zero_budget_is_an_err():
    assert isinstance(allocate(_ivs(("a", 0, 1, 64)), budget=0), Err)


def test_an_unknown_policy_is_an_err_naming_the_known_ones():
    out = allocate(_ivs(("a", 0, 1, 64)), budget=4096, policy="magic")
    assert isinstance(out, Err)
    assert "magic" in out.detail
    assert "linear_scan" in out.detail


def test_every_registered_policy_produces_a_valid_allocation():
    assert ALLOC_POLICIES, "policy table is empty; this test would pass vacuously"
    ivs = _ivs(("a", 0, 3, 1024), ("b", 1, 2, 512), ("c", 2, 5, 2048))
    for name in ALLOC_POLICIES:
        result = allocate(ivs, budget=1 << 20, policy=name)
        assert not isinstance(result, Err), name
        slots, high_water = result
        assert allocation_problems(slots, ivs, 1 << 20) == [], name


def test_allocation_problems_catches_an_overlap():
    from hexlib.graph.plan import Slot

    ivs = _ivs(("a", 0, 3, 1024), ("b", 1, 2, 1024))
    bad = (
        Slot(tensor="a", offset=0, size=1024, first_use=0, last_use=3),
        Slot(tensor="b", offset=512, size=1024, first_use=1, last_use=2),
    )
    problems = allocation_problems(bad, ivs, budget=1 << 20)
    assert problems and "overlap" in problems[0]


def test_allocation_problems_catches_a_missing_slot():
    from hexlib.graph.plan import Slot

    ivs = _ivs(("a", 0, 3, 1024), ("b", 1, 2, 1024))
    only_a = (Slot(tensor="a", offset=0, size=1024, first_use=0, last_use=3),)
    problems = allocation_problems(only_a, ivs, budget=1 << 20)
    assert any("b" in p for p in problems)


def test_allocation_problems_catches_a_slot_smaller_than_its_tensor():
    from hexlib.graph.plan import Slot

    ivs = _ivs(("a", 0, 3, 1024),)
    too_small = (Slot(tensor="a", offset=0, size=16, first_use=0, last_use=3),)
    problems = allocation_problems(too_small, ivs, budget=1 << 20)
    assert problems and "1024" in problems[0]


def test_allocation_problems_catches_a_slot_evicted_while_live():
    from hexlib.graph.plan import Slot

    ivs = _ivs(("a", 0, 5, 1024),)
    short = (Slot(tensor="a", offset=0, size=1024, first_use=0, last_use=2),)
    problems = allocation_problems(short, ivs, budget=1 << 20)
    assert problems and ("live" in problems[0] or "evict" in problems[0])

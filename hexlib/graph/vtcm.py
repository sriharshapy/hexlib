"""VTCM allocation: register allocation with spilling, under a different name.

It is a known problem class, which is why this starts with linear scan and
makes the policy swappable rather than reaching for something clever. The
five invariants in `allocation_problems` hold for ANY allocator, so a
contributor can write a better one and prove it on a laptop.

The budget is ALWAYS a parameter. VTCM is acquired at session start, so the
part total is not what a process gets.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

from hexlib.graph.liveness import Interval
from hexlib.graph.plan import Slot
from hexlib.result import Err

VTCM_ALIGN = 128
"""HVX loads want 128-byte alignment. An unaligned slot silently costs a split
load in every kernel that touches it."""


def _align_up(value: int, to: int = VTCM_ALIGN) -> int:
    return ((value + to - 1) // to) * to


def _linear_scan(intervals: Sequence[Interval]) -> dict[str, int]:
    """Sort by first use; place each in the lowest gap that fits."""
    placed: list[tuple[int, int, Interval]] = []  # (offset, end, interval)
    offsets: dict[str, int] = {}
    for iv in sorted(intervals, key=lambda i: (i.first_use, -i.nbytes, i.tensor)):
        size = _align_up(iv.nbytes)
        busy = sorted(
            (off, end)
            for off, end, other in placed
            if other.overlaps(iv)
        )
        offset = 0
        for start, end in busy:
            if offset + size <= start:
                break
            offset = max(offset, end)
        offsets[iv.tensor] = offset
        placed.append((offset, offset + size, iv))
    return offsets


def _largest_first(intervals: Sequence[Interval]) -> dict[str, int]:
    """First-fit placement, largest interval first.

    This is NOT best-fit (smallest-sufficient-gap) despite the name this
    policy briefly had here. The only difference from `_linear_scan` is visitation
    order -- largest `nbytes` first rather than earliest `first_use` first --
    which is still a real, distinct placement and can pack differently, just
    not by the "best-fit" mechanism the term of art implies.
    """
    placed: list[tuple[int, int, Interval]] = []
    offsets: dict[str, int] = {}
    for iv in sorted(intervals, key=lambda i: (-i.nbytes, i.first_use, i.tensor)):
        size = _align_up(iv.nbytes)
        busy = sorted(
            (off, end)
            for off, end, other in placed
            if other.overlaps(iv)
        )
        offset = 0
        for start, end in busy:
            if offset + size <= start:
                break
            offset = max(offset, end)
        offsets[iv.tensor] = offset
        placed.append((offset, offset + size, iv))
    return offsets


ALLOC_POLICIES: Mapping[str, Callable[[Sequence[Interval]], dict[str, int]]] = {
    "linear_scan": _linear_scan,
    "largest_first": _largest_first,
}


def allocate(
    intervals: Sequence[Interval],
    budget: int,
    policy: str = "linear_scan",
) -> tuple[tuple[Slot, ...], int] | Err:
    if not intervals:
        return Err(
            "nothing to allocate",
            "the interval list is empty; an allocator with nothing to allocate "
            "reporting success is not a result",
        )
    if budget <= 0:
        return Err("invalid VTCM budget", f"budget must be positive, got {budget}")

    for iv in intervals:
        if iv.nbytes <= 0:
            return Err(
                "malformed interval",
                f"tensor {iv.tensor!r} has nbytes={iv.nbytes}; nbytes must be "
                "positive",
            )
        if iv.last_use < iv.first_use:
            return Err(
                "malformed interval",
                f"tensor {iv.tensor!r} has last_use={iv.last_use} before "
                f"first_use={iv.first_use}",
            )

    place = ALLOC_POLICIES.get(policy)
    if place is None:
        return Err(
            "unknown allocation policy",
            f"policy {policy!r} is not registered. Known: "
            f"{', '.join(sorted(ALLOC_POLICIES))}",
        )

    offsets = place(intervals)
    slots = tuple(
        sorted(
            (
                Slot(
                    tensor=iv.tensor,
                    offset=offsets[iv.tensor],
                    size=_align_up(iv.nbytes),
                    first_use=iv.first_use,
                    last_use=iv.last_use,
                )
                for iv in intervals
            ),
            key=lambda s: (s.offset, s.tensor),
        )
    )
    high_water = max(s.end for s in slots)

    if high_water > budget:
        worst = max(slots, key=lambda s: s.end)
        return Err(
            "VTCM allocation does not fit",
            f"high water {high_water} bytes exceeds the budget of {budget} bytes; "
            f"the tensor reaching furthest is {worst.tensor!r} "
            f"({worst.size} bytes at offset {worst.offset}). No partial plan is "
            "produced.",
        )

    problems = allocation_problems(slots, intervals, budget)
    if problems:
        return Err("allocation violates its own invariants", "\n".join(problems))

    return slots, high_water


def allocation_problems(
    slots: Sequence[Slot],
    intervals: Sequence[Interval],
    budget: int,
) -> list[str]:
    """Invariants 1, 2 and 4 from the spec. Checkable on ANY allocation.

    3 (every read preceded by a write or a completed DMA) and 5 (every op's
    working set is satisfied) need the op list and the DMA schedule; they live
    in dma.plan_problems and are called alongside this from the pipeline.
    """
    problems: list[str] = []
    by_name = {s.tensor: s for s in slots}
    by_interval = {iv.tensor: iv for iv in intervals}

    if len(by_name) != len(slots):
        problems.append("two slots name the same tensor")

    for iv in intervals:
        slot = by_name.get(iv.tensor)
        if slot is None:
            problems.append(f"tensor {iv.tensor!r} is live but has no slot")
            continue
        if slot.size < iv.nbytes:
            problems.append(
                f"slot for {iv.tensor!r} is {slot.size} bytes but the tensor needs "
                f"{iv.nbytes}"
            )
        # Invariant 4: no tensor is evicted while still live.
        if slot.last_use < iv.last_use or slot.first_use > iv.first_use:
            problems.append(
                f"slot for {iv.tensor!r} covers [{slot.first_use}, {slot.last_use}] "
                f"but the tensor is live over [{iv.first_use}, {iv.last_use}]; it "
                "would be evicted while still live"
            )

    # Invariant 1: no two simultaneously-live tensors overlap in VTCM.
    ordered = sorted(slots, key=lambda s: s.offset)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            if b.offset >= a.end:
                break
            ia, ib = by_interval.get(a.tensor), by_interval.get(b.tensor)
            if ia is not None and ib is not None and ia.overlaps(ib):
                problems.append(
                    f"slots for {a.tensor!r} [{a.offset}, {a.end}) and {b.tensor!r} "
                    f"[{b.offset}, {b.end}) overlap while both are live"
                )

    # Invariant 2: the high-water mark never exceeds the budget.
    if slots:
        high_water = max(s.end for s in slots)
        if high_water > budget:
            problems.append(f"high water {high_water} exceeds budget {budget}")

    return problems

"""A byte-accurate VTCM image: the plan's allocation, actually inhabited.

WHY BYTES AND NOT A DICT OF ARRAYS. The allocator's output is a set of
(offset, size) slots that are claimed not to overlap while both tensors are
live. A dict keyed by tensor name cannot disagree with that claim -- every
tensor gets its own storage no matter what the offsets say -- so an aliasing bug
would execute perfectly and prove nothing. Writing into one flat buffer at the
planned offsets is what makes the claim testable: if two live tensors share an
address, the second write corrupts the first and a downstream value goes wrong.

This is the one thing a host executor can check that the M1 passes cannot check
about themselves, and it matters because M1's overlap invariant is enforced by
`allocation_problems` -- code produced by the same pass whose output it judges.

STORED DTYPE VS COMPUTE DTYPE. Storage is the tensor's declared dtype, so an
fp16 activation occupies two bytes per element and the slot sizes are exercised
for real. Compute is fp32: `get` upcasts and `put` rounds on the way in. That is
what the kernels do -- `kernels/rmsnorm_fp16/kernel_api.h` states the reduction
and the reciprocal square root are computed in float and only the stored result
is rounded to fp16 -- so an interpreter that computed in fp16 throughout would
report accuracy the hardware does not actually have.
"""
from __future__ import annotations

import math

import numpy as np

from hexlib.graph.ir import Tensor
from hexlib.graph.plan import Slot

# Storage dtypes. Distinct from eager.NUMPY_DTYPE, which maps fp16 -> float32
# BECAUSE it is validating the math and not the storage format. Here the
# storage format is the point.
STORAGE_DTYPE: dict[str, np.dtype] = {
    "fp32": np.dtype(np.float32),
    "fp16": np.dtype(np.float16),
    "int32": np.dtype(np.int32),
}

COMPUTE_DTYPE = np.dtype(np.float32)


class VtcmError(Exception):
    pass


class VtcmImage:
    """`budget` bytes of addressable fast memory, indexed by the plan's slots."""

    def __init__(self, budget: int, slots: tuple[Slot, ...]) -> None:
        if budget <= 0:
            raise VtcmError(f"budget must be positive, got {budget}")
        self._buf = bytearray(budget)
        self.budget = budget
        self._slots: dict[str, Slot] = {}
        for slot in slots:
            if slot.tensor in self._slots:
                raise VtcmError(
                    f"two slots for tensor {slot.tensor!r}; a tensor has one address"
                )
            if slot.end > budget:
                raise VtcmError(
                    f"slot {slot.tensor!r} ends at {slot.end} which is past the "
                    f"{budget}-byte budget"
                )
            self._slots[slot.tensor] = slot
        # Written-byte bookkeeping, so reading storage nothing ever wrote is an
        # error rather than a silent field of zeros that looks like a plausible
        # activation.
        self._written: set[str] = set()

    def __contains__(self, name: str) -> bool:
        return name in self._slots

    def slot(self, name: str) -> Slot:
        try:
            return self._slots[name]
        except KeyError:
            raise VtcmError(f"no VTCM slot for tensor {name!r}") from None

    def _check_fits(self, tensor: Tensor, slot: Slot) -> None:
        need = tensor.nbytes
        if need > slot.size:
            raise VtcmError(
                f"tensor {tensor.name!r} needs {need} bytes but its slot holds "
                f"{slot.size}; the allocator and the tensor disagree about size"
            )

    def put(self, tensor: Tensor, array: np.ndarray) -> None:
        """Round to the stored dtype and write at the planned offset."""
        slot = self.slot(tensor.name)
        if tensor.dtype not in STORAGE_DTYPE:
            raise VtcmError(
                f"tensor {tensor.name!r} has dtype {tensor.dtype!r}, which has no "
                "dense storage form; block-quantized tensors are staged as raw "
                "bytes, not through put()"
            )
        if tuple(array.shape) != tensor.shape:
            raise VtcmError(
                f"tensor {tensor.name!r} is declared {tensor.shape} but the value "
                f"has shape {tuple(array.shape)}"
            )
        self._check_fits(tensor, slot)
        packed = np.ascontiguousarray(array, dtype=STORAGE_DTYPE[tensor.dtype])
        raw = packed.tobytes()
        if len(raw) != tensor.nbytes:
            raise VtcmError(
                f"tensor {tensor.name!r} packed to {len(raw)} bytes but nbytes says "
                f"{tensor.nbytes}"
            )
        self._buf[slot.offset:slot.offset + len(raw)] = raw
        self._written.add(tensor.name)

    def get(self, tensor: Tensor) -> np.ndarray:
        """Read at the planned offset and upcast to the compute dtype."""
        slot = self.slot(tensor.name)
        if tensor.name not in self._written:
            raise VtcmError(
                f"tensor {tensor.name!r} is read before anything wrote it. Its slot "
                "holds zeros, which would pass for an activation and quietly "
                "produce a wrong answer instead of an error."
            )
        self._check_fits(tensor, slot)
        raw = bytes(self._buf[slot.offset:slot.offset + tensor.nbytes])
        flat = np.frombuffer(raw, dtype=STORAGE_DTYPE[tensor.dtype])
        return flat.reshape(tensor.shape).astype(COMPUTE_DTYPE)

    def put_raw(self, name: str, raw: bytes) -> None:
        """Stage opaque bytes -- a block-quantized weight, which has no dense
        numpy form and is dequantized by whatever reads it."""
        slot = self.slot(name)
        if len(raw) > slot.size:
            raise VtcmError(
                f"{name!r}: {len(raw)} bytes do not fit a {slot.size}-byte slot"
            )
        self._buf[slot.offset:slot.offset + len(raw)] = raw
        self._written.add(name)

    def high_water(self) -> int:
        """The highest end offset over slots, which is what the plan's
        `vtcm_high_water` claims to be."""
        return max((s.end for s in self._slots.values()), default=0)

    def live_bytes(self) -> int:
        return sum(s.size for s in self._slots.values())


def overlapping_slots(slots: tuple[Slot, ...]) -> list[str]:
    """Slot pairs that share an address while both are live.

    Computed here, independently of `vtcm.allocation_problems`, and on purpose:
    the allocator's own invariant check is produced by the same pass that
    produces the allocation, so it cannot be the only thing that verifies it.
    Liveness is INCLUSIVE at both ends -- an op that reads `a` at step 5 and
    writes `b` at step 5 has both live at 5.
    """
    problems: list[str] = []
    ordered = sorted(slots, key=lambda s: s.offset)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if b.offset >= a.end:
                break  # sorted by offset, so nothing later can overlap `a`
            if a.first_use <= b.last_use and b.first_use <= a.last_use:
                problems.append(
                    f"{a.tensor!r} [{a.offset},{a.end}) live [{a.first_use},"
                    f"{a.last_use}] overlaps {b.tensor!r} [{b.offset},{b.end}) "
                    f"live [{b.first_use},{b.last_use}]"
                )
    return problems


def aligned_up(value: int, alignment: int = 128) -> int:
    return int(math.ceil(value / alignment) * alignment)

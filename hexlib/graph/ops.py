"""The op registry: an op is a record, not a class.

Everything every pass needs to know about an op kind lives in one `OpDef`.
Adding an op means adding a record, a kernel and a test -- never editing a
dispatch `switch` in five places. The eager executor is driven by this
registry rather than hand-written, so the oracle cannot drift from the
definitions it exists to validate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class OpDef:
    """One op kind. All three callables are mandatory.

    infer:        (inputs, attrs) -> ((shape, dtype), ...) one pair per output
    working_set:  (inputs, outputs, attrs) -> VTCM bytes that must be resident
    reference:    (arrays, attrs) -> (arrays, ...)   the numpy oracle
    kernel:       registered kernel symbol, or None for "not implemented yet"
    """

    kind: str
    infer: Callable[..., Any]
    working_set: Callable[..., int]
    reference: Callable[..., Any]
    kernel: str | None = None

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("OpDef requires a non-empty kind")
        for name in ("infer", "working_set", "reference"):
            if not callable(getattr(self, name)):
                raise TypeError(
                    f"OpDef {self.kind!r}: {name} must be callable. An op definition "
                    "missing one of its three callables would let a pass silently "
                    "skip this op kind."
                )


class Registry:
    __slots__ = ("_defs",)

    def __init__(self) -> None:
        self._defs: dict[str, OpDef] = {}

    def register(self, opdef: OpDef) -> None:
        if opdef.kind in self._defs:
            raise ValueError(
                f"op kind {opdef.kind!r} is already registered; two definitions for "
                "one kind means one of them is silently unused"
            )
        self._defs[opdef.kind] = opdef

    def get(self, kind: str) -> OpDef:
        try:
            return self._defs[kind]
        except KeyError:
            known = ", ".join(self.all_kinds()) or "<none>"
            raise KeyError(
                f"no OpDef registered for op kind {kind!r}. Registered kinds: {known}. "
                "An unregistered op is an error, never a no-op."
            ) from None

    def all_kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self._defs))


REGISTRY = Registry()


def register(opdef: OpDef) -> None:
    REGISTRY.register(opdef)


def get(kind: str) -> OpDef:
    return REGISTRY.get(kind)


def all_kinds() -> tuple[str, ...]:
    return REGISTRY.all_kinds()

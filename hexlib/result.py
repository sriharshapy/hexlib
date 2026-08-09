"""Fail-closed result types.

The one invariant: absence of results is failure, never success. `Ok` cannot be
constructed without `Measurements`, and `Measurements` cannot be constructed
without a cycle count. This is the structural form of the QDC false-pass
(job 742504: job `completed`, zero test results, exit 0) — there is no code path
that turns "we got nothing back" into a pass, because no such value exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union


@dataclass(frozen=True)
class Measurements:
    kernel_cycles: int
    toolchain_version: str
    sdk_version: str
    host: str
    timestamp: str

    def __post_init__(self) -> None:
        if not isinstance(self.kernel_cycles, int):
            raise TypeError(
                f"kernel_cycles must be an int, got {type(self.kernel_cycles).__name__}. "
                "A result without a cycle count is a failure, not a pass."
            )
        for field in ("toolchain_version", "sdk_version", "host", "timestamp"):
            if not getattr(self, field):
                raise TypeError(f"{field} must be a non-empty string")


class Ok:
    """A success. Cannot exist without measurements."""

    __slots__ = ("_value",)

    def __init__(self, value: Measurements) -> None:
        if not isinstance(value, Measurements):
            raise TypeError(
                "Ok requires Measurements; a success value with nothing measured "
                "is exactly the bug this type exists to prevent."
            )
        self._value = value

    def unwrap(self) -> Measurements:
        return self._value

    def __repr__(self) -> str:
        return f"Ok({self._value!r})"


class Err:
    """A failure, with a reason a human can act on."""

    __slots__ = ("reason", "detail")

    def __init__(self, reason: str, detail: str = "") -> None:
        if not reason:
            raise TypeError("Err requires a non-empty reason")
        self.reason = reason
        self.detail = detail

    def unwrap(self) -> Any:
        raise RuntimeError(f"{self.reason}\n{self.detail}".rstrip())

    def __repr__(self) -> str:
        return f"Err({self.reason!r})"


Result = Union[Ok, Err]


def is_ok(r: Result) -> bool:
    return isinstance(r, Ok)

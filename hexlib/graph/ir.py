"""The graph IR: what the numbers MEAN, never how they are arranged.

A `Tensor` carries a logical shape in numpy axis order and nothing else.
Strides, layouts, buffers and offsets are decided by the M1 allocator and
live in the plan, not here. Putting them in `Tensor` would force the graph
builder to know about repacking and VTCM placement before any pass has run.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

DTYPES = frozenset({"fp32", "fp16", "int32", "q4_0"})

# Dense dtypes only. q4_0 is block-structured and handled separately in nbytes().
_DENSE_BYTES = {"fp32": 4, "fp16": 2, "int32": 4}

# Q4_0: 32 four-bit values (16 bytes) + one fp16 scale (2 bytes) = 18 bytes.
# Spec 5.1.1; matches llama.cpp's block_q4_0.
Q4_0_BLOCK = 32
Q4_0_BLOCK_BYTES = 18

_ATTR_SCALARS = (bool, int, float, str, type(None))


def _check_attr_value(key: str, value: Any) -> None:
    if isinstance(value, _ATTR_SCALARS):
        return
    if isinstance(value, tuple):
        for v in value:
            _check_attr_value(key, v)
        return
    raise TypeError(
        f"attr {key!r} has value of type {type(value).__name__}; attrs must be "
        "scalars or tuples of scalars so a plan stays serializable"
    )


def nbytes(shape: tuple[int, ...], dtype: str) -> int:
    """Bytes occupied by a dense tensor of this shape and dtype."""
    if dtype not in DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; expected one of {sorted(DTYPES)}")
    numel = math.prod(shape) if shape else 1
    if dtype == "q4_0":
        if not shape or shape[-1] % Q4_0_BLOCK != 0:
            raise ValueError(
                f"q4_0 tensor of shape {shape} has last dim {shape[-1] if shape else 0}, which is not a "
                f"multiple of the {Q4_0_BLOCK}-element block size"
            )
        return numel // Q4_0_BLOCK * Q4_0_BLOCK_BYTES
    return numel * _DENSE_BYTES[dtype]


@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    const: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Tensor requires a non-empty name")
        if self.dtype not in DTYPES:
            raise ValueError(
                f"tensor {self.name!r} has unknown dtype {self.dtype!r}; "
                f"expected one of {sorted(DTYPES)}"
            )
        if not isinstance(self.shape, tuple):
            raise TypeError(f"tensor {self.name!r} shape must be a tuple")
        for d in self.shape:
            if not isinstance(d, int) or isinstance(d, bool) or d <= 0:
                raise ValueError(
                    f"tensor {self.name!r} has non-positive or non-integer dim {d!r}; "
                    "shapes are static and fully known"
                )

    @property
    def nbytes(self) -> int:
        return nbytes(self.shape, self.dtype)


@dataclass(frozen=True)
class Op:
    id: int
    kind: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError(f"op {self.id} requires a non-empty kind")

        # Coerce and validate inputs to tuple of strings
        inputs_tuple = tuple(self.inputs) if not isinstance(self.inputs, tuple) else self.inputs
        for i, val in enumerate(inputs_tuple):
            if not isinstance(val, str):
                raise TypeError(f"op {self.id}: inputs[{i}] must be str, got {type(val).__name__}")
        object.__setattr__(self, "inputs", inputs_tuple)

        # Coerce and validate outputs to tuple of strings
        outputs_tuple = tuple(self.outputs) if not isinstance(self.outputs, tuple) else self.outputs
        for i, val in enumerate(outputs_tuple):
            if not isinstance(val, str):
                raise TypeError(f"op {self.id}: outputs[{i}] must be str, got {type(val).__name__}")
        object.__setattr__(self, "outputs", outputs_tuple)

        for key, value in self.attrs.items():
            _check_attr_value(key, value)
        object.__setattr__(self, "attrs", MappingProxyType(dict(self.attrs)))


@dataclass(frozen=True)
class Graph:
    tensors: Mapping[str, Tensor]
    ops: tuple[Op, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))

        # Coerce and validate ops to tuple of Op instances
        ops_tuple = tuple(self.ops) if not isinstance(self.ops, tuple) else self.ops
        for i, val in enumerate(ops_tuple):
            if not isinstance(val, Op):
                raise TypeError(f"ops[{i}] must be Op, got {type(val).__name__}")
        object.__setattr__(self, "ops", ops_tuple)

        # Coerce and validate inputs to tuple of strings
        inputs_tuple = tuple(self.inputs) if not isinstance(self.inputs, tuple) else self.inputs
        for i, val in enumerate(inputs_tuple):
            if not isinstance(val, str):
                raise TypeError(f"inputs[{i}] must be str, got {type(val).__name__}")
        object.__setattr__(self, "inputs", inputs_tuple)

        # Coerce and validate outputs to tuple of strings
        outputs_tuple = tuple(self.outputs) if not isinstance(self.outputs, tuple) else self.outputs
        for i, val in enumerate(outputs_tuple):
            if not isinstance(val, str):
                raise TypeError(f"outputs[{i}] must be str, got {type(val).__name__}")
        object.__setattr__(self, "outputs", outputs_tuple)

    def tensor(self, name: str) -> Tensor:
        try:
            return self.tensors[name]
        except KeyError:
            raise KeyError(f"no tensor named {name!r} in this graph") from None

    def problems(self) -> list[str]:
        """Structural problems, as strings a human can act on. Empty means valid.

        An empty graph is a problem, not a pass: a validator with nothing to
        check reporting success is the failure mode CONTRIBUTING.md names.
        """
        problems: list[str] = []
        if not self.ops:
            problems.append("graph has no ops; an empty graph is not a valid encoder")
        if not self.outputs:
            problems.append("graph declares no outputs")

        for name, t in self.tensors.items():
            if name != t.name:
                problems.append(f"tensor keyed {name!r} but named {t.name!r}")

        seen_ids: set[int] = set()
        written: set[str] = set(self.inputs)
        for t in self.tensors.values():
            if t.const:
                written.add(t.name)

        for op in self.ops:
            if op.id in seen_ids:
                problems.append(f"duplicate op id {op.id}")
            seen_ids.add(op.id)
            for name in op.inputs:
                if name not in self.tensors:
                    problems.append(f"op {op.id} ({op.kind}) reads undeclared tensor {name!r}")
                elif name not in written:
                    problems.append(
                        f"op {op.id} ({op.kind}) reads {name!r} before it is written"
                    )
            for name in op.outputs:
                if name not in self.tensors:
                    problems.append(f"op {op.id} ({op.kind}) writes undeclared tensor {name!r}")
                elif name in written:
                    problems.append(f"tensor {name!r} is written twice (op {op.id})")
                else:
                    written.add(name)

        for name in self.inputs:
            if name not in self.tensors:
                problems.append(f"graph input {name!r} is not a declared tensor")
        for name in self.outputs:
            if name not in self.tensors:
                problems.append(f"graph output {name!r} is not a declared tensor")
            elif name not in written:
                problems.append(f"graph output {name!r} is never written")

        return problems

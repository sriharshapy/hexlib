"""Pass: verify every declared shape and dtype against the op registry.

The builder declares what it thinks each tensor is; `OpDef.infer` says what it
must be. Running both and comparing catches a builder bug here, cheaply, rather
than as a wrong number after allocation.
"""
from __future__ import annotations

from hexlib.graph.ir import Graph
from hexlib.graph.ops import REGISTRY, Registry
from hexlib.result import Err


def infer_shapes(graph: Graph, registry: Registry = REGISTRY) -> Graph | Err:
    problems = graph.problems()
    if problems:
        return Err("graph is structurally invalid", "\n".join(problems))

    for op in graph.ops:
        try:
            opdef = registry.get(op.kind)
        except KeyError as e:
            return Err("unregistered op kind", f"op {op.id}: {e.args[0]}")

        inputs = tuple(graph.tensor(name) for name in op.inputs)
        try:
            inferred = opdef.infer(inputs, op.attrs)
        except Exception as e:  # noqa: BLE001 -- a failing infer is a reportable Err
            return Err(
                "shape inference failed",
                f"op {op.id} ({op.kind}): {type(e).__name__}: {e}",
            )

        if len(inferred) != len(op.outputs):
            return Err(
                "output count mismatch",
                f"op {op.id} ({op.kind}) declares {len(op.outputs)} outputs but infer "
                f"returned {len(inferred)}",
            )

        for name, (shape, dtype) in zip(op.outputs, inferred):
            declared = graph.tensor(name)
            if tuple(shape) != declared.shape:
                return Err(
                    "declared shape disagrees with inference",
                    f"op {op.id} ({op.kind}) output {name!r} is declared "
                    f"{declared.shape} but infer says {tuple(shape)}",
                )
            if dtype != declared.dtype:
                return Err(
                    "declared dtype disagrees with inference",
                    f"op {op.id} ({op.kind}) output {name!r} is declared "
                    f"{declared.dtype!r} but infer says {dtype!r}",
                )

    return graph

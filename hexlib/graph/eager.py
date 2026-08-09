"""The reference executor -- the oracle every later milestone is measured against.

It is slow and obviously correct, and it stays that way. Spec 7: "The eager
executor is the oracle and must stay obviously correct. It is never optimized.
If it becomes slow enough to be inconvenient, that is accepted, not fixed."

It is driven by the op registry rather than by a hand-written dispatch table,
so it cannot drift from the op definitions it exists to validate.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from hexlib.graph.ir import Graph
from hexlib.graph.ops import REGISTRY, Registry
from hexlib.result import Err

# Every dtype is fed and computed as fp32 in the oracle. The oracle validates
# the MATH of the graph, not the numerics of a storage format: a q4_0 weight is
# fed as the fp32 values it represents. Quantization error is a separate
# measurement against this oracle, not a property of it.
NUMPY_DTYPE: Mapping[str, np.dtype] = {
    "fp32": np.dtype(np.float32),
    "fp16": np.dtype(np.float32),
    "int32": np.dtype(np.int32),
    "q4_0": np.dtype(np.float32),
}


def run(
    graph: Graph,
    feeds: Mapping[str, np.ndarray],
    registry: Registry = REGISTRY,
) -> dict[str, np.ndarray] | Err:
    """Execute `graph` on `feeds`, returning the declared outputs or an Err.

    `feeds` must supply every graph input and every const tensor. Anything
    missing, mis-shaped, or unregistered is an Err naming the offender --
    never a silent skip and never a partial result.
    """
    problems = graph.problems()
    if problems:
        return Err(
            "graph is structurally invalid",
            "\n".join(problems),
        )

    env: dict[str, np.ndarray] = {}
    required = list(graph.inputs) + [t.name for t in graph.tensors.values() if t.const]
    for name in required:
        if name not in feeds:
            return Err(
                "missing feed",
                f"tensor {name!r} is a graph input or a const and was not supplied; "
                f"supplied: {sorted(feeds)}",
            )
        spec = graph.tensor(name)
        try:
            array = np.asarray(feeds[name])
        except (ValueError, TypeError) as e:
            return Err(
                "feed conversion failed",
                f"tensor {name!r}: {type(e).__name__}: {e}",
            )
        if tuple(array.shape) != spec.shape:
            return Err(
                "feed shape mismatch",
                f"tensor {name!r} was declared {spec.shape} but the feed has shape "
                f"{tuple(array.shape)}",
            )
        try:
            env[name] = array.astype(NUMPY_DTYPE[spec.dtype], copy=False)
        except (ValueError, TypeError) as e:
            return Err(
                "feed dtype conversion failed",
                f"tensor {name!r}: {type(e).__name__}: {e}",
            )

    for op in graph.ops:
        try:
            opdef = registry.get(op.kind)
        except KeyError as e:
            return Err("unregistered op kind", f"op {op.id}: {e.args[0]}")

        arrays = tuple(env[name] for name in op.inputs)
        try:
            results = opdef.reference(arrays, op.attrs)
        except Exception as e:  # noqa: BLE001 -- a failing reference is a reportable Err
            return Err(
                "op reference raised",
                f"op {op.id} ({op.kind}): {type(e).__name__}: {e}",
            )

        if not isinstance(results, tuple):
            return Err(
                "op reference returned a non-tuple",
                f"op {op.id} ({op.kind}) returned {type(results).__name__}; "
                "references must return a tuple of arrays",
            )
        if len(results) != len(op.outputs):
            return Err(
                "op output count mismatch",
                f"op {op.id} ({op.kind}) declares {len(op.outputs)} outputs but its "
                f"reference returned {len(results)}",
            )

        for name, value in zip(op.outputs, results):
            spec = graph.tensor(name)
            try:
                value = np.asarray(value)
            except (ValueError, TypeError) as e:
                return Err(
                    "op result conversion failed",
                    f"op {op.id} ({op.kind}) output {name!r}: {type(e).__name__}: {e}",
                )
            if tuple(value.shape) != spec.shape:
                return Err(
                    "op result shape mismatch",
                    f"op {op.id} ({op.kind}) output {name!r} is declared {spec.shape} "
                    f"but the reference produced {tuple(value.shape)}",
                )
            try:
                env[name] = value.astype(NUMPY_DTYPE[spec.dtype], copy=False)
            except (ValueError, TypeError) as e:
                return Err(
                    "op result dtype conversion failed",
                    f"op {op.id} ({op.kind}) output {name!r}: {type(e).__name__}: {e}",
                )

    return {name: env[name] for name in graph.outputs}

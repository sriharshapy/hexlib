"""Pass: fold a bias add and an activation into the matmul that feeds them.

For the encoder's one bandwidth-bound op -- GELU on [n, 3072], about 1.3 flops
per byte -- fusion removes the memory pass entirely rather than making it
faster. That is why the plan has a fusion pass and does not have an l2fetch
mechanism (spec 4.3.1).
"""
from __future__ import annotations

from hexlib.graph.ir import Graph, Op, Tensor
from hexlib.result import Err

FUSABLE_ACTS = ("gelu_tanh", "gelu_erf")


def fuse(graph: Graph) -> Graph | Err:
    problems = graph.problems()
    if problems:
        return Err("graph is structurally invalid", "\n".join(problems))

    consumers: dict[str, int] = {}
    for op in graph.ops:
        for name in op.inputs:
            consumers[name] = consumers.get(name, 0) + 1
    protected = set(graph.outputs)

    producer = {op.outputs[0]: op for op in graph.ops if len(op.outputs) == 1}
    by_id = {op.id: op for op in graph.ops}
    consumed: set[int] = set()
    rewritten: dict[int, Op] = {}
    dropped_tensors: set[str] = set()

    def single_consumer(name: str) -> bool:
        return consumers.get(name, 0) == 1 and name not in protected

    for op in graph.ops:
        if op.id in consumed or op.kind != "matmul":
            continue
        mm_out = op.outputs[0]
        if not single_consumer(mm_out):
            continue

        add_op = _sole_consumer(graph, mm_out, consumed)
        if add_op is None or add_op.kind != "add":
            continue
        # The second operand must be a const 1-D bias, not another activation:
        # a residual add(x, proj) has the same shape signature and must not fuse.
        bias_name = add_op.inputs[1] if add_op.inputs[0] == mm_out else None
        if bias_name is None:
            continue
        bias = graph.tensors.get(bias_name)
        if bias is None or not bias.const or len(bias.shape) != 1:
            continue

        act_kind = "none"
        tail = add_op
        add_out = add_op.outputs[0]
        if single_consumer(add_out):
            act_op = _sole_consumer(graph, add_out, consumed | {add_op.id})
            if act_op is not None and act_op.kind in FUSABLE_ACTS:
                act_kind = act_op.kind
                tail = act_op

        rewritten[op.id] = Op(
            id=op.id,
            kind="matmul_epilogue",
            inputs=(op.inputs[0], op.inputs[1], bias_name),
            outputs=tail.outputs,
            attrs={"act": act_kind},
        )
        consumed.add(add_op.id)
        dropped_tensors.add(mm_out)
        if tail is not add_op:
            consumed.add(tail.id)
            dropped_tensors.add(add_out)

    if not rewritten:
        return graph

    ops = tuple(
        rewritten.get(op.id, op) for op in graph.ops if op.id not in consumed
    )
    tensors = {
        name: t for name, t in graph.tensors.items() if name not in dropped_tensors
    }
    return Graph(tensors=tensors, ops=ops, inputs=graph.inputs, outputs=graph.outputs)


def _sole_consumer(graph: Graph, name: str, skip: set[int]) -> Op | None:
    found = [op for op in graph.ops if name in op.inputs and op.id not in skip]
    return found[0] if len(found) == 1 else None

"""Every real-work op in the compiled encoder must select a kernel.

WHY THIS EXISTS. "Has a gated kernel" is not "can be dispatched" -- a
RunnerSpec is what makes an op reachable on the DSP batch path, and this
project has published a wrong coverage number three times by counting kernel
directories instead. layernorm was gated and unreachable; softmax and rope_2d
repeated it the same week; matmul and matmul_epilogue repeated it again.

Counting is what goes wrong, so this test does not count. It asks select()
about every step of the real compiled plan.
"""
import collections

import pytest

import hexlib.graph.opdefs  # noqa: F401 -- registers the op definitions
from hexlib.exec.runner import select
from hexlib.graph.pipeline import compile_model
from hexlib.graph.plan import V75_VTCM_TOTAL_BYTES
from hexlib.models.qwen35 import qwen35_at
from hexlib.models.vit import build_vision_encoder


@pytest.fixture(scope="module")
def encoder_plan():
    graph = build_vision_encoder(qwen35_at(256))
    compiled = compile_model(graph, budget=V75_VTCM_TOTAL_BYTES)
    assert not hasattr(compiled, "reason"), (
        f"compile: {getattr(compiled, 'detail', compiled)}"
    )
    return compiled.plan


def test_every_non_reshape_step_selects_a_kernel(encoder_plan):
    undispatchable = collections.Counter()
    for step in encoder_plan.steps:
        kind = step.op.kind
        if kind == "reshape":
            continue
        try:
            select(kind, dict(step.op.attrs or {}))
        except Exception:
            undispatchable[kind] += 1
    assert not undispatchable, (
        f"ops with no reachable kernel: {dict(undispatchable)}. A gated kernel "
        f"is not enough -- each needs a RunnerSpec in hexlib/exec/runner.py."
    )


def test_the_plan_is_the_expected_size(encoder_plan):
    """A guard on the guard: if the encoder shrank, the test above could pass
    while covering almost nothing."""
    kinds = collections.Counter(s.op.kind for s in encoder_plan.steps)
    assert len(encoder_plan.steps) == 308
    assert kinds["reshape"] == 49
    assert kinds["matmul"] == 24
    assert kinds["matmul_epilogue"] == 75

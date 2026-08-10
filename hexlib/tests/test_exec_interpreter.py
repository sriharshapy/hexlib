"""Executing a plan: the encoder runs, and a bad allocation produces a bad number.

The M1 passes were validated structurally -- offsets do not overlap, transfers
are well formed, the high water is the maximum slot end -- and every one of those
checks is performed by code from the same pass that produced the thing checked.
These tests execute the plan through its own VTCM allocation instead.

`test_aliased_slots_actually_corrupt_a_value` is the one that gives the rest
their meaning. Without it, the byte-level image would be an elaborate way to get
the same answer a dict of arrays gives, and an aliasing bug would still pass.
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

import hexlib.graph.opdefs  # noqa: F401  -- registers the op definitions
from hexlib.exec import interpreter
from hexlib.exec.vtcm import VtcmError, VtcmImage, overlapping_slots
from hexlib.graph.compiled import Compiled
from hexlib.graph.ir import Graph, Op, Tensor
from hexlib.graph.pipeline import compile_model
from hexlib.graph.plan import Plan, Slot, Step
from hexlib.models.vit import build_vision_encoder
from hexlib.result import Err
from hexlib.tests.test_vision_oracle import TINY_CFG, VISION_NPZ

TINY_BUDGET = 1 << 22


def _golden():
    if not os.path.isfile(VISION_NPZ):
        pytest.skip("golden vectors are not committed")
    return np.load(VISION_NPZ)


def _run(act: str, weight: str):
    from hexlib.tests.test_vision_oracle import _feeds_from_golden

    z = _golden()
    cfg = dataclasses.replace(TINY_CFG, act_dtype=act, weight_dtype=weight)
    graph = build_vision_encoder(cfg)
    assert not isinstance(graph, Err), graph
    model = compile_model(graph, budget=TINY_BUDGET)
    assert not isinstance(model, Err), model
    report = interpreter.run(model, _feeds_from_golden(graph, z))
    assert not isinstance(report, Err), f"{report.reason}\n{report.detail}"
    return model, report, z


# --- the gate: the plan, executed, still matches PyTorch --------------------


def test_fp32_plan_execution_matches_pytorch():
    """The M0 gate, but reached through the plan and a byte-addressed VTCM
    image rather than through the eager executor's dict of arrays."""
    model, report, z = _run("fp32", "fp32")
    got = report.outputs[model.graph.outputs[0]]
    want = z["expected_merged"]
    assert got.shape == want.shape
    assert np.max(np.abs(got - want)) < 1e-6


def test_fp16_storage_costs_less_than_a_thousandth_relative():
    """Storing activations as real fp16 is what the hardware does. This records
    what it costs end to end, so a later regression is visible as a number."""
    model, report, z = _run("fp16", "fp16")
    got = report.outputs[model.graph.outputs[0]]
    want = z["expected_merged"]
    rel = np.max(np.abs(got - want)) / np.max(np.abs(want))
    assert rel < 1e-3, f"fp16 relative error {rel:.3e} is worse than expected"
    # And it must be genuinely worse than fp32, or the fp16 path is not
    # actually storing fp16 and this test is measuring nothing.
    _, fp32_report, _ = _run("fp32", "fp32")
    fp32_err = np.max(np.abs(fp32_report.outputs[model.graph.outputs[0]] - want))
    assert np.max(np.abs(got - want)) > fp32_err


def test_actual_traffic_equals_the_predicted_traffic():
    """The plan's cost model, checked against execution instead of asserted."""
    for act, weight in (("fp32", "fp32"), ("fp16", "fp16")):
        model, report, _ = _run(act, weight)
        assert report.bytes_moved == model.plan.predicted_bytes_moved, (
            f"{act}/{weight}: moved {report.bytes_moved} but the plan predicted "
            f"{model.plan.predicted_bytes_moved}"
        )


def test_every_read_is_accounted_for_by_a_slot_or_a_tiling():
    """A read from host memory that no transfer and no tiling explains is a gap
    in the plan: the plan is supposed to account for what a kernel reads."""
    for act, weight in (("fp32", "fp32"), ("fp16", "fp16")):
        _, report, _ = _run(act, weight)
        assert report.unstaged_reads == [], sorted(set(report.unstaged_reads))


def test_streamed_weights_are_the_matmul_weights_and_have_no_slot():
    """Chunk-streamed weights deliberately have no Slot -- only buffers*chunk
    bytes are resident, which is why the high water is chunk-sized rather than
    layer-sized. Recorded so 'no slot' is never read as 'forgotten'."""
    model, report, _ = _run("fp16", "fp16")
    slotted = {s.tensor for s in model.plan.vtcm}
    assert report.streamed_weights, "no weight was streamed; the tiling is gone"
    for name in report.streamed_weights:
        assert name not in slotted, f"{name} is both streamed and resident"
        assert model.graph.tensor(name).const


def test_every_step_ran():
    model, report, _ = _run("fp16", "fp16")
    assert report.steps_run == len(model.plan.steps)
    assert report.ops_run == len(model.graph.ops)


# --- the teeth ------------------------------------------------------------


def _aliasing_case(p_offset: int, q_offset: int):
    """r = 2x + 3x, with p and q placed by hand.

    Both p and q are live when op 2 reads them, so giving them the same address
    means writing q destroys p and the result is 6x instead of 5x.
    """
    n = 8
    shape = (n,)
    tensors = {
        "x": Tensor(name="x", dtype="fp32", shape=shape),
        "p": Tensor(name="p", dtype="fp32", shape=shape),
        "q": Tensor(name="q", dtype="fp32", shape=shape),
        "r": Tensor(name="r", dtype="fp32", shape=shape),
    }
    ops = (
        Op(id=0, kind="scale", inputs=("x",), outputs=("p",), attrs={"factor": 2.0}),
        Op(id=1, kind="scale", inputs=("x",), outputs=("q",), attrs={"factor": 3.0}),
        Op(id=2, kind="add", inputs=("p", "q"), outputs=("r",), attrs={}),
    )
    graph = Graph(tensors=tensors, ops=ops, inputs=("x",), outputs=("r",))
    size = n * 4
    slots = (
        Slot(tensor="x", offset=0, size=size, first_use=0, last_use=1),
        Slot(tensor="p", offset=p_offset, size=size, first_use=0, last_use=2),
        Slot(tensor="q", offset=q_offset, size=size, first_use=1, last_use=2),
        Slot(tensor="r", offset=1024, size=size, first_use=2, last_use=2),
    )
    steps = tuple(
        Step(op=op, dma_in=(), dma_wait=(), dma_out=()) for op in ops
    )
    plan = Plan(
        steps=steps,
        vtcm=slots,
        vtcm_high_water=max(s.end for s in slots),
        predicted_bytes_moved=0,
        vtcm_budget=4096,
        unimplemented=(),
    )
    return Compiled(graph=graph, plan=plan)


def test_disjoint_slots_give_the_right_answer():
    """The control. Without this the corruption test below proves nothing --
    the machinery has to produce 5x when the allocation is sound."""
    x = np.arange(8, dtype=np.float32) + 1.0
    model = _aliasing_case(p_offset=128, q_offset=256)
    report = interpreter.run(model, {"x": x})
    assert not isinstance(report, Err), report
    np.testing.assert_allclose(report.outputs["r"], 5.0 * x)


def test_aliased_slots_actually_corrupt_a_value():
    """Two simultaneously-live tensors at one address produce a WRONG NUMBER.

    This is what makes the byte-addressed image worth having. A dict of arrays
    keyed by tensor name cannot fail this way, so it could never detect the
    allocator bug this exists to detect.
    """
    x = np.arange(8, dtype=np.float32) + 1.0
    model = _aliasing_case(p_offset=128, q_offset=128)
    report = interpreter.run(model, {"x": x}, check_allocation=False)
    assert not isinstance(report, Err), report
    got = report.outputs["r"]
    assert not np.allclose(got, 5.0 * x), (
        "aliased slots produced the correct answer, so the VTCM image is not "
        "actually storing tensors at the offsets the plan assigns them"
    )
    np.testing.assert_allclose(got, 6.0 * x)


def test_aliasing_is_refused_before_execution_by_default():
    x = np.arange(8, dtype=np.float32) + 1.0
    model = _aliasing_case(p_offset=128, q_offset=128)
    result = interpreter.run(model, {"x": x})
    assert isinstance(result, Err)
    assert "alias" in result.reason


def test_overlap_check_is_inclusive_at_both_ends():
    """a(0,1) and b(1,2) are BOTH live during op 1, which reads a and writes b.
    A strict comparison would call them disjoint and let them share storage."""
    slots = (
        Slot(tensor="a", offset=0, size=64, first_use=0, last_use=1),
        Slot(tensor="b", offset=0, size=64, first_use=1, last_use=2),
    )
    assert overlapping_slots(slots), "inclusive liveness overlap was missed"


def test_non_overlapping_lifetimes_may_share_an_address():
    slots = (
        Slot(tensor="a", offset=0, size=64, first_use=0, last_use=1),
        Slot(tensor="b", offset=0, size=64, first_use=2, last_use=3),
    )
    assert overlapping_slots(slots) == []


# --- fail closed ----------------------------------------------------------


def test_reading_storage_nothing_wrote_is_an_error_not_zeros():
    """Zeros would pass for an activation and give a plausible wrong answer."""
    image = VtcmImage(1024, (Slot(tensor="a", offset=0, size=32, first_use=0, last_use=1),))
    with pytest.raises(VtcmError, match="before anything wrote it"):
        image.get(Tensor(name="a", dtype="fp32", shape=(8,)))


def test_a_slot_too_small_for_its_tensor_is_an_error():
    image = VtcmImage(1024, (Slot(tensor="a", offset=0, size=16, first_use=0, last_use=1),))
    with pytest.raises(VtcmError, match="disagree about size"):
        image.put(Tensor(name="a", dtype="fp32", shape=(8,)), np.zeros(8, np.float32))


def test_missing_feed_is_an_err():
    model = _aliasing_case(p_offset=128, q_offset=256)
    result = interpreter.run(model, {})
    assert isinstance(result, Err)
    assert result.reason == "missing feed"


def test_a_supplied_backend_replaces_the_reference_one():
    """The hook a real kernel arrives through. If the supplied backend were
    ignored, every kernel measurement would silently be of the reference."""
    x = np.arange(8, dtype=np.float32) + 1.0
    model = _aliasing_case(p_offset=128, q_offset=256)
    calls = []

    def fake_scale(arrays, attrs):
        calls.append(attrs["factor"])
        return (arrays[0] * 0.0,)

    report = interpreter.run(model, {"x": x}, backends={"scale": fake_scale})
    assert not isinstance(report, Err), report
    assert sorted(calls) == [2.0, 3.0]
    np.testing.assert_allclose(report.outputs["r"], np.zeros(8, np.float32))
    assert report.backend_used["scale"] == "supplied"
    assert report.backend_used["add"] == "reference"


def test_a_backend_returning_the_wrong_shape_is_an_err():
    x = np.arange(8, dtype=np.float32) + 1.0
    model = _aliasing_case(p_offset=128, q_offset=256)
    report = interpreter.run(
        model, {"x": x}, backends={"scale": lambda a, at: (np.zeros(3, np.float32),)}
    )
    assert isinstance(report, Err)
    assert "shape mismatch" in report.reason

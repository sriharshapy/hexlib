# hexlib/tests/test_wholeplan.py
"""The whole plan as ONE batch: a single entry point into the DSP, one exit.

WHAT THESE PROVE THAT `test_encoder_on_sim.py` DOES NOT. That file runs the
encoder with one simulator launch per op. It proves the kernels and the wiring;
it cannot prove that 259 ops in a SINGLE blob address each other correctly,
because it never builds one. The failure modes are different in kind: a tensor
index that is right per-op and wrong in a shared table, an arena offset that
collides with a live value, a reshape elided into an address someone else owns.

Most of these need no SDK. The arena is built and checked on the host; only the
last test launches a simulator.
"""
import os
import struct

import numpy as np
import pytest

import hexlib.graph.opdefs  # noqa: F401  -- registers the op definitions
from hexlib import toolchain as tc
from hexlib.exec.wholeplan import WholePlanError, build_whole_plan_batch
from hexlib.graph.pipeline import compile_model
from hexlib.models.vit import build_vision_encoder
from hexlib.runtime import wire
from hexlib.tests.test_encoder_on_sim import _feeds, _tiny_cfg

HAS_SDK = os.path.isdir(tc.default_sdk_root())
sdk = pytest.mark.skipif(not HAS_SDK, reason="Hexagon SDK not present")

VTCM_BUDGET = 8 * 1024 * 1024


def _built():
    graph = build_vision_encoder(_tiny_cfg())
    compiled = compile_model(
        graph, budget=VTCM_BUDGET,
        order_policy="min_peak", alloc_policy="largest_first",
    )
    feeds = _feeds(compiled.graph, compiled.plan)
    return compiled, feeds, build_whole_plan_batch(compiled, feeds)


def test_every_real_work_op_is_in_one_batch():
    """The count, asserted rather than eyeballed. A spec that stopped matching
    would quietly shrink this batch, and the run would still succeed -- on
    fewer ops than the encoder has."""
    compiled, _, b = _built()
    steps = [s for s in compiled.plan.steps if getattr(s, "op", None) is not None]
    reshapes = [s for s in steps if s.op.kind == "reshape"]
    assert b.n_ops == len(steps) - len(reshapes)
    assert b.skipped_reshapes == len(reshapes)
    assert b.n_ops > 30, f"only {b.n_ops} ops reached the batch"


def test_the_blob_declares_the_ops_it_carries():
    """`wire.py` has `pack_batch` and `unpack_response` and no batch-header
    DECODER -- the only thing that parses a batch is `skel_dispatch.c`. So this
    reads `n_ops` out of the packed header directly rather than skipping.

    An earlier version of this test called a decoder that does not exist and
    `pytest.skip`ped when `hasattr` said so, which made it a test that could
    never run and never fail -- this project's own named failure mode, in the
    file that exists to catch that class of thing."""
    _, _, b = _built()
    hdr = struct.unpack(wire._HDR, b.blob[:wire.HDR_SIZE])
    assert b.n_ops in hdr, (
        f"n_ops={b.n_ops} appears nowhere in the packed header {hdr}"
    )


def test_no_tensor_runs_past_the_end_of_the_arena():
    """`pack_batch` checks this per tensor against the declared buffer size. The
    check here is the complementary one: the arena is actually that big."""
    _, _, b = _built()
    assert len(b.payload) == b.arena_bytes
    for name, p in b.placements.items():
        assert p.offset + p.nbytes <= b.arena_bytes, name


def test_two_live_tensors_never_share_an_address():
    """THE INVARIANT THE WHOLE ARENA RESTS ON, checked directly rather than
    inferred from the allocator having run.

    With `alias_activations=False` no activation shares an address at all, so
    this is a strong statement: any overlap is a placement bug."""
    _, _, b = _built()
    acts = [p for p in b.placements.values() if p.region == "act"]
    acts.sort(key=lambda p: p.offset)
    for a, c in zip(acts, acts[1:]):
        assert a.offset + a.nbytes <= c.offset, (
            f"{a.name} [{a.offset}, {a.offset + a.nbytes}) overlaps "
            f"{c.name} at {c.offset}"
        )


def test_a_reshape_output_shares_its_inputs_bytes_and_emits_no_op():
    """A reshape is a reinterpretation, so its output must land on its input's
    storage with the SAME byte count and a DIFFERENT ne. If it ever gets its own
    offset, the value is silently never written there."""
    compiled, _, b = _built()
    pairs = [
        (s.op.outputs[0], s.op.inputs[0])
        for s in compiled.plan.steps
        if getattr(s, "op", None) is not None and s.op.kind == "reshape"
    ]
    assert pairs, "the tiny encoder has no reshape; this test is vacuous"
    for out, inp in pairs:
        po, pi = b.placements[out], b.placements[inp]
        assert po.offset == pi.offset, f"{out} does not share {inp}'s storage"
        assert po.nbytes == pi.nbytes


def test_aliasing_the_plans_vtcm_slots_is_refused_not_silently_wrong():
    """THE BUG THIS GUARD EXISTS FOR, PINNED.

    Reusing the plan's VTCM offsets as flat-arena addresses looks obviously
    right and is not: this builder moves a reshape output onto its input, which
    keeps the input's slot live past the point the allocator was told it died.
    Measured at the tiny config, the encoder's output correlation with the
    reference fell from 1.000000 to 0.277 -- a wrong answer indistinguishable
    from a kernel bug.

    The allocator is NOT at fault; its own slots have zero overlapping live
    ranges. So this asserts the REFUSAL, because a mode that silently corrupts
    is worse than one that does not exist."""
    compiled, feeds, _ = _built()
    with pytest.raises(WholePlanError, match="live"):
        build_whole_plan_batch(compiled, feeds, alias_activations=True)


def test_the_plans_own_slots_do_not_overlap_in_live_range():
    """The companion to the test above, and the reason it can name a culprit.
    Without this, 'aliasing is unsafe' would be equally consistent with a broken
    allocator, and the fix would have been attempted in the wrong file."""
    compiled, _, _ = _built()
    slots = list(compiled.plan.vtcm)
    by_offset = {}
    for s in slots:
        by_offset.setdefault(s.offset, []).append(s)
    for offset, group in by_offset.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, c = group[i], group[j]
                assert not (a.first_use <= c.last_use and c.first_use <= a.last_use), (
                    f"the allocator put {a.tensor} and {c.tensor} at offset "
                    f"{offset} with overlapping live ranges"
                )


def test_a_const_is_staged_in_the_dtype_its_consumer_declares():
    """`pos_embed` is fp32 in the graph and `add` declares both inputs fp16.
    The per-op path coerces at call time; an arena has to do it at staging, and
    getting this wrong is an ERR_REQUIRES from the DSP rather than a wrong
    answer -- which is why it is worth pinning that it stays right."""
    _, _, b = _built()
    p = b.placements["pos_embed"]
    assert p.dtype == "fp16", (
        f"pos_embed staged as {p.dtype}; add declares fp16 inputs and the DSP's "
        f"generated entry checks it"
    )


@sdk
def test_the_whole_encoder_runs_as_a_single_invoke_and_matches_the_reference():
    """ONE batch, ONE simulator launch, 49 ops, compared against the op
    registry's numpy reference over the encoder's declared outputs.

    The tolerance is the same 5% `test_encoder_on_sim.py` uses and for the same
    reason -- fp16 storage compounding across two layers. What makes this test
    worth its runtime is not the bound but that the single-blob path reaches it
    at all: every tensor index, every arena offset and every elided reshape has
    to be right simultaneously for the answer to land anywhere near."""
    from hexlib.exec import dsp as dspmod
    from hexlib.exec import interpreter
    from hexlib.exec.runner import SPECS, WIRE_DTYPE
    from hexlib.tests.test_encoder_on_sim import _result, _work_dir

    compiled, feeds, b = _built()
    work = os.environ.get("HEXLIB_SIM_WORK") or _work_dir()
    sim = dspmod.DspSimBackend(
        sorted({os.path.basename(s.kernel_dir) for s in SPECS.values()}), work
    )
    sim._write_call(b.blob, bytes(b.payload))
    res = dspmod.run_sim(work, sdk_root=sim.sdk_root)
    assert res.status == wire.STATUS["OK"], (
        f"batch status {wire.STATUS_NAME.get(res.status, res.status)}"
    )

    rsp = sim._read_response()
    assert len(rsp.results) == b.n_ops
    bad = [(i, r) for i, r in enumerate(rsp.results) if not r.ok]
    assert not bad, (
        "ops failed on the DSP: "
        + ", ".join(f"{b.op_names[i]}="
                    f"{wire.STATUS_NAME.get(r.status, r.status)}" for i, r in bad[:5])
    )
    assert res.cycles > 0, "cycles_total is zero; nothing was measured"

    arena = open(os.path.join(work, "hexlib_out.bin"), "rb").read()
    assert len(arena) == b.arena_bytes

    ref = interpreter.run(compiled, feeds)
    assert not hasattr(ref, "reason"), getattr(ref, "detail", ref)

    for name in b.outputs:
        p = b.placements[name]
        got = np.frombuffer(
            arena[p.offset:p.offset + p.nbytes], dtype=WIRE_DTYPE[p.dtype]
        ).astype(np.float64)
        want = np.asarray(_result(ref, name), dtype=np.float64).reshape(-1)
        assert got.shape == want.shape
        assert np.all(np.isfinite(got)), f"{name}: non-finite values"
        rel = float(np.abs(got - want).max()) / max(float(np.abs(want).max()), 1e-6)
        assert rel < 0.05, f"{name}: max relative error {rel:.4e}"
        corr = float(np.corrcoef(got, want)[0, 1])
        assert corr > 0.99, f"{name}: correlation {corr:.6f}"

# hexlib/tests/test_encoder_on_sim.py
"""THE WHOLE ENCODER, end to end, with every kernel that exists running on the
Hexagon simulator's DSP batch path.

WHAT THIS IS FOR. Everything else that touches the simulator drives ONE op. That
proves a kernel and its wiring; it does not prove the encoder. The failures this
catches are the ones that only exist between ops:

  * a kernel that is correct in isolation and wrong on the shape the encoder
    actually hands it (the gate ran layernorm at R=4; the encoder needs R=256);
  * a spec whose scalars are right for the harness's shape and wrong for the
    graph's, which no single-op test at a hand-chosen shape can see;
  * an op whose output feeds another kernel rather than a test assertion, so a
    systematic error is only visible after it has propagated;
  * a variant routed to the wrong kernel for SOME of its ops -- `transpose` is
    two kernels and the encoder uses both, in the same graph, interleaved.

HOW IT COMPARES. Twice through `interpreter.run` over the same compiled plan and
the same feeds: once with DSP backends for every kind that has a kernel, once
with none, so the second run is the op registry's numpy reference throughout.
The reference path is the one the plan executor was validated against, so a
disagreement is the DSP's -- and the comparison is over the encoder's declared
OUTPUTS, after every intermediate has passed through the DSP and back.

AT A TINY CONFIG, DELIBERATELY, AND THE REASON IS NOT CONVENIENCE. Every call to
a DSP backend is one `hexagon-sim` launch -- process start, QuRT boot, run, tear
down. The 256x256 encoder is 259 real-work ops, so a full-size run is 259
launches of a simulator that takes seconds each even before the work, plus
patchify at ~2.0M cycles. Hours, for a correctness signal that a 58-step graph
gives in minutes. What a tiny config does NOT cover is recorded in the test that
needs it: shape-dependent behaviour still belongs in the per-kernel gate and in
`test_dsp_sim.py`, which drives the encoder's real shapes one op at a time.

THIS TEST GETS STRONGER ON ITS OWN. It asks `runner.SPECS` which kinds have
kernels rather than listing them, so registering a kernel moves ops onto the DSP
here with no edit. It also ASSERTS how many ops went to the DSP, so a spec that
silently stops matching -- a `requires` that no longer fits the graph, a variant
whose attrs drifted -- fails loudly instead of quietly falling back to the
reference and still passing.
"""
import os

import numpy as np
import pytest

import hexlib.graph.opdefs  # noqa: F401  -- registers the op definitions
from hexlib import toolchain as tc
from hexlib.exec import dsp as dspmod
from hexlib.exec import interpreter
from hexlib.exec.runner import SPECS, select
from hexlib.graph.pipeline import compile_model
from hexlib.models.vit import VitConfig, build_vision_encoder

HAS_SDK = os.path.isdir(tc.default_sdk_root())
sdk = pytest.mark.skipif(not HAS_SDK, reason="Hexagon SDK not present")

VTCM_BUDGET = 8 * 1024 * 1024


def _tiny_cfg() -> VitConfig:
    """The same tiny shape `test_models_vit.py` uses, with the encoder's REAL
    dtypes: fp16 activations and q4_0 weights. The dtypes matter more than the
    sizes here -- they are what decide which kernel each op selects and whether
    the block-quantized staging path is exercised at all."""
    return VitConfig(
        depth=2,
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        patch_size=4,
        temporal_patch_size=2,
        in_channels=3,
        spatial_merge_size=2,
        out_hidden_size=32,
        layernorm_eps=1e-6,
        rope_theta=10000.0,
        image_size=32,
        act_dtype="fp16",
        weight_dtype="q4_0",
    )


def _compiled():
    """`compile_model`, NOT `compile_graph` + `Compiled(...)` by hand.

    `compile_graph` returns the plan alone and its docstring says why that is not
    enough: "fusion creates ops that appear in no graph the caller holds, so the
    plan on its own is not executable." Pairing a plan with the PRE-fusion graph
    builds a `Compiled` that validates fine -- every tensor a step names is
    declared -- and then fails at run time, because `interpreter.run` resolves
    backends by walking `graph.ops` while `_run_op` looks them up by the PLAN
    step's kind. `matmul_epilogue` exists only after fusion, so the lookup raises
    `KeyError: 'matmul_epilogue'` on the third step. Ask for the post-fusion pair.
    """
    graph = build_vision_encoder(_tiny_cfg())
    assert not hasattr(graph, "reason"), f"graph: {getattr(graph, 'detail', graph)}"
    compiled = compile_model(
        graph, budget=VTCM_BUDGET,
        order_policy="min_peak", alloc_policy="largest_first",
    )
    assert not hasattr(compiled, "reason"), (
        f"compile: {getattr(compiled, 'detail', compiled)}"
    )
    return compiled, compiled.graph, compiled.plan


def _feeds(graph, seed=3):
    """Every graph input and every const, in the shapes the graph declares.

    Consts are RANDOM rather than zero or one. A zero weight makes every matmul
    return zeros, which agrees with any reference for any reason; a weight of one
    makes a transposed operand undetectable. Neither would fail if the DSP were
    wrong.
    """
    rng = np.random.default_rng(seed)
    feeds = {}
    for name in list(graph.inputs) + [
        t.name for t in graph.tensors.values() if t.const
    ]:
        spec = graph.tensor(name)
        feeds[name] = (rng.standard_normal(spec.shape) * 0.5).astype(np.float32)
    return feeds


def _dispatchable(plan):
    """(ops that will go to the DSP, ops that will fall back), by kind."""
    on_dsp, fallback = {}, {}
    for step in plan.steps:
        kind = step.op.kind
        if kind == "reshape":
            continue                      # a view; the interpreter needs no kernel
        try:
            name, _ = select(kind, dict(step.op.attrs))
        except (KeyError, ValueError):
            fallback[kind] = fallback.get(kind, 0) + 1
        else:
            on_dsp[name] = on_dsp.get(name, 0) + 1
    return on_dsp, fallback


def test_the_tiny_encoder_plan_is_the_shape_this_test_assumes():
    """Runs with no SDK. If the graph or the pass pipeline changes what this
    encoder compiles to, the SDK-gated test below would start measuring something
    else while still passing -- so the shape is pinned here, cheaply, where CI
    can see it."""
    _, graph, plan = _compiled()
    on_dsp, fallback = _dispatchable(plan)
    total = sum(on_dsp.values()) + sum(fallback.values())

    assert len(plan.steps) > 40, f"only {len(plan.steps)} plan steps"
    assert total > 30, f"only {total} real-work ops"
    # Both transpose variants must appear, since routing between them in one
    # graph is a thing only this test exercises.
    assert on_dsp.get("transpose", 0) > 0, "no perm(1,0,2) transpose in the plan"
    assert on_dsp.get("transpose_hd", 0) > 0, (
        "no perm(0,2,1) transpose in the plan -- the variant-routing claim below "
        "would be vacuous"
    )
    # Every kind either dispatches or is a known gap, never something else.
    known_gaps = {"matmul", "matmul_epilogue"}
    assert set(fallback) <= known_gaps, (
        f"unexpected kinds fell back to the reference: "
        f"{sorted(set(fallback) - known_gaps)}. Either a kernel regressed out of "
        f"SPECS or the graph grew an op kind nobody has looked at."
    )


@sdk
def test_the_whole_encoder_agrees_with_the_reference_with_every_kernel_on_the_dsp():
    """THE END-TO-END RUN.

    Both paths execute the same compiled plan over the same feeds. The DSP path
    routes every op with a kernel through `hexagon-sim`; the reference path uses
    the op registry throughout. The encoder's declared outputs must agree.

    The tolerance is not tight and should not be: the DSP path computes in fp16
    while the reference accumulates in the interpreter's compute dtype, and the
    error compounds across a 2-layer encoder -- every layernorm, softmax and
    rotation narrows to fp16 and feeds the next op. What this test is for is
    catching a WRONG op, not measuring the last ULP; per-op accuracy is pinned
    at the encoder's real shapes by `test_dsp_sim.py` and by each kernel's gate,
    both of which compare against a reference at a single op where the bound can
    actually be tight.

    So the assertions are: finite, right shape, and close in a relative sense
    that a genuinely wrong kernel cannot satisfy -- plus a correlation floor,
    which is the assertion that survives a rescaling and would catch an output
    that is the right size and the wrong content.
    """
    compiled, graph, plan = _compiled()
    on_dsp, fallback = _dispatchable(plan)
    feeds = _feeds(graph)

    sim = dspmod.DspSimBackend(
        sorted({os.path.basename(s.kernel_dir) for s in SPECS.values()}),
        os.environ.get("HEXLIB_SIM_WORK") or _work_dir(),
    )
    backends = dspmod.interpreter_backends(sim)

    dsp_report = interpreter.run(compiled, feeds, backends=backends)
    assert not hasattr(dsp_report, "reason"), (
        f"the DSP run failed: {getattr(dsp_report, 'reason', '')} — "
        f"{getattr(dsp_report, 'detail', '')}"
    )
    ref_report = interpreter.run(compiled, feeds)
    assert not hasattr(ref_report, "reason"), (
        f"the reference run failed: {getattr(ref_report, 'detail', ref_report)}"
    )

    # THE COVERAGE CLAIM, ASSERTED. Without this, a spec that stopped matching
    # would fall back to the reference and this test would compare the reference
    # against itself and pass.
    used = dsp_report.backend_used
    supplied = sorted(k for k, v in used.items() if v == "supplied")
    expected = sorted({SPECS[n].kind for n in on_dsp})
    assert supplied == expected, (
        f"kinds actually sent to the DSP were {supplied}, expected {expected}"
    )
    assert len(supplied) >= 6, f"only {len(supplied)} kinds ran on the DSP"

    outs = list(graph.outputs)
    assert outs, "the encoder graph declares no outputs"
    for name in outs:
        got = np.asarray(_result(dsp_report, name), dtype=np.float64)
        want = np.asarray(_result(ref_report, name), dtype=np.float64)
        assert got.shape == want.shape
        assert np.all(np.isfinite(got)), f"{name}: DSP produced non-finite values"

        denom = max(float(np.abs(want).max()), 1e-6)
        rel = float(np.abs(got - want).max()) / denom
        assert rel < 0.05, (
            f"{name}: max relative error {rel:.4f} between the DSP path and the "
            f"reference over the whole encoder"
        )
        # Survives a rescaling, unlike the bound above: a kernel returning a
        # scaled or permuted version of the right answer fails here.
        gv, wv = got.reshape(-1), want.reshape(-1)
        if gv.size > 1 and wv.std() > 0:
            corr = float(np.corrcoef(gv, wv)[0, 1])
            assert corr > 0.99, f"{name}: correlation with the reference {corr:.4f}"


def _result(report, name):
    """The named output, from whichever attribute the report carries it in."""
    for attr in ("outputs", "results", "ddr"):
        table = getattr(report, attr, None)
        if isinstance(table, dict) and name in table:
            return table[name]
    raise AssertionError(
        f"ExecReport has no output {name!r}; attributes are "
        f"{[a for a in dir(report) if not a.startswith('_')]}"
    )


def _work_dir():
    import tempfile

    return tempfile.mkdtemp(prefix="hexlib_encoder_sim_")

"""The encoder, executed with a real Hexagon kernel in it.

SDK-GATED, and the gate is narrow on purpose. These tests need hexagon-clang and
hexagon-sim, so they cannot run in CI. What they must never do is skip into a
false pass -- so the skip is decided by whether the toolchain resolves, and the
offline suite carries its own coverage of the same marshalling logic that does
not depend on the SDK.
"""
from __future__ import annotations

import dataclasses
import os
import struct

import numpy as np
import pytest

import hexlib.graph.opdefs  # noqa: F401  -- registers the op definitions
from hexlib import toolchain as tc
from hexlib.exec import interpreter
from hexlib.result import Err

KERNEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "kernels",
    "scale_fp16",
)


def _sdk_available() -> bool:
    try:
        bin_dir = tc.find_toolchain_bin(tc.default_sdk_root())
    except FileNotFoundError:
        return False
    return os.path.isfile(os.path.join(bin_dir, tc.exe(tc.COMPILER)))


sdk = pytest.mark.skipif(
    not _sdk_available(), reason="Hexagon SDK toolchain not found"
)


def test_the_kernel_dir_carries_a_runner():
    """Offline. A runner is what lets the executor dispatch to this kernel; its
    absence is why an op falls back to the reference."""
    from hexlib.exec.hexagon import has_runner

    assert has_runner(KERNEL_DIR)


def test_the_runner_is_not_the_harness():
    """Offline, and load-bearing. The harness's whole value is that it builds
    its own inputs and cannot be handed a passing answer. If the runner printed
    the gate's verdict line, a binary that merely computed could be read as one
    that checked."""
    runner = open(os.path.join(KERNEL_DIR, "runner.c"), encoding="utf-8").read()
    assert "HEXLIB_VERDICT" not in runner
    # The CALL, not the name: runner.c names hexlib_report in a comment that
    # explains why it deliberately does not use that format, and a substring
    # check on the bare name flags the explanation as the offence.
    assert "hexlib_report(" not in runner
    harness = open(os.path.join(KERNEL_DIR, "harness.c"), encoding="utf-8").read()
    assert "hexlib_report(" in harness
    assert "fopen" not in harness, (
        "the correctness harness must not read host-supplied data"
    )


@sdk
def test_hexagon_scale_matches_the_reference_bit_for_bit(tmp_path):
    """One op, dispatched to the real HVX kernel.

    The factor is a power of two, so the scale is exact in fp16 and the kernel
    and the reference must agree EXACTLY. Any difference here is a marshalling
    error -- byte order, dtype, length -- not a numerical one, which is what
    makes an exact comparison the right assertion for this shape.
    """
    from hexlib.exec.hexagon import RunnerStats, scale_backend

    stats = RunnerStats()
    backend = scale_backend(KERNEL_DIR, work_dir=str(tmp_path), stats=stats)

    rng = np.random.default_rng(7)
    x = (rng.standard_normal((4, 64)) * 3.0).astype(np.float32)
    (got,) = backend((x,), {"factor": 0.25})

    want = (x.astype(np.float16) * np.float16(0.25)).astype(np.float32)
    np.testing.assert_array_equal(got, want)
    assert stats.calls == 1
    assert stats.kernel_cycles > 0, "no cycle count came back from the runner"


@sdk
def test_a_tail_is_computed_not_dropped(tmp_path):
    """n not a multiple of the 64-lane vector. The kernel's tail path is the
    one a near-miss omits, and the encoder's own shape would never exercise it."""
    from hexlib.exec.hexagon import scale_backend

    backend = scale_backend(KERNEL_DIR, work_dir=str(tmp_path))
    x = np.arange(1, 132, dtype=np.float32)  # 131 elements = 2*64 + 3
    (got,) = backend((x,), {"factor": 0.5})
    want = (x.astype(np.float16) * np.float16(0.5)).astype(np.float32)
    np.testing.assert_array_equal(got, want)


@sdk
def test_the_encoder_runs_with_a_hexagon_kernel_and_still_matches_pytorch(tmp_path):
    """The whole point: a plan, executed, with a real kernel doing one of its ops.

    Also asserts the Hexagon result is identical to the reference-backend
    result. If the two differed, one of them would be wrong and the PyTorch
    comparison alone could not say which.
    """
    from hexlib.exec.hexagon import RunnerStats, scale_backend
    from hexlib.graph.pipeline import compile_model
    from hexlib.models.vit import build_vision_encoder
    from hexlib.tests.test_vision_oracle import (
        TINY_CFG,
        VISION_NPZ,
        _feeds_from_golden,
    )

    if not os.path.isfile(VISION_NPZ):
        pytest.skip("golden vectors are not committed")
    z = np.load(VISION_NPZ)

    cfg = dataclasses.replace(TINY_CFG, act_dtype="fp16", weight_dtype="fp16")
    graph = build_vision_encoder(cfg)
    assert not isinstance(graph, Err), graph
    model = compile_model(graph, budget=1 << 22)
    assert not isinstance(model, Err), model

    n_scale = sum(1 for op in model.graph.ops if op.kind == "scale")
    assert n_scale > 0, "no scale op in the graph, so this test proves nothing"

    stats = RunnerStats()
    backend = scale_backend(KERNEL_DIR, work_dir=str(tmp_path), stats=stats)
    feeds = _feeds_from_golden(graph, z)

    hexagon = interpreter.run(model, feeds, backends={"scale": backend})
    assert not isinstance(hexagon, Err), f"{hexagon.reason}\n{hexagon.detail}"
    assert stats.calls == n_scale, (
        f"{n_scale} scale ops but the kernel was invoked {stats.calls} times"
    )
    assert hexagon.backend_used["scale"] == "supplied"

    reference = interpreter.run(model, feeds)
    assert not isinstance(reference, Err), reference

    out = model.graph.outputs[0]
    want = z["expected_merged"]
    np.testing.assert_array_equal(hexagon.outputs[out], reference.outputs[out])
    rel = np.max(np.abs(hexagon.outputs[out] - want)) / np.max(np.abs(want))
    assert rel < 1e-3, f"relative error against PyTorch is {rel:.3e}"


@sdk
def test_a_stale_output_cannot_be_read_as_a_result(tmp_path):
    """If the runner failed to write an output, a leftover file from a previous
    call must not be returned as this call's answer."""
    from hexlib.exec.hexagon import OUT_NAME, scale_backend

    backend = scale_backend(KERNEL_DIR, work_dir=str(tmp_path))
    x = np.ones(64, dtype=np.float32)
    backend((x,), {"factor": 0.5})
    assert os.path.exists(os.path.join(str(tmp_path), OUT_NAME))

    # A second call with a different length must not be able to return the
    # first call's bytes.
    (got,) = backend((np.full(128, 2.0, dtype=np.float32),), {"factor": 0.5})
    assert got.shape == (128,)
    np.testing.assert_array_equal(got, np.ones(128, dtype=np.float32))


def test_the_input_protocol_is_what_the_runner_parses():
    """Offline. The header is int32 n then float32 factor, little-endian. If
    Python and C disagreed here every value would be wrong, so the layout is
    pinned by a test rather than by two comments that can drift apart."""
    packed = struct.pack("<if", 4100, 0.125)
    assert len(packed) == 8
    n, factor = struct.unpack("<if", packed)
    assert n == 4100
    assert factor == 0.125
    runner = open(os.path.join(KERNEL_DIR, "runner.c"), encoding="utf-8").read()
    assert 'fread(&n, sizeof(int), 1, in)' in runner
    assert 'fread(&factor, sizeof(float), 1, in)' in runner

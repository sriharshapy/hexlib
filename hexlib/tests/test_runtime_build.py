# hexlib/tests/test_runtime_build.py
"""qaic generation. SDK-gated, in the same shape as the existing SDK tests."""
import os
import pathlib

import pytest

from hexlib import toolchain as tc
from hexlib.runtime import build as rb

HAS_SDK = os.path.isdir(tc.default_sdk_root())
sdk = pytest.mark.skipif(not HAS_SDK, reason="Hexagon SDK not present")

IDL = "hexlib/runtime/idl/hexlib_iface.idl"


def test_the_idl_declares_invoke_and_not_a_queue_id():
    """A dsp_queue_id in `start` would mean we had drifted back to dspqueue,
    which has no simulator path. Asserted so the drift is caught, not noticed."""
    src = pathlib.Path(IDL).read_text()
    assert "invoke(" in src
    assert "dsp_queue_id" not in src
    assert "sequence<octet> batch" in src


def test_the_idl_hwinfo_reports_arch_and_acquired_vtcm():
    src = pathlib.Path(IDL).read_text()
    assert "rout uint32 arch" in src
    assert "rout uint64 vtcm_size" in src


def test_qaic_path_is_platform_correct():
    p = rb.qaic_path("/fake/sdk")
    assert "qaic" in p
    if os.name == "nt":
        assert p.endswith("qaic.exe") and "bin" in p
    else:
        assert "Ubuntu" in p


@sdk
def test_qaic_generates_three_files(tmp_path):
    out = rb.run_qaic(IDL, str(tmp_path))
    for f in (out.header, out.stub, out.skel):
        assert os.path.isfile(f), f
    hdr = pathlib.Path(out.header).read_text()
    assert "hexlib_iface_invoke" in hdr
    assert "hexlib_iface_hwinfo" in hdr


@sdk
def test_generated_skel_dispatches_invoke(tmp_path):
    out = rb.run_qaic(IDL, str(tmp_path))
    skel = pathlib.Path(out.skel).read_text()
    assert "hexlib_iface_invoke" in skel


@sdk
def test_missing_idl_is_an_error_not_an_empty_success(tmp_path):
    with pytest.raises(rb.RuntimeBuildError, match="not found"):
        rb.run_qaic(str(tmp_path / "nope.idl"), str(tmp_path))

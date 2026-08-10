# hexlib/tests/test_runtime_sim_build.py
import os

import pytest

from hexlib import toolchain as tc
from hexlib.runtime import build as rb

HAS_SDK = os.path.isdir(tc.default_sdk_root())
sdk = pytest.mark.skipif(not HAS_SDK, reason="Hexagon SDK not present")


def test_sim_link_extras_names_the_libraries_that_are_known_to_work():
    """Recovered from a working v75 calculator build, not reconstructed."""
    extras = rb.SIM_LINK_EXTRAS("/sdk", "/tools")
    joined = " ".join(extras)
    assert "rtld.a" in joined
    assert "hexagon_toolv19_v75" in joined
    assert "test_util.a" in joined
    assert "atomic.a" in joined
    assert "libhexagon.a" in joined and "v75" in joined and "G0" in joined


def test_sim_link_flags_include_force_dynamic_and_G0():
    flags = rb.SIM_LINK_FLAGS
    assert "-G0" in flags
    assert any("--force-dynamic" in f for f in flags)
    assert any("ISDB_TRUSTED_FLAG=2" in f for f in flags)


@sdk
def test_skel_library_builds(tmp_path):
    lib = rb.build_skel_lib(["scale_fp16"], str(tmp_path))
    assert os.path.isfile(lib)
    assert os.path.getsize(lib) > 0
    # An ar archive, not merely a path that happens to exist: the magic bytes
    # a mock or a `touch` would not reproduce.
    with open(lib, "rb") as f:
        assert f.read(8) == b"!<arch>\n"


@sdk
def test_sim_qexe_builds(tmp_path):
    rb.build_skel_lib(["scale_fp16"], str(tmp_path))
    elf = rb.build_sim_qexe(str(tmp_path))
    assert os.path.isfile(elf)
    assert os.path.getsize(elf) > 0
    # A real Hexagon ELF, not merely a path: the ELF magic plus EM_HEXAGON
    # (0xa4 in e_machine, little-endian half at offset 18) -- a stub file
    # written by a gutted build_sim_qexe would not carry either.
    with open(elf, "rb") as f:
        header = f.read(20)
    assert header[:4] == b"\x7fELF"
    e_machine = header[18] | (header[19] << 8)
    assert e_machine == 0xA4

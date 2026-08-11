# hexlib/tests/test_runtime_device_build.py
"""Task 10 -- STAGE 2 GATE: cross-compile hexlib_run (Android aarch64) and
libhexlib_skel.so (Hexagon device shared object). NEITHER IS EVER RUN HERE --
no device is available -- so every SDK-gated test below asserts the built
ARTIFACT and its machine type, never merely that a function returned a path
string that happens to exist.
"""
import os
import struct

import pytest

from hexlib import toolchain as tc
from hexlib.runtime import build as rb

HAS_SDK = os.path.isdir(tc.default_sdk_root())
sdk = pytest.mark.skipif(not HAS_SDK, reason="Hexagon SDK not present")


def test_ndk_is_discovered_inside_the_sdk_never_vendored():
    p = tc.ndk_root("/fake/sdk")
    assert "android-ndk-r25c" in p
    assert p.startswith("/fake/sdk") or p.startswith("\\fake")


def test_android_api_is_pinned():
    assert tc.ANDROID_API == 33


def test_ndk_clang_name_is_api_specific_not_a_generic_alias():
    """A regression to some other clang alias (e.g. a bare `clang` or a
    different API level) would still be "a file that exists" on a real SDK,
    so the SDK-gated existence test below cannot by itself catch a wrong
    name -- this pins the exact filename, offline."""
    p = rb.ndk_clang("/fake/sdk")
    base = os.path.basename(p)
    assert base.startswith(f"aarch64-linux-android{tc.ANDROID_API}-clang")
    assert "android-ndk-r25c" in p


def test_ndk_clang_uses_the_cmd_wrapper_on_windows():
    """The bare-name driver next to the `.cmd` wrapper is a Bourne shell
    script (confirmed with `file` against the real NDK) that a plain
    subprocess call cannot execute on native Windows. A regression back to
    the bare name would still look plausible in a path string, so this pins
    the suffix directly rather than trusting the SDK-gated existence check to
    notice (os.path.isfile is also true of the unusable bare-name file)."""
    p = rb.ndk_clang("/fake/sdk")
    if os.name == "nt":
        assert p.endswith(".cmd"), p
    else:
        assert not p.endswith(".cmd"), p


def test_device_skel_link_flags_are_the_dll_recipe_not_the_sim_one():
    """Recovered from the SDK's OWN defines_hexagon_1_9.min DLL_LD_FLAGS, a
    DIFFERENT recipe from SIM_SO_LINK_FLAGS -- the distinguishing content is
    the five --wrap= flags (PD heap interposition a real FastRPC-loaded skel
    needs and the QuRT-hosted sim .so does not). Asserting only "-shared" and
    "-fpic" would also pass for SIM_SO_LINK_FLAGS and would not catch a
    build_device_binary that accidentally reused the sim recipe for the
    device skel -- a real risk since both produce a Hexagon .so from the same
    kind of object files."""
    flags = rb.DEVICE_SKEL_LINK_FLAGS
    assert "-shared" in flags
    assert "-fpic" in flags
    for wrapped in ("malloc", "calloc", "free", "realloc", "memalign"):
        assert f"-Wl,--wrap={wrapped}" in flags, f"missing --wrap={wrapped}"
    joined = " ".join(flags)
    assert "--force-dynamic" not in joined


@sdk
def test_ndk_clang_exists():
    assert os.path.isfile(rb.ndk_clang(tc.default_sdk_root()))


@sdk
def test_device_binary_and_skel_so_build(tmp_path):
    exe = rb.build_device_binary(str(tmp_path))
    so = os.path.join(str(tmp_path), "libhexlib_skel.so")
    assert os.path.isfile(exe)
    assert os.path.isfile(so)
    # FAIL CLOSED: a build step that exits 0 without writing real content
    # (e.g. `open(path, "w").close()`) must not pass as "it builds".
    assert os.path.getsize(exe) > 4096, "hexlib_run is implausibly small"
    assert os.path.getsize(so) > 4096, "libhexlib_skel.so is implausibly small"


def _elf_header(path):
    with open(path, "rb") as f:
        head = f.read(20)
    assert head[:4] == b"\x7fELF", f"{path} has no ELF magic"
    ei_class = head[4]
    e_machine = struct.unpack("<H", head[18:20])[0]
    return ei_class, e_machine


@sdk
def test_the_device_binary_is_aarch64(tmp_path):
    exe = rb.build_device_binary(str(tmp_path))
    ei_class, e_machine = _elf_header(exe)
    assert ei_class == 2, "expected ELFCLASS64 (64-bit)"
    assert e_machine == 0xB7, f"expected EM_AARCH64 (0xB7), got {e_machine:#x}"


@sdk
def test_the_skel_so_is_hexagon(tmp_path):
    rb.build_device_binary(str(tmp_path))
    so = os.path.join(str(tmp_path), "libhexlib_skel.so")
    _, e_machine = _elf_header(so)
    assert e_machine == 164, f"expected EM_QDSP6 (164), got {e_machine}"


@sdk
def test_the_stub_not_the_skel_is_linked_into_the_aarch64_binary(tmp_path):
    """THE CRUX OF THIS TASK, asserted directly rather than only relying on
    "it linked without a duplicate-symbol error". `hexlib_run` must carry the
    qaic-marshalling call (`remote_handle64_invoke`, only ever referenced by
    the generated STUB, never by skel.c/skel_dispatch.c) and a host-only
    symbol (`hexlib_drv_init`, driver.c) -- and it must NOT carry
    `hexlib_bufs_register`, a symbol that exists ONLY in skel_bufs.c, which is
    never one of this binary's sources. A build that accidentally compiled
    skel.c/skel_bufs.c into hexlib_run instead of (or alongside) the stub
    would, in the ordinary case, simply fail to link on a duplicate-symbol
    error against the stub's identical names (see hexlib_iface_stub.c) -- but
    a build that dropped the stub and silently substituted the skel's plain
    functions (the exact simulator-shaped mistake this task exists to avoid)
    would link JUST FINE and produce a binary that can never marshal. This
    test catches that mistake by content, not by "did the linker exit 0"."""
    exe = rb.build_device_binary(str(tmp_path))
    with open(exe, "rb") as f:
        blob = f.read()
    assert b"remote_handle64_invoke" in blob, (
        "hexlib_run does not reference remote_handle64_invoke -- the qaic "
        "stub was not linked in, so this binary can never marshal a real "
        "FastRPC call"
    )
    assert b"hexlib_drv_init" in blob, "driver.c was not linked into hexlib_run"
    assert b"hexlib_bufs_register" not in blob, (
        "hexlib_run contains skel_bufs.c's hexlib_bufs_register -- the "
        "DSP-side skel was linked into the aarch64 host binary, which is "
        "backwards (see the module docstring's stub/skel split)"
    )


@sdk
def test_the_skel_so_contains_the_skel_not_the_host(tmp_path):
    """The mirror image of the test above, from the Hexagon side.
    `libhexlib_skel.so` must carry `hexlib_bufs_register` (skel_bufs.c) and
    must NOT carry `hexlib_drv_init` (driver.c, host-only) -- a build that
    accidentally bundled the aarch64 host sources into the device skel
    (nonsensical machine-code-wise, but a real risk if out_dir/object-name
    bookkeeping were wrong) would still produce *a* Hexagon .so, which the
    machine-type test above cannot by itself catch."""
    rb.build_device_binary(str(tmp_path))
    so = os.path.join(str(tmp_path), "libhexlib_skel.so")
    with open(so, "rb") as f:
        blob = f.read()
    assert b"hexlib_bufs_register" in blob, "skel_bufs.c was not linked into libhexlib_skel.so"
    assert b"hexlib_drv_init" not in blob, (
        "libhexlib_skel.so contains driver.c's hexlib_drv_init -- the "
        "aarch64 host code was linked into the DSP-side skel"
    )

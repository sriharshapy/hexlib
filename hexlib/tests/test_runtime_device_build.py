# hexlib/tests/test_runtime_device_build.py
"""Task 10 -- STAGE 2 GATE: cross-compile hexlib_run (Android aarch64) and
libhexlib_iface_skel.so (Hexagon device shared object). NEITHER IS EVER RUN HERE --
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


def test_the_aarch64_host_link_treats_warnings_as_errors():
    """The Hexagon side gets -Wall -Werror from tc.HVX_CFLAGS; this aarch64
    link builds its own command line, so it needs them spelled out here or the
    host half of the wire (host/*.c, which assembles the batch blob and the
    buffer table by hand) keeps compiling with its diagnostics discarded --
    every caller of tc.run decides success from `rc != 0` alone.

    Source-text rather than behavioural because building hexlib_run needs the
    SDK and the NDK, and this must fail on a machine with neither. Comment
    lines are excluded for the same reason
    test_runtime_sim_build.py's -fpic check excludes them: the flag appears in
    this function's own explanatory comment, and a `re.search` over the whole
    body would keep passing after the real flag was deleted.
    """
    import inspect
    import re

    live = [
        ln for ln in inspect.getsource(rb.build_device_binary).splitlines()
        if not ln.strip().startswith("#")
    ]
    joined = "\n".join(live)
    assert re.search(r'cmd\s*=\s*\[clang[^\]]*"-Werror"', joined), (
        "build_device_binary no longer passes -Werror to the aarch64 clang"
    )
    assert '"-Wall"' in joined


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
def test_device_skel_so_actually_carries_the_symbolic_dynamic_flag(tmp_path):
    """FIX 2 (coordinator review): the test above only inspects the
    `DEVICE_SKEL_LINK_FLAGS` constant and would still pass if
    `_build_device_skel_so` stopped splicing it into the real link command
    (e.g. if it started building the flags list itself instead of using this
    one). This closes that gap by reading `-Wl,-Bsymbolic`'s effect back OUT
    OF THE ARTIFACT: a linker that honoured `-Bsymbolic` records a `DT_SYMBOLIC`
    (0x10) tag in the `.so`'s own `PT_DYNAMIC` segment -- confirmed against a
    real build with the Hexagon toolchain's own `hexagon-readelf -d` before
    writing this parser.

    `--wrap=malloc/calloc/free/realloc/memalign` is DELIBERATELY NOT CHECKED
    THIS WAY. Checked with `hexagon-nm -u` against a real build: none of
    skel.c/skel_bufs.c/skel_vtcm.c/skel_dispatch.c or the generated entries
    reference malloc/calloc/free/realloc/memalign at all (Hexagon's `--wrap`
    only rewrites a call site that actually exists), so there is NO ARTIFACT
    EVIDENCE the flags could leave in THIS SPECIFIC BUILD even when they are
    genuinely present on the link line and doing exactly what they are meant
    to. Asserting anything artifact-shaped here would be inventing a proxy,
    which the review this test responds to explicitly said not to do.
    """
    rb.build_device_binary(str(tmp_path))
    so = os.path.join(str(tmp_path), rb.device_skel_so_name())
    with open(so, "rb") as f:
        data = f.read()
    DT_SYMBOLIC = 0x10
    assert DT_SYMBOLIC in _elf32_dynamic_tags(data), (
        "the skel .so has no DT_SYMBOLIC dynamic tag -- -Wl,-Bsymbolic "
        "from DEVICE_SKEL_LINK_FLAGS did not actually reach the link"
    )


def _elf32_dynamic_tags(data):
    """Every DT_* tag present in an ELF32 file's PT_DYNAMIC segment (Hexagon
    is ELFCLASS32 -- confirmed by reading ei_class, byte 4, == 1 -- so this
    uses Elf32_Phdr/Elf32_Dyn layouts, not the 64-bit ones the aarch64 helper
    below needs)."""
    assert data[:4] == b"\x7fELF"
    assert data[4] == 1, "expected ELFCLASS32 for a Hexagon ELF"
    e_phoff = struct.unpack_from("<I", data, 0x1C)[0]
    e_phentsize = struct.unpack_from("<H", data, 0x2A)[0]
    e_phnum = struct.unpack_from("<H", data, 0x2C)[0]
    PT_DYNAMIC = 2
    tags = set()
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type, p_offset, _, _, p_filesz, _, _, _ = struct.unpack_from("<IIIIIIII", data, off)
        if p_type != PT_DYNAMIC:
            continue
        pos, end = p_offset, p_offset + p_filesz
        while pos + 8 <= end:
            d_tag, _ = struct.unpack_from("<iI", data, pos)
            pos += 8
            if d_tag == 0:  # DT_NULL
                break
            tags.add(d_tag)
    return tags


def _elf64_note_android_api(data):
    """The NDK API level actually baked into an aarch64 ELF's
    `.note.android.ident` PT_NOTE segment (every NDK-clang-built Android ELF
    carries one; verified against the real hexlib_run with the NDK's own
    llvm-readelf before writing this parser). Returns None if no such note is
    present -- callers must not treat that as API 0."""
    assert data[:4] == b"\x7fELF"
    assert data[4] == 2, "expected ELFCLASS64 for an aarch64 ELF"
    e_phoff = struct.unpack_from("<Q", data, 0x20)[0]
    e_phentsize = struct.unpack_from("<H", data, 0x36)[0]
    e_phnum = struct.unpack_from("<H", data, 0x38)[0]
    PT_NOTE = 4
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type, _ = struct.unpack_from("<II", data, off)
        if p_type != PT_NOTE:
            continue
        p_offset, _, _, p_filesz = struct.unpack_from("<QQQQ", data, off + 8)
        pos, end = p_offset, p_offset + p_filesz
        while pos < end:
            namesz, descsz, ntype = struct.unpack_from("<III", data, pos)
            pos += 12
            name = data[pos:pos + namesz]
            pos += (namesz + 3) & ~3
            desc = data[pos:pos + descsz]
            pos += (descsz + 3) & ~3
            if name.rstrip(b"\x00") == b"Android" and ntype == 1:  # NT_ANDROID_TYPE_IDENT
                return struct.unpack_from("<I", desc, 0)[0]
    return None


@sdk
def test_ndk_clang_exists():
    assert os.path.isfile(rb.ndk_clang(tc.default_sdk_root()))


@sdk
def test_the_built_binary_actually_embeds_the_pinned_api_level(tmp_path):
    """FIX 2 (coordinator review): `test_ndk_clang_name_is_api_specific_not_a_
    generic_alias` and `test_android_api_is_pinned` only check constants and a
    path string in isolation -- both would still pass if `build_device_binary`
    quietly called a DIFFERENT clang (any other API level, or a generic
    `aarch64-linux-android-clang` some NDKs also ship) as long as ndk_clang()
    itself still returned the pinned name. This closes that gap by reading it
    back OUT OF THE ARTIFACT: every NDK-clang-built Android ELF embeds its
    target API level as the first 4 bytes of `.note.android.ident`'s
    NT_ANDROID_TYPE_IDENT description (confirmed against a real build with the
    NDK's own llvm-readelf before this test was written) -- so this is real
    evidence the pinned-API clang was the one that actually ran, not a second
    assertion of the same constant."""
    exe = rb.build_device_binary(str(tmp_path))
    with open(exe, "rb") as f:
        data = f.read()
    api = _elf64_note_android_api(data)
    assert api is not None, "hexlib_run has no .note.android.ident -- not an NDK-clang build?"
    assert api == tc.ANDROID_API, f"binary embeds API {api}, expected {tc.ANDROID_API}"


@sdk
def test_device_binary_and_skel_so_build(tmp_path):
    exe = rb.build_device_binary(str(tmp_path))
    so = os.path.join(str(tmp_path), rb.device_skel_so_name())
    assert os.path.isfile(exe)
    assert os.path.isfile(so)
    # FAIL CLOSED: a build step that exits 0 without writing real content
    # (e.g. `open(path, "w").close()`) must not pass as "it builds".
    assert os.path.getsize(exe) > 4096, "hexlib_run is implausibly small"
    assert os.path.getsize(so) > 4096, "the skel .so is implausibly small"


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
    so = os.path.join(str(tmp_path), rb.device_skel_so_name())
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
    `libhexlib_iface_skel.so` must carry `hexlib_bufs_register` (skel_bufs.c) and
    must NOT carry `hexlib_drv_init` (driver.c, host-only) -- a build that
    accidentally bundled the aarch64 host sources into the device skel
    (nonsensical machine-code-wise, but a real risk if out_dir/object-name
    bookkeeping were wrong) would still produce *a* Hexagon .so, which the
    machine-type test above cannot by itself catch."""
    rb.build_device_binary(str(tmp_path))
    so = os.path.join(str(tmp_path), rb.device_skel_so_name())
    with open(so, "rb") as f:
        blob = f.read()
    assert b"hexlib_bufs_register" in blob, "skel_bufs.c was not linked into the skel .so"
    assert b"hexlib_drv_init" not in blob, (
        "the skel .so contains driver.c's hexlib_drv_init -- the "
        "aarch64 host code was linked into the DSP-side skel"
    )

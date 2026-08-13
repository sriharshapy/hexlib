# hexlib/tests/test_device_skel_so_name.py
"""The skel's filename, bound to the one FastRPC will actually dlopen.

THE DEFECT THIS EXISTS FOR WAS FOUND ON SILICON AND COULD NOT HAVE BEEN FOUND
ANYWHERE ELSE. `_build_device_skel_so` linked `libhexlib_skel.so`. The URI qaic
generates into `hexlib_iface.h` -- `hexlib_iface_URI`, which `session.c:163`
passes to `remote_handle64_open` -- names the library after the IDL, so the
device looks for `libhexlib_iface_skel.so`. `hexlib_iface_open` returned
rc -2147482618 until the file was pushed under both names by hand.

WHY NO SIMULATOR TEST COULD CATCH IT: stage 1's build LINKS the skel directly
(`simhost.c` calls `hexlib_iface_open/_start/_invoke/...` as plain C functions,
bound by the linker straight to `skel.c`), so the filename never participates in
that path at all. The name only matters when something loads it BY NAME, and
only a device does.

So the name is now DERIVED from the IDL stem in one place, and every other copy
is checked against that derivation here rather than trusted.
"""
import pathlib
import re

from hexlib.runtime.build import HEXLIB_IDL_STEM, device_skel_so_name

REPO = pathlib.Path(__file__).resolve().parents[2]


def test_the_name_follows_qaics_own_convention():
    """qaic names its outputs `<stem>.h` / `<stem>_stub.c` / `<stem>_skel.c`
    from the IDL, and the URI it writes names `lib<stem>_skel.so`. This pins the
    derivation itself, so a future change to the helper has to be deliberate."""
    assert device_skel_so_name() == "libhexlib_iface_skel.so"
    assert device_skel_so_name("some_other_iface") == "libsome_other_iface_skel.so"


def test_the_idl_that_stem_comes_from_actually_exists():
    """A derivation from a stem that names no IDL would be a convention with
    nothing behind it -- and qaic's URI comes from the real file's name."""
    idl = REPO / "hexlib" / "runtime" / "idl" / f"{HEXLIB_IDL_STEM}.idl"
    assert idl.is_file(), f"{idl} does not exist, so the derived name is a guess"


def test_the_linker_is_told_the_derived_name_and_not_a_literal():
    """`build.py` must call the helper. A hardcoded string here would compile
    and link perfectly and fail only on a device, which is the whole history of
    this bug."""
    src = (REPO / "hexlib" / "runtime" / "build.py").read_text(encoding="utf-8")
    assert "device_skel_so_name()" in src
    assert 'os.path.join(out_dir, "libhexlib_skel.so")' not in src, (
        "build.py still links explicitly to the old name"
    )


def test_the_on_device_module_spells_the_same_name():
    """`test_on_device.py` runs on the QDC runner without hexlib installed, so
    it cannot import the helper and has to carry a literal. That literal is read
    back here and compared -- which is the only thing that keeps the two in step,
    since nothing else imports both."""
    src = (REPO / "hexlib" / "device" / "qdc" / "test_on_device.py").read_text(
        encoding="utf-8"
    )
    m = re.search(r'^SKEL_SO\s*=\s*"([^"]+)"', src, re.M)
    assert m, "test_on_device.py declares no SKEL_SO literal"
    assert m.group(1) == device_skel_so_name(), (
        f"the on-device module pushes {m.group(1)!r} but the linker produces "
        f"{device_skel_so_name()!r}; the device would dlopen a file that is "
        f"not there, exactly as it did before this was bound"
    )


def test_nothing_still_names_the_old_library():
    """The old name in a *comment* is fine and deliberate -- several of them
    explain the bug. What must not survive is a live reference in code that
    packs, pushes or links the file."""
    offenders = []
    for rel in ("hexlib/cli.py", "hexlib/device/qdc/test_on_device.py",
                "hexlib/runtime/build.py"):
        for i, line in enumerate((REPO / rel).read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("*"):
                continue
            if '"libhexlib_skel.so"' in line or "'libhexlib_skel.so'" in line:
                offenders.append(f"{rel}:{i}: {stripped}")
    assert not offenders, (
        "these still name the pre-fix library as a string literal:\n"
        + "\n".join(offenders)
    )

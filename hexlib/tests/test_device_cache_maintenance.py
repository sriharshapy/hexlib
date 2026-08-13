# hexlib/tests/test_device_cache_maintenance.py
"""The DSP must write its cache back, and the check has to read the ARTIFACT.

WHAT THIS EXISTS FOR, MEASURED ON SM8650 (Pineapple, SM8650, 2026-08-13).
FastRPC keeps the two caches coherent for anything passed as an invoke
ARGUMENT. hexlib's data buffers are not arguments -- they are mapped out of
band through `fastrpc_mmap` and named on the wire only by fd, so that no
address ever crosses between the processors. FastRPC therefore does not know
they were written, and before `hexlib_bufs_flush` existed, nothing wrote them
back. Three symptoms, one cause:

  * `hexlib_run --self-test`: 3859 of 4100 fp16 values not bit-exact, against
    a simulator that gives exactly 0 error. AFTER the flush: `PASS (4100
    values, bit-exact)`.
  * `--coherency-check`: `COHERENCY sentinel_unchanged`, exit 6.
    AFTER: `sentinel_overwritten`, RC=0.
  * the 49-op encoder through `--batch`: 179,124 arena bytes changed but
    `merger.out` -- the LAST op's 1024 bytes -- came back all zero.
    AFTER: max relative error 1.1319e-03, correlation 1.000000, which is the
    SAME figure the simulator produces.

WHY THESE TESTS READ THE LINKED .so AND NOT THE SOURCE. `skel_bufs.c` carries a
host-compiler fallback that #defines the QuRT cache API away, because
`test_genentry_entry_probe.py` compiles that file with gcc and there is no
`qurt_memory.h` off-target. A fallback whose whole job is to do nothing is
exactly the thing that could silently become what ships. A source grep cannot
tell which branch a real build took. The linked artifact can: if the Hexagon
build compiled the stub, `qurt_mem_cache_clean` would be DEFINED (or absent)
rather than left UNDEFINED for QuRT to bind at load.
"""
import os
import pathlib
import subprocess

import pytest

from hexlib import toolchain as tc
from hexlib.runtime import build as rb
from hexlib.runtime import wire

HAS_SDK = os.path.isdir(tc.default_sdk_root())
sdk = pytest.mark.skipif(not HAS_SDK, reason="Hexagon SDK not present")

REPO = pathlib.Path(__file__).resolve().parents[2]
SKEL = REPO / "hexlib" / "runtime" / "skel"


def test_a_cache_failure_has_its_own_status_on_both_sides():
    """Not folded into ERR_INTERNAL, because the consequence is specific and
    misleading: every op ran, and what the host reads back may be STALE rather
    than wrong. Nothing on the host can tell those apart without a code."""
    assert wire.STATUS["ERR_CACHE"] == 15
    header = (SKEL / "hexlib_dsp.h").read_text(encoding="utf-8")
    assert "HEXLIB_DSP_ERR_CACHE = 15," in header
    assert 'case HEXLIB_DSP_ERR_CACHE:          return "ERR_CACHE";' in header


def test_both_cache_directions_are_called_and_not_just_one():
    """Flushing without invalidating works for exactly ONE invoke per session
    and then silently computes on data this DSP cached during the previous one.
    That is a worse bug than the one the flush fixes, because it needs two runs
    to appear -- so the presence of BOTH calls is pinned, not just the flush."""
    src = (SKEL / "skel_dispatch.c").read_text(encoding="utf-8")
    live = "\n".join(
        ln for ln in src.splitlines()
        if not ln.strip().startswith("*") and not ln.strip().startswith("/*")
    )
    assert "hexlib_bufs_invalidate(ctx)" in live, (
        "nothing invalidates before the ops read host-written data"
    )
    assert "hexlib_bufs_flush(ctx)" in live, (
        "nothing flushes after the ops write their results"
    )


def test_the_flush_runs_even_when_the_batch_failed():
    """An op that died halfway still wrote whatever it wrote. Leaving those
    lines in cache makes the wreckage invisible to anyone debugging from the
    host, which is the position this whole file exists to get out of."""
    src = (SKEL / "skel_dispatch.c").read_text(encoding="utf-8")
    flush = src.index("cache_rc = hexlib_bufs_flush(ctx);")
    tail = src[flush:]
    assert "batch_status == HEXLIB_DSP_OK" in tail, (
        "the flush must not overwrite a real failure status"
    )
    # It must not be inside the `if (batch_status == OK)` guard itself.
    before = src[:flush].rstrip().splitlines()[-1].strip()
    assert not before.startswith("if "), (
        "the flush is guarded by the batch status, so a failed batch leaves "
        "its writes in cache"
    )


@sdk
def test_the_linked_skel_really_calls_qurts_cache_api(tmp_path):
    """THE ONE THAT CANNOT BE SATISFIED BY THE HOST STUB.

    `qurt_mem_cache_clean` must appear as an UNDEFINED symbol in the linked
    device .so -- QuRT binds it at load time inside the PD. If the Hexagon
    build had compiled `skel_bufs.c`'s host fallback instead, the symbol would
    be locally defined and this fails. That is the whole point: a `#if
    defined(__hexagon__)` guard is a source-level claim, and this is the
    artifact-level check of it."""
    rb.build_device_binary(str(tmp_path))
    so = os.path.join(str(tmp_path), rb.device_skel_so_name())
    nm = os.path.join(
        tc.find_toolchain_bin(tc.default_sdk_root()),
        "hexagon-nm" + (".exe" if os.name == "nt" else ""),
    )
    out = subprocess.run([nm, "-u", so], capture_output=True, text=True).stdout
    assert any("qurt_mem_cache_clean" in ln for ln in out.splitlines()), (
        "qurt_mem_cache_clean is not an undefined symbol in the linked skel -- "
        "the host fallback in skel_bufs.c was compiled into the DEVICE build, "
        "so nothing writes the DSP's cache back and the host reads stale data"
    )

    defined = subprocess.run([nm, so], capture_output=True, text=True).stdout
    for sym in ("hexlib_bufs_flush", "hexlib_bufs_invalidate"):
        assert any(sym in ln for ln in defined.splitlines()), f"{sym} not linked"

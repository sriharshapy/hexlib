"""VTCM acquisition under contention, and the status that survives the RPC boundary.

TWO THINGS THE SIMULATOR CANNOT TELL US, so they are tested here instead.

The skel used to ask for the part's ENTIRE VTCM with `min_vtcm_size = 0`, which
`HAP_compute_res.h:544-546` defines as "the size is an absolute requirement".
On a shared CDSP one other client holding a single 4 KB page therefore made
`HAP_compute_res_acquire` fail, `hexlib_iface_start` fail, and every mode exit at
session open. The simulator cannot reproduce it -- nothing else there holds VTCM,
which is exactly why stage 1 was green while this was live -- so what is checked
here is the SHAPE of the request (both queried sizes used, the floor derived from
`avail` rather than a constant), plus the arithmetic of the status encoding,
COMPILED AND RUN rather than pattern-matched.

The source assertions use csource so they are comment-blind and function-scoped:
a claim satisfied by a comment is the defect this project keeps rediscovering.
"""
import re
import shutil
import subprocess

import pytest

from hexlib.tests import csource

# Same discovery order as test_session_arch_decode.py and
# test_coherency_lane_classification.py, so a machine without a host compiler
# skips the compiled checks uniformly rather than in three different ways.
HOST_CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
needs_cc = pytest.mark.skipif(
    HOST_CC is None,
    reason=(
        "no host C compiler found (tried: cc, gcc, clang); the BEHAVIOURAL "
        "status-encoding checks need one. The source-shape checks above do not "
        "and still run."
    ),
)

VTCM_C = "hexlib/runtime/skel/skel_vtcm.c"
SKEL_C = "hexlib/runtime/skel/skel.c"
DSP_H = "hexlib/runtime/skel/hexlib_dsp.h"


def _src(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _alloc_body():
    return csource.function_body(_src(VTCM_C), "hexlib_vtcm_alloc")


# --------------------------------------------------------------------------
# The request shape
# --------------------------------------------------------------------------

def test_the_query_asks_for_the_available_size_not_only_the_total():
    """`avail_block_size` is the 4th out-parameter and it used to be 0.

    Fails if anyone reverts to `HAP_compute_res_query_VTCM(0, &x, 0, 0, 0)`,
    which is what made the total the only number the skel knew.
    """
    body = _alloc_body()
    m = re.search(r"HAP_compute_res_query_VTCM\s*\(([^()]*)\)", body)
    assert m, "hexlib_vtcm_alloc must query VTCM sizes"
    args = [a.strip() for a in m.group(1).split(",")]
    assert len(args) == 5, f"expected 5 arguments, got {args}"
    assert args[3] != "0", (
        "the 4th argument is avail_block_size -- the SDK's 'largest contiguous "
        "memory chunk available'. Passing 0 discards it, which is what made the "
        "skel demand the part total as an absolute requirement"
    )
    assert args[3].startswith("&"), f"avail must be an out-parameter, got {args[3]}"


def test_the_min_vtcm_size_floor_is_not_zero_and_not_a_constant():
    """`min_vtcm_size = 0` means "absolute requirement" -- the bug.

    A hardcoded floor would also violate this file's governing rule that the
    size comes from the runtime, so the floor must be a variable.
    """
    body = _alloc_body()
    m = re.search(r"HAP_compute_res_attr_set_vtcm_param_v2\s*\(([^()]*)\)", body)
    assert m, "must set the v2 VTCM params"
    args = [a.strip() for a in m.group(1).split(",")]
    assert len(args) == 4, f"expected 4 arguments, got {args}"
    floor = args[3]
    assert floor != "0", (
        "min_vtcm_size = 0 is 'the size is an absolute requirement' "
        "(HAP_compute_res.h:544-546) -- any contention then fails session open"
    )
    assert not re.fullmatch(r"[0-9]+[uU]?|0[xX][0-9a-fA-F]+[uU]?", floor), (
        f"the floor must come from the runtime, not the constant {floor!r}"
    )


def test_a_fully_contended_partition_is_refused_with_its_own_status():
    """avail == 0 is distinguishable from a query failure."""
    body = _alloc_body()
    zero_check = re.search(r"if\s*\(\s*\w*avail\w*\s*==\s*0\s*\)", body)
    assert zero_check, "a fully contended partition (avail == 0) must be refused"
    guarded = csource.block_from(body, zero_check.start())
    assert "HEXLIB_DSP_ERR_VTCM_TOO_SMALL" in guarded, (
        "refusing with a specific status is what lets a device operator tell "
        "contention from a load failure"
    )


def test_start_does_not_flatten_the_vtcm_status_into_a_bare_failure():
    """`return AEE_EFAILED` threw away which of 14 statuses occurred."""
    body = csource.function_body(_src(SKEL_C), "hexlib_iface_start")
    assert "HEXLIB_AEE_FROM_STATUS" in body, (
        "start() must carry the specific status out; a bare AEE_EFAILED is the "
        "result the host already prints for a dozen unrelated causes"
    )
    assert not re.search(r"return\s+AEE_EFAILED\s*;", body), (
        "the flattening return is what this fixes"
    )


# --------------------------------------------------------------------------
# The status encoding, compiled and driven with real values
# --------------------------------------------------------------------------

_PROBE = r"""
#include <stdio.h>
#include <stdint.h>
%s
int main(void) {
    /* every status in the enum, plus the tag boundaries */
    for (int s = 1; s <= 14; s++) {
        int r = HEXLIB_AEE_FROM_STATUS(s);
        printf("%%d %%u %%d %%d\n", s, (unsigned) r,
               HEXLIB_AEE_IS_STATUS(r) ? 1 : 0, HEXLIB_AEE_STATUS(r));
    }
    /* things that must NOT decode as our status */
    unsigned others[] = {0u, 0x80000008u, 0x8FA10000u, 0x0FA00000u};
    for (int i = 0; i < 4; i++) {
        printf("other %%u %%d\n", others[i], HEXLIB_AEE_IS_STATUS(others[i]) ? 1 : 0);
    }
    return 0;
}
"""


def _macros():
    """The four macros, sliced out of the real header."""
    src = _src(DSP_H)
    lines = [
        ln for ln in src.splitlines()
        if re.match(r"\s*#define\s+HEXLIB_AEE_", ln)
    ]
    assert len(lines) == 4, f"expected 4 HEXLIB_AEE_ macros, found {len(lines)}"
    return "\n".join(lines)


@needs_cc
def test_every_status_round_trips_through_the_aee_encoding(tmp_path):
    """COMPILED AND RUN, not pattern-matched.

    A source assertion cannot see that a mask and a shift disagree -- which is
    precisely how the arch cross-check shipped comparing 75 against 0x8c75.
    """
    src = tmp_path / "probe.c"
    src.write_text(_PROBE % _macros(), encoding="utf-8")
    exe = tmp_path / "probe.exe"
    subprocess.run([HOST_CC, str(src), "-o", str(exe)], check=True,
                   capture_output=True)
    out = subprocess.run([str(exe)], check=True, capture_output=True,
                         text=True).stdout

    seen = {}
    for line in out.strip().splitlines():
        parts = line.split()
        if parts[0] == "other":
            assert parts[2] == "0", (
                f"{parts[1]} must not decode as a hexlib status -- it would make "
                "an RPC-layer failure look like a VTCM one"
            )
            continue
        s, raw, is_status, decoded = (int(p) for p in parts)
        assert is_status == 1, f"status {s} must be recognised by its own tag"
        assert decoded == s, f"status {s} decoded as {decoded}"
        assert raw != 0, "the encoded value must be nonzero so every caller still fails"
        seen[s] = raw

    assert len(seen) == 14
    assert len(set(seen.values())) == 14, "each status must encode distinctly"


@needs_cc
def test_the_encoding_is_never_zero_so_a_failure_is_never_read_as_success(tmp_path):
    """AEE_SUCCESS is 0. An encoding that produced 0 for some status would turn
    a VTCM failure into a successful session -- absence read as success, in the
    place it would cost a device job."""
    src = tmp_path / "probe.c"
    src.write_text(_PROBE % _macros(), encoding="utf-8")
    exe = tmp_path / "probe.exe"
    subprocess.run([HOST_CC, str(src), "-o", str(exe)], check=True,
                   capture_output=True)
    out = subprocess.run([str(exe)], check=True, capture_output=True,
                         text=True).stdout
    for line in out.strip().splitlines():
        parts = line.split()
        if parts[0] == "other":
            continue
        assert int(parts[1]) != 0, f"status {parts[0]} encoded to 0 (== AEE_SUCCESS)"

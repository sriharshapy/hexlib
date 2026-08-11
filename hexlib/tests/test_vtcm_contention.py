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


def _query_out_params(body):
    """The names (without `&`) of HAP_compute_res_query_VTCM's total and avail
    out-parameters, read off the real call. Derived rather than hardcoded: the
    fix for the "absolute requirement" bug split one local into two, and a test
    that names them dictates code layout instead of checking behaviour (see
    test_skel_vtcm_source.py's note on the same rename)."""
    m = re.search(r"HAP_compute_res_query_VTCM\s*\(([^()]*)\)", body)
    assert m, "hexlib_vtcm_alloc must query VTCM sizes"
    args = [a.strip() for a in m.group(1).split(",")]
    assert len(args) == 5, f"expected 5 arguments, got {args}"
    # Signature (HAP_compute_res.h:1087-1106): (application_id,
    # total_block_size, total_block_layout, avail_block_size,
    # avail_block_layout).
    return args[1].lstrip("&").strip(), args[3].lstrip("&").strip()


def test_the_min_vtcm_size_floor_is_below_the_request_and_comes_from_the_query():
    """`min_vtcm_size = 0` means "absolute requirement" -- the bug.

    THIS PINNED THE BUG BY ARGUMENT SPELLING, AND THE BUG'S SEMANTIC EQUIVALENT
    PASSED. The old form asserted only that the floor was not the literal `0`
    and not a numeric constant. `min_vtcm_size = vtcm_total` satisfies both and
    re-demands the WHOLE partition: it is the original defect restored, since a
    floor equal to the request means any contention at all fails
    HAP_compute_res_acquire, which fails hexlib_iface_start, which exits every
    mode at session open. The identifier differing from `0` was never the
    requirement.

    THE REQUIREMENT, STATED AS TWO RELATIONS INSTEAD OF ONE SPELLING. Ask for
    the total the runtime reported, and accept down to the AVAIL the runtime
    reported -- so the floor is (a) derived from the query's availability
    out-parameter, and (b) a different quantity from the request, which is what
    makes it a floor at all. `avail <= total` is the SDK's own guarantee about
    those two out-parameters ("largest contiguous memory chunk available" vs the
    partition total), so pinning WHICH out-parameter each argument is pins the
    inequality without this test having to know either number.

    Name-agnostic in both directions: both names are read off the query call, so
    a rename that keeps the semantics passes and a swap that keeps the names
    fails."""
    body = _alloc_body()
    total, avail = _query_out_params(body)
    assert total != avail, (
        f"the total and available sizes must be two distinct out-parameters -- "
        f"HAP_compute_res_query_VTCM was passed {total!r} for both, so there is "
        f"no availability figure for the floor to come from"
    )

    m = re.search(r"HAP_compute_res_attr_set_vtcm_param_v2\s*\(([^()]*)\)", body)
    assert m, "must set the v2 VTCM params"
    args = [a.strip() for a in m.group(1).split(",")]
    assert len(args) == 4, f"expected 4 arguments, got {args}"
    requested, floor = args[1], args[3]

    assert requested == total, (
        f"the request (total_block_size) must be the total the runtime just "
        f"reported ({total!r}), not {requested!r} -- asking for `avail` directly "
        f"caps the session at a value that can go stale between query and "
        f"acquire"
    )
    assert floor == avail, (
        f"min_vtcm_size must be the runtime's own AVAILABLE size ({avail!r}), "
        f"not {floor!r}. 0 is 'the size is an absolute requirement' "
        f"(HAP_compute_res.h:544-546); {total!r} is the same thing spelled "
        f"differently, since a floor equal to the request refuses any "
        f"contention at all; a constant would violate this file's governing "
        f"rule that the size comes from the runtime"
    )
    assert floor != requested, (
        "the floor must be BELOW the request, not equal to it -- a floor equal "
        "to the request is the absolute-requirement bug with a variable name on "
        "it"
    )


def test_the_reservation_is_actually_acquired_and_its_results_stored():
    """A REQUEST SHAPE IS NOT AN ACQUISITION. Every check above reads arguments
    off two `HAP_compute_res_attr_set_*` calls, and attribute setters acquire
    nothing: replacing the rest of hexlib_vtcm_alloc with `return
    HEXLIB_DSP_OK;` -- so the session never holds VTCM and every kernel gets a
    null base with size 0 -- left this whole file green. The floor being right
    is only interesting if the request built from it is submitted, checked, and
    its results recorded on the session.

    Everything here is derived from the calls themselves rather than named, for
    the same reason as the floor check above."""
    body = _alloc_body()
    m = re.search(r"HAP_compute_res_attr_set_vtcm_param_v2\s*\(([^()]*)\)", body)
    assert m, "must set the v2 VTCM params"
    attr = [a.strip() for a in m.group(1).split(",")][0].lstrip("&").strip()

    acquire = re.search(
        rf"(\w+)\s*=\s*HAP_compute_res_acquire\s*\(\s*&\s*{re.escape(attr)}\b", body
    )
    assert acquire, (
        f"hexlib_vtcm_alloc must submit the attributes it just built "
        f"(&{attr}) to HAP_compute_res_acquire and keep the result -- a "
        f"discarded reservation context cannot be released or re-acquired later"
    )
    rctx = acquire.group(1)
    assert re.search(rf"if\s*\(\s*!\s*{re.escape(rctx)}\s*\)", body), (
        f"a failed acquire returns 0, so `{rctx}` must be checked for it -- "
        f"HAP_compute_res_acquire burns its full timeout before failing and "
        f"then every kernel would run with no VTCM at all"
    )

    ptr_query = re.search(
        r"HAP_compute_res_attr_get_vtcm_ptr_v2\s*\(([^()]*)\)", body
    )
    assert ptr_query, "the acquired VTCM's base and size must be read back"
    ptr_args = [a.strip().lstrip("&").strip() for a in ptr_query.group(1).split(",")]
    assert len(ptr_args) == 3, f"expected 3 arguments, got {ptr_args}"
    got_ptr, got_size = ptr_args[1], ptr_args[2]

    for field, var, why in (
        ("vtcm_base", got_ptr, "no kernel can use VTCM it has no pointer to"),
        ("vtcm_size", got_size, "hwinfo reports this number to the host, and "
                                "the M1 allocator's budget is it"),
        ("vtcm_rctx", rctx, "without the reservation context the dispatcher "
                            "cannot release VTCM at an op boundary, which is "
                            "what a competing session waits on"),
    ):
        assert re.search(
            rf"ctx->{field}\s*=\s*(?:\([^;)]*\)\s*)?{re.escape(var)}\s*;", body
        ), (
            f"ctx->{field} must be set from `{var}` -- {why}"
        )


def test_a_fully_contended_partition_is_refused_with_its_own_status():
    """avail == 0 is distinguishable from a query failure."""
    body = _alloc_body()
    zero_check = re.search(r"if\s*\(\s*\w*avail\w*\s*==\s*0\s*\)", body)
    assert zero_check, "a fully contended partition (avail == 0) must be refused"
    guarded = csource.block_from(body, zero_check.start())
    # A RETURN, not the token. `"X" in guarded` was satisfiable by a FARF
    # naming the constant while the function carried on to acquire a
    # reservation it had just proven impossible -- the file-wide shape of
    # defect this whole area was reviewed for. (csource blanks literals now, so
    # the FARF vector is closed at the source; requiring the return closes the
    # "assign it to an unused local" one too.)
    assert re.search(r"return\s+HEXLIB_DSP_ERR_VTCM_TOO_SMALL\s*;", guarded), (
        "refusing with a specific status is what lets a device operator tell "
        "contention from a load failure -- and it must be RETURNED from inside "
        "this branch, not merely named in it"
    )


# --------------------------------------------------------------------------
# The simulator/device asymmetry, pinned so it stays a decision
# --------------------------------------------------------------------------

SIMHOST_C = "hexlib/runtime/simhost/simhost.c"
SESSION_C = "hexlib/runtime/host/session.c"


def _start_args(path, _fn=None):
    """The argument list of the hexlib_iface_start CALL in `path`.

    Whole-file, comment-blanked: the call sites are in different functions on
    the two sides and the argument list contains a cast (`(uint64) MAX_BLOB`),
    so this anchors on the `);` that ends the statement rather than on the
    first close paren.
    """
    src = csource.code_only(_src(path))
    m = re.search(r"hexlib_iface_start\s*\((.*?)\)\s*;", src, re.DOTALL)
    assert m, f"{path} must call hexlib_iface_start"
    return [a.strip() for a in m.group(1).split(",")]


def test_the_simulator_requests_hmx_and_the_device_host_does_not():
    """THE GATE EXERCISES A DIFFERENT ACQUISITION PATH THAN PRODUCTION.

    `skel_vtcm.c` only calls `HAP_compute_res_attr_set_hmx_param` when
    `ctx->n_hmx > 0`, and n_hmx is whatever start() was passed. simhost passes
    1, the device host passes 0 -- so stage 1 acquires VTCM *with* an HMX
    request and a device never has. That is a real difference in the one call
    most likely to fail first on unfamiliar silicon, and it was pinned by
    nothing.

    This test does not judge which is right. It fails if either side changes
    silently, so the asymmetry stays a recorded decision. The trigger to
    revisit is the first HMX kernel, which is what skel_vtcm.c also says.
    """
    sim = _start_args(SIMHOST_C, "main")
    assert len(sim) == 5, f"unexpected simhost start signature: {sim}"
    assert sim[3] == "1", (
        f"simhost passes n_hmx={sim[3]}; this test and skel.c's hwinfo comment "
        "both record it as 1. If you changed it, the asymmetry note needs updating"
    )

    dev = _start_args(SESSION_C, "hexlib_open")
    n_hmx = dev[3]
    assert re.fullmatch(r"/\*\s*n_hmx\s*\*/\s*0|0", n_hmx), (
        f"the device host passes n_hmx={n_hmx!r}; it was 0, meaning the device "
        "never requests HMX. Changing this changes compute-res acquisition on "
        "hardware that has never run this code -- see skel.c's hwinfo comment"
    )


def test_hwinfo_does_not_claim_the_echoed_fields_are_dsp_facts():
    """The IDL said "what the DSP says about itself" for five fields when it is
    true of one. A doc asserting a guarantee the code does not deliver counts
    the same as a code defect here, because these files are the handoff record."""
    idl = _src("hexlib/runtime/idl/hexlib_iface.idl")
    assert "ONLY vtcm_size IS A DSP FACT" in idl, (
        "the hwinfo block must state which outputs are queried and which are "
        "compile-time constants or host echoes"
    )
    body = csource.function_body(_src(SKEL_C), "hexlib_iface_hwinfo")
    assert re.search(r"\*n_threads\s*=\s*1\s*;", body), (
        "n_threads is hardcoded; if that changed, the IDL note must change too"
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

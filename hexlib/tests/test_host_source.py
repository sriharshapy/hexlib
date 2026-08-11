# hexlib/tests/test_host_source.py
"""The CPU-side driver. Source assertions -- it cannot be RUN without a
device, which is exactly why stage 2 exists as a separate gate: so stage 3
spends minutes on one unknown rather than five.

TIGHTENED PAST THE DRAFT. A first draft of these ten checked for bare
substrings anywhere in a file -- which a comment, a dead branch, or a FARF log
line naming the right constant would also satisfy. Every earlier task in this
plan had the same problem and needed the same fix (see
test_skel_bufs_source.py, test_skel_vtcm_source.py), so these are
function-scoped wherever the underlying claim is about ONE function's
behaviour, and check actual `return`s / call ORDER rather than mere presence.
"""
import pathlib
import re

import pytest

from hexlib.tests.csource import block_from as _block_from
from hexlib.tests.csource import function_body as _function_body

H = pathlib.Path("hexlib/runtime/host")


@pytest.fixture(scope="module")
def driver():
    return (H / "driver.c").read_text()


@pytest.fixture(scope="module")
def session():
    return (H / "session.c").read_text()


@pytest.fixture(scope="module")
def buffers():
    return (H / "buffers.c").read_text()


@pytest.fixture(scope="module")
def main():
    return (H / "main.c").read_text()


def _macro_body(src, name):
    """Slice an object/function-like `#define` by following backslash line
    continuations. `_function_body`'s brace-counting does not apply to a
    macro definition (its own braces are a `do { ... } while (0)` wrapper,
    not the boundary we want), so this is a narrowly-scoped sibling rather
    than a reuse -- there is exactly one macro these tests need to isolate."""
    m = re.search(rf"#define\s+{re.escape(name)}\b", src)
    assert m, f"could not find #define {name} in the source"
    lines = src[m.start():].splitlines()
    out = []
    for line in lines:
        out.append(line)
        if not line.rstrip().endswith("\\"):
            break
    return "\n".join(out)


def test_libcdsprpc_is_dlopened_not_linked(driver):
    """A missing driver becomes a readable message instead of a loader
    failure with no output -- this is why a well-built capability probe works
    the first time it runs on real hardware. Scoped to hexlib_drv_init():
    dlopen() and the candidate path must both live in the function that
    actually loads the driver, not merely somewhere in the file (e.g. a
    comment mentioning both). AND a NULL handle -- every candidate path
    failed -- must actually fail hexlib_drv_init from inside its own check,
    not merely be logged: a build that calls dlopen() and ignores a NULL
    result would otherwise satisfy the presence checks above and still crash
    the first time a dlsym() runs against it."""
    body = _function_body(driver, "hexlib_drv_init")
    assert "dlopen(" in body
    assert "libcdsprpc.so" in body

    null_check = re.search(r"handle\s*==\s*NULL", body)
    assert null_check, "a failed dlopen() must be checked, not assumed to succeed"
    null_block = _block_from(body, null_check.end())
    assert re.search(r"return\s+-1\s*;", null_block), (
        "a NULL driver handle must actually fail hexlib_drv_init, not just "
        "be logged"
    )


def test_every_symbol_is_resolved_by_name_and_checked(driver):
    """Each required symbol must be the subject of an actual HEXLIB_DLSYM(...)
    call inside hexlib_drv_init -- not merely named somewhere in the file,
    which a stale comment or a typedef alone would also satisfy."""
    body = _function_body(driver, "hexlib_drv_init")
    for sym in (
        "rpcmem_alloc", "rpcmem_free", "rpcmem_to_fd", "fastrpc_mmap",
        "remote_handle64_open", "remote_handle64_invoke",
        "remote_handle_control", "remote_session_control",
    ):
        assert re.search(rf"HEXLIB_DLSYM\([^;]*\b{re.escape(sym)}\b", body), sym
    assert "dlsym(" in driver


def test_a_missing_symbol_is_an_error_not_a_null_call(driver):
    """The dlsym-and-check macro itself, not just the word `dlerror` anywhere
    in the file, must report via dlerror() AND actually fail (`return -1`) on
    a required symbol -- logging and continuing would reintroduce exactly the
    null-call-later bug this exists to prevent."""
    macro = _macro_body(driver, "HEXLIB_DLSYM")
    assert "dlerror()" in macro
    assert "return -1;" in macro
    # The failure path must be reachable from the `required` branch, not
    # dead code after an unconditional return.
    assert re.search(r"if\s*\(required\)", macro)


def test_cdsp_domain_three_and_unsigned_pd(session):
    """CDSP only. ADSP is a v73 part with UNSIGNED_PD_SUPPORT = 0, so
    targeting it would silently be a different measurement on different
    hardware. Scoped to hexlib_open(): the domain check and the unsigned-PD
    request must both be real control flow in the function that opens a
    session, not just mentioned in the file somewhere."""
    body = _function_body(session, "hexlib_open")
    assert "CDSP_DOMAIN_ID" in body
    reject = re.search(r"domain\s*!=\s*CDSP_DOMAIN_ID", body)
    assert reject, "hexlib_open must refuse any domain that isn't CDSP_DOMAIN_ID"
    # A real refusal, not a comment: the `if` block guarded by the comparison
    # above -- and nothing outside it, which a later unrelated `return -1;`
    # (e.g. from the driver-init check further down) could otherwise satisfy
    # by accident -- must itself contain the error return.
    reject_block = _block_from(body, reject.end())
    assert re.search(r"return\s+-1\s*;", reject_block), (
        "the domain check must actually refuse inside its own if-block, not "
        "merely be followed eventually by some other return"
    )

    unsigned_call = re.search(r"enable_unsigned_pd\s*\(", body)
    assert unsigned_call, "hexlib_open must request an unsigned PD"
    open_call = re.search(r"\bhexlib_iface_open\s*\(", body)
    assert open_call, "hexlib_open must actually open the qaic handle"
    assert unsigned_call.start() < open_call.start(), (
        "unsigned PD must be requested BEFORE the handle is opened -- "
        "afterwards the PD already exists"
    )

    enable_body = _function_body(session, "enable_unsigned_pd")
    assert "DSPRPC_CONTROL_UNSIGNED_MODULE" in enable_body


def test_the_uri_is_built_not_hardcoded_with_a_domain(session):
    """The URI must be assembled from hexlib_iface_URI (qaic-generated) and
    CDSP_DOMAIN (<remote.h>'s own "&_dom=cdsp" macro) as adjacent string
    literals -- never spelled out as a literal "&_dom=cdsp" string, which
    would silently stop tracking either constant if it ever changed.

    THIS CHECKS CONSTRUCTION *FLOWING INTO USE*, not mere co-occurrence in
    the file: it is not enough for `hexlib_iface_URI CDSP_DOMAIN` to appear
    somewhere in session.c if the variable it builds is dead code and
    hexlib_iface_open() is actually called with something else entirely.
    So this captures the constructed variable's name and asserts THAT name
    is what is passed as hexlib_iface_open's first argument, in hexlib_open
    itself, after the construction."""
    body = _function_body(session, "hexlib_open")
    construct = re.search(
        r"(\w+)\s*\[\]\s*=\s*hexlib_iface_URI\s+CDSP_DOMAIN\b", body
    )
    assert construct, (
        "hexlib_open must build the URI from hexlib_iface_URI and CDSP_DOMAIN"
    )
    assert '"&_dom=cdsp"' not in session

    var = construct.group(1)
    open_call = re.search(rf"\bhexlib_iface_open\s*\(\s*{re.escape(var)}\s*,", body)
    assert open_call, (
        f"the constructed URI variable ({var!r}) must be passed as the "
        "first argument to hexlib_iface_open -- not merely constructed and "
        "left unused while something else is opened"
    )
    assert construct.start() < open_call.start(), (
        "the URI must be constructed before it is passed to hexlib_iface_open"
    )


def test_arch_is_queried_from_the_driver_not_assumed(session):
    """hexlib_query_caps must actually issue the ARCH_VER / DSPRPC_GET_DSP_INFO
    query (function-scoped, not just present in the file), and hexlib_open
    must cross-check that value against what the skel itself reports, failing
    on disagreement rather than trusting either side alone."""
    caps_body = _function_body(session, "hexlib_query_caps")
    assert "ARCH_VER" in caps_body
    assert "DSPRPC_GET_DSP_INFO" in caps_body

    open_body = _function_body(session, "hexlib_open")
    mismatch = re.search(r"arch\s*!=\s*caps\.arch_ver", open_body)
    assert mismatch, "hexlib_open must cross-check driver arch against skel arch"
    mismatch_block = _block_from(open_body, mismatch.end())
    assert re.search(r"return\s+-1\s*;", mismatch_block), (
        "an arch mismatch must actually fail hexlib_open from inside its own "
        "if-block, not just be logged"
    )


def test_no_literal_request_ids(session):
    """An earlier probe in this project hardcoded DSPRPC_GET_DSP_INFO as 11 by
    counting an enum in a doc comment; the real value is 2. A wrong request id
    does not fail loudly -- it queries something else. So every request id
    must come from <remote.h>'s own enums, never a bare number passed
    straight to the control APIs.

    NEGATIVE HALF (absence of the bad pattern) paired with a POSITIVE HALF
    (presence of the right one): a purely negative check would pass
    vacuously if the real `remote_handle_control`/`remote_session_control`
    calls were deleted outright, which is exactly the kind of gutted
    implementation these tests exist to catch."""
    assert "#include <remote.h>" in session
    assert not re.search(r"=\s*11\b", session)
    # Nothing may pass a literal digit as the request id argument itself --
    # that would dodge the "= 11" check above while still hardcoding a
    # different id the same way.
    assert not re.search(r"remote_(handle|session)_control\s*\(\s*\d", session)

    caps_body = _function_body(session, "hexlib_query_caps")
    assert re.search(
        r"hexlib_remote_handle_control\s*\(\s*DSPRPC_GET_DSP_INFO\s*,", caps_body
    ), "hexlib_query_caps must actually call remote_handle_control with DSPRPC_GET_DSP_INFO"

    enable_body = _function_body(session, "enable_unsigned_pd")
    assert re.search(
        r"hexlib_remote_session_control\s*\(\s*DSPRPC_CONTROL_UNSIGNED_MODULE\s*,",
        enable_body,
    ), (
        "enable_unsigned_pd must actually call remote_session_control with "
        "DSPRPC_CONTROL_UNSIGNED_MODULE"
    )


def test_buffers_use_rpcmem_and_fastrpc_mmap(buffers):
    """Scoped to hexlib_alloc(): the real allocation sequence must be
    rpcmem_alloc, then rpcmem_to_fd, then fastrpc_mmap -- each an actual call
    in the function that allocates a buffer, in that order, not merely
    present somewhere in the file (e.g. in hexlib_free's teardown calls,
    which mention fastrpc_mmap's sibling but not in this order)."""
    body = _function_body(buffers, "hexlib_alloc")
    i_alloc = body.index("hexlib_rpcmem_alloc(")
    i_fd = body.index("hexlib_rpcmem_to_fd(")
    i_mmap = body.index("hexlib_fastrpc_mmap(")
    assert i_alloc < i_fd < i_mmap


def test_the_host_never_puts_an_address_on_the_wire(buffers):
    """hexlib_buf_to_desc -- the one place a hexlib_buf_desc is filled in from
    this side -- must zero `base` itself, first (right after the memset, not
    merely somewhere before the struct is used), and nothing in the file may
    derive `base` from the host pointer (`buf->ptr`/`ptr`)."""
    body = _function_body(buffers, "hexlib_buf_to_desc")
    assert re.search(r"d->base\s*=\s*0", body) or re.search(r"\bbase\s*=\s*0", body)
    memset_end = body.index(";", body.index("memset(")) + 1
    base_clear = re.search(r"\bbase\s*=\s*0\s*;", body)
    assert base_clear, "base must be explicitly cleared, not left to memset alone"
    assert base_clear.start() < body.index("d->size", memset_end), (
        "base must be cleared before the other fields are filled in"
    )
    assert not re.search(r"base\s*=[^;]*\bptr\b", buffers), (
        "base must never be derived from a host pointer anywhere in this file"
    )


def test_absence_of_a_response_is_a_failure(main):
    """response_is_valid() must check HEXLIB_BATCH_MAGIC and refuse (return a
    falsy value) both when the response is absent/short AND when the magic is
    wrong -- and the --batch path must not even attempt to write the output
    file until AFTER that check has passed, so an invalid response can never
    leave a stale or garbage file behind."""
    valid_body = _function_body(main, "response_is_valid")
    assert "HEXLIB_BATCH_MAGIC" in valid_body
    assert valid_body.count("return 0;") >= 2, (
        "both the too-short case and the bad-magic case must return falsy"
    )

    batch_body = _function_body(main, "run_batch_file")
    check_call = re.search(r"response_is_valid\s*\(", batch_body)
    write_call = re.search(r"fopen\s*\(\s*out_path", batch_body)
    assert check_call and write_call
    assert check_call.start() < write_call.start(), (
        "the output file must never be opened before the response is "
        "confirmed valid"
    )


# ==============================================================================
# --unmapped, --self-test's printed cycles_total, and --coherency-check.
#
# NONE of these three can be exercised by actually running hexlib_run (no
# device here) -- see test_runtime_device_build.py's own "NEITHER ARTIFACT IS
# EVER RUN HERE" note. So, like every other test in this file, these check
# SOURCE STRUCTURE: real call order, real branches, real string literals --
# never merely "the flag's name appears somewhere in the file", which a stale
# comment or a dead branch would also satisfy.
# ==============================================================================


def test_unmapped_alloc_skips_only_the_dsp_registration_call(main):
    """alloc_maybe_unmapped() must still run the ordinary CPU-side
    rpcmem_alloc -> rpcmem_to_fd -> fastrpc_mmap sequence in full ("the host
    allocates its rpcmem buffer and gets its fd as usual") -- only the
    DSP-side hexlib_iface_mmap() registration (hexlib_bufs_register's table,
    skel_bufs.c) is withheld when skip_dsp_register is true. Scoped to the
    function itself and to each branch of its own if/else, not merely
    "hexlib_iface_mmap is absent somewhere in the file", which the sibling
    branch satisfying it would also make trivially true."""
    body = _function_body(main, "alloc_maybe_unmapped")
    i_alloc = body.index("hexlib_rpcmem_alloc(")
    i_fd = body.index("hexlib_rpcmem_to_fd(")
    i_map = body.index("hexlib_fastrpc_mmap(")
    assert i_alloc < i_fd < i_map, (
        "the CPU-side allocation sequence must run in the same order "
        "hexlib_alloc() (buffers.c) uses, unconditionally"
    )

    skip_if = re.search(r"if\s*\(\s*skip_dsp_register\s*\)", body)
    assert skip_if, "alloc_maybe_unmapped must branch on skip_dsp_register"
    skip_block = _block_from(body, skip_if.end())
    assert "hexlib_iface_mmap(" not in skip_block, (
        "the skip_dsp_register branch must NOT call hexlib_iface_mmap -- "
        "withholding exactly that call is the whole point of --unmapped"
    )

    else_pos = body.index("else", skip_if.end())
    else_block = _block_from(body, else_pos)
    assert "hexlib_iface_mmap(" in else_block, (
        "the ordinary (mapped) branch must still register the fd with the "
        "skel, exactly like hexlib_alloc() does"
    )


def test_run_self_test_unmapped_path_uses_the_unmapped_allocator(main):
    """run_self_test(unmapped): when the flag is set, BOTH self-test buffers
    must go through alloc_maybe_unmapped(..., 1)/free_maybe_unmapped(..., 1)
    -- not the ordinary hexlib_alloc()/hexlib_free() -- so the DSP's
    hexlib_bufs_map() table lookup (skel_bufs.c) genuinely has nothing to
    find for either fd."""
    body = _function_body(main, "run_self_test")
    assert "alloc_maybe_unmapped(ctx, &bx, nbytes, 1)" in body
    assert "alloc_maybe_unmapped(ctx, &by, nbytes, 1)" in body
    assert "free_maybe_unmapped(ctx, bx, 1)" in body
    assert "free_maybe_unmapped(ctx, by, 1)" in body


def test_self_test_prints_cycles_total_after_the_existing_pass_line(main):
    """The response header's cycles_total (skel_dispatch.c's PCYCLE bracket
    around the kernel call) must be printed AFTER, never instead of, the
    existing 'PASS (%d values, bit-exact)' line -- so the exact success
    string test_on_device.py's `test_scale_fp16_runs_on_the_dsp_and_is_
    correct` already asserts on stays byte-for-byte intact, and the new
    cycles line is strictly additive."""
    body = _function_body(main, "run_self_test")
    pass_idx = body.index("PASS (%d values, bit-exact)")
    cycles_idx = body.index("cycles_total=%llu", pass_idx)
    assert pass_idx < cycles_idx
    assert "full_hdr.cycles_total" in body


def test_self_test_flag_parsing_routes_unmapped_and_coherency_correctly(main):
    """main()'s --self-test branch must recognize both --unmapped and
    --coherency-check past argv[1], route --coherency-check to
    run_coherency_check(), and thread the --unmapped flag straight into
    run_self_test(unmapped) -- not merely mention both flag strings
    somewhere in the function, which a comment or an unreachable branch
    would also satisfy."""
    body = _function_body(main, "main")
    self_test_pos = body.index('"--self-test"')
    self_test_block = _block_from(body, self_test_pos)

    assert '"--unmapped"' in self_test_block
    assert '"--coherency-check"' in self_test_block
    assert re.search(r"run_coherency_check\s*\(\s*\)", self_test_block)
    assert re.search(r"run_self_test\s*\(\s*unmapped\s*\)", self_test_block)

    coherency_if = re.search(r"if\s*\(\s*coherency\s*\)", self_test_block)
    assert coherency_if, "--coherency-check must be checked as its own branch"
    coherency_block = _block_from(self_test_block, coherency_if.end())
    assert re.search(r"run_coherency_check\s*\(\s*\)", coherency_block), (
        "the coherency branch must actually call run_coherency_check(), not "
        "merely check the flag and fall through"
    )


def test_usage_mentions_the_new_self_test_modifiers(main):
    body = _function_body(main, "usage")
    assert "--unmapped" in body
    assert "--coherency-check" in body


def test_build_scale_batch_factor_is_call_site_specific(main):
    """run_self_test must build its batch with SELF_TEST_FACTOR (0.125f, a
    power of two, exact in fp16) and run_coherency_check must use
    COHERENCY_FACTOR (0.0f) -- never the other's constant, since a nonzero
    factor in the coherency check would let a wrong result be blamed on
    kernel arithmetic instead of ruling that out entirely."""
    self_test_body = _function_body(main, "run_self_test")
    coherency_body = _function_body(main, "run_coherency_check")
    assert re.search(r"build_scale_batch\([^)]*SELF_TEST_FACTOR", self_test_body)
    assert re.search(r"build_scale_batch\([^)]*COHERENCY_FACTOR", coherency_body)
    assert "SELF_TEST_FACTOR" not in coherency_body, (
        "the coherency check must never fall back to the self-test's own "
        "nonzero factor"
    )


def test_coherency_check_writes_the_sentinel_before_invoking(main):
    """The sentinel must be written into the OUTPUT buffer strictly before
    hexlib_invoke() -- writing it afterwards would prove nothing about
    whether the DSP's own write reached the host."""
    body = _function_body(main, "run_coherency_check")
    sentinel_idx = body.index("COHERENCY_SENTINEL")
    invoke_idx = body.index("hexlib_invoke(")
    assert sentinel_idx < invoke_idx


def test_coherency_check_reads_the_sentinel_only_after_both_statuses_are_ok(main):
    """The sentinel read-back (and both printed verdict lines) must live
    strictly inside the branch reached only once the batch-level status AND
    the op's own result status are both confirmed HEXLIB_DSP_OK -- reading it
    any earlier would make a marshalling failure indistinguishable from a
    coherency one, exactly the confusion this check exists to resolve."""
    body = _function_body(main, "run_coherency_check")
    op_ok_check = re.search(r"result->status\s*!=\s*HEXLIB_DSP_OK", body)
    assert op_ok_check, "must check the op's own status, not merely the batch-level one"
    else_pos = body.index("else", op_ok_check.end())
    success_block = _block_from(body, else_pos)

    assert "memcmp(&yr[i]" in success_block, (
        "the sentinel must only be read back once both statuses are "
        "confirmed OK"
    )
    cycles_idx = success_block.index("cycles_total=%llu")
    overwritten_idx = success_block.index('"COHERENCY sentinel_overwritten\\n"')
    unchanged_idx = success_block.index('"COHERENCY sentinel_unchanged\\n"')
    assert cycles_idx < overwritten_idx
    assert cycles_idx < unchanged_idx, (
        "cycles_total must be printed before either COHERENCY verdict line "
        "-- it is the signal that tells a dispatch bug (0 cycles) apart from "
        "a genuine coherency miss (>0 cycles), and both must be visible "
        "together regardless of which branch runs"
    )


def test_coherency_miss_has_its_own_distinct_exit_code(main):
    """A coherency miss (or a dispatch bug -- see the function's own header
    comment on why cycles_total, not this exit code, is what tells the two
    apart) must exit with something other than 0-5, which are all already
    claimed by other outcomes."""
    assert re.search(r"HEXLIB_EXIT_COHERENCY_MISS\s*=\s*6", main)
    body = _function_body(main, "run_coherency_check")
    assert "exit_code = HEXLIB_EXIT_COHERENCY_MISS;" in body


def test_coherency_check_documents_its_own_scope_limits(main):
    """Design doc §6.1 (corrected 2026-08-11): the table that makes
    cycles_total load-bearing covers ONLY the DSP-write -> host-read
    direction, for scale_fp16's own write pattern -- not the reverse
    direction, and not every kernel. That caveat must live in this file's
    own comments, not only in the on-device test's docstring, or a future
    reader of just this file could believe a pass here is a general
    coherency proof."""
    assert "DSP-write" in main and "host-read" in main
    assert "host-write" in main and "DSP-read" in main
    assert "kernel-independent" in main.lower()

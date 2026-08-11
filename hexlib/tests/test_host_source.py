# hexlib/tests/test_host_source.py
"""The CPU-side driver. Source assertions -- it cannot be RUN without a
device, which is exactly why stage 2 exists as a separate gate: so stage 3
spends minutes on one unknown rather than five.

TIGHTENED TWICE. A first draft of these checked for bare substrings anywhere
in a file -- which a comment, a dead branch, or a FARF log line naming the
right constant would also satisfy. Every earlier task in this plan had the
same problem and needed the same fix (see test_skel_bufs_source.py,
test_skel_vtcm_source.py), so these became function-scoped wherever the
underlying claim is about ONE function's behaviour, checking actual `return`s
and call ORDER rather than mere presence.

THE SECOND TIGHTENING, AND WHY IT WAS NEEDED. Function scope alone was not
enough, because the checks still ran against comment-BEARING text. Three
mutations proved it:

  * reverting hexlib_decode_bcd_arch (session.c) to the shipped bug
    (`return arch_ver;`) with the arithmetic left in a comment INSIDE the body
    still matched the `>> 4` / `* 10` / `& 0x0f` regexes below;
  * reverting hexlib_classify_coherency_lane (main.c) to its `bits == 0x0000u`
    bug, old code left in a body comment, left this whole file at 22 passed;
  * `handle = dlopen(...)` -> `handle = NULL;` passed, because `"dlopen(" in
    body` was satisfied by driver.c's own dlopen-failed error format string;
    same shape for `"dlsym(" in driver`.

So every fixture here is COMMENT-BLANKED (`csource.code_only`), every slice
inherits that, and presence checks that used to be bare tokens are now call-
or assignment-shaped. The single test that legitimately inspects COMMENTS --
test_coherency_check_documents_its_own_scope_limits, whose whole claim is that
a caveat is written down for a human reader -- takes the `main_comments`
fixture instead and says so.
"""
import pathlib
import re

import pytest

from hexlib.tests.csource import block_from as _block_from
from hexlib.tests.csource import code_only as _code_only
from hexlib.tests.csource import function_body as _function_body

H = pathlib.Path("hexlib/runtime/host")


@pytest.fixture(scope="module")
def driver():
    return _code_only((H / "driver.c").read_text())


@pytest.fixture(scope="module")
def session():
    return _code_only((H / "session.c").read_text())


@pytest.fixture(scope="module")
def buffers():
    return _code_only((H / "buffers.c").read_text())


@pytest.fixture(scope="module")
def main():
    return _code_only((H / "main.c").read_text())


@pytest.fixture(scope="module")
def main_comments():
    """main.c WITH its comments, for the one test whose subject IS a comment
    (test_coherency_check_documents_its_own_scope_limits). Every other check
    in this file must use the `main` fixture above -- see the module
    docstring."""
    return (H / "main.c").read_text()


def _macro_body(src, name):
    """Slice an object/function-like `#define` by following backslash line
    continuations. `_function_body`'s brace-counting does not apply to a
    macro definition (its own braces are a `do { ... } while (0)` wrapper,
    not the boundary we want), so this is a narrowly-scoped sibling rather
    than a reuse -- there is exactly one macro these tests need to isolate.

    Comment-blanked like everything else here: `csource.code_only` preserves
    length, and a blanked `/* ... */` inside a macro leaves any trailing
    backslash continuation exactly where it was, so the line walk is
    unaffected."""
    src = _code_only(src)
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
    the first time a dlsym() runs against it.

    THE HANDLE MUST COME FROM THE CALL. `"dlopen(" in body` was satisfied by
    driver.c's own `"hexlib: dlopen(%s) failed: %s\\n"` format string, three
    lines below the real call -- so `handle = dlopen(candidates[i], RTLD_NOW);`
    could be replaced outright with `handle = NULL;` and this test still
    passed, with the readable-message property it exists to protect gone and
    the loader never consulted at all."""
    body = _function_body(driver, "hexlib_drv_init")
    assert re.search(r"\bhandle\s*=\s*dlopen\s*\(", body), (
        "the driver handle must be ASSIGNED from a real dlopen() call -- a "
        "mention of dlopen in an error message is not loading anything"
    )
    assert '"libcdsprpc.so"' in body, (
        "the candidate path must be a real string literal in the loading "
        "function, not merely named in prose"
    )

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
    which a stale comment or a typedef alone would also satisfy.

    AND THE MACRO MUST RESOLVE BY NAME, THROUGH dlsym, INTO THE POINTER.
    `"dlsym(" in driver` was whole-file and was satisfied by the macro's own
    `"hexlib: dlsym(%s) failed: %s\\n"` error format string, so the real
    `(pfn) = (__typeof__(pfn)) dlsym(handle, #symbol);` could be replaced with
    anything at all -- including `(pfn) = NULL;`, which would make every
    symbol below "resolve" and then null-call later, the exact bug the macro
    exists to prevent."""
    body = _function_body(driver, "hexlib_drv_init")
    for sym in (
        "rpcmem_alloc", "rpcmem_free", "rpcmem_to_fd", "fastrpc_mmap",
        "remote_handle64_open", "remote_handle64_invoke",
        "remote_handle_control", "remote_session_control",
    ):
        assert re.search(rf"HEXLIB_DLSYM\([^;]*\b{re.escape(sym)}\b", body), sym

    macro = _macro_body(driver, "HEXLIB_DLSYM")
    assert re.search(r"\(\s*pfn\s*\)\s*=[^;]*\bdlsym\s*\(", macro), (
        "HEXLIB_DLSYM must assign the function pointer from an actual "
        "dlsym() call -- naming dlsym in its own failure message is not "
        "resolving anything"
    )
    assert re.search(r"\bdlsym\s*\(\s*handle\s*,\s*#\s*symbol\s*\)", macro), (
        "the symbol must be resolved BY NAME out of the dlopen'd handle "
        "(dlsym(handle, #symbol)), which is what makes a missing symbol a "
        "named error rather than a null call later"
    )


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
    on disagreement rather than trusting either side alone.

    THE DECODE ITSELF IS PINNED HERE, NOT JUST THE COMPARISON. `arch` (skel
    hwinfo, plain decimal, e.g. 75) and `caps.arch_ver` (driver ARCH_VER, BCD
    nibble-packed, e.g. 0x8c75 = 35957) are in different encodings --
    comparing them raw is unconditionally false on every real device (see
    session.c's own file header). A test that only checked "a comparison and
    a `return -1` exist" could not see that the two operands were
    incommensurable; that is exactly the shape of the bug this pins. See
    test_session_arch_decode.py for a genuine, compiled-and-run behavioural
    test of the same decode function against the one measured value
    (0x8c75 -> 75)."""
    caps_body = _function_body(session, "hexlib_query_caps")
    assert "ARCH_VER" in caps_body
    assert "DSPRPC_GET_DSP_INFO" in caps_body

    decode_body = _function_body(session, "hexlib_decode_bcd_arch")
    assert re.search(r">>\s*4", decode_body), "must extract the high BCD nibble"
    assert re.search(r"\*\s*10", decode_body), "must weight the high nibble by 10"
    assert re.search(r"&\s*0x0f\b", decode_body), "must extract the low BCD nibble"

    open_body = _function_body(session, "hexlib_open")
    decode_call = re.search(
        r"hexlib_decode_bcd_arch\s*\(\s*caps\.arch_ver\s*\)", open_body
    )
    assert decode_call, (
        "hexlib_open must decode caps.arch_ver through hexlib_decode_bcd_arch "
        "before comparing it against the skel's arch -- comparing the raw "
        "ARCH_VER directly against __HEXAGON_ARCH__ can never agree "
        "(0x8c75 != 75) and would refuse every device session unconditionally"
    )
    mismatch = re.search(r"arch\s*!=\s*\w+", open_body)
    assert mismatch, "hexlib_open must cross-check driver arch against skel arch"
    assert not re.search(r"arch\s*!=\s*caps\.arch_ver\b", open_body), (
        "hexlib_open must never compare the skel's arch against the raw, "
        "undecoded caps.arch_ver"
    )
    assert mismatch.start() > decode_call.start(), (
        "the decoded value, not the raw caps.arch_ver, must be what gets "
        "compared against arch"
    )
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


def test_the_hosts_scale_kind_id_is_the_same_number_the_dsp_dispatches_on(main):
    """THE ONE HAND-COPIED KIND ID, BOUND TO ITS SOURCE OF TRUTH.

    `genentry.KIND_ID` is where kind ids live; `genentry.emit_table` writes the
    DSP's dispatch table straight from it. This host binary has no generated
    header to read, so `#define HEXLIB_KIND_SCALE 9u` is a hand-copy, pinned by
    a comment and -- until now -- by nothing executable. Mutating it to `10u`
    left the entire offline suite green.

    A WRONG VALUE HERE IS NOT ALWAYS LOUD. main.c's own comment argues it is
    (`hexlib_dispatch_batch` would answer ERR_NO_KERNEL, which --self-test
    reports as a failure), and that is true only while the number it drifts to
    is unregistered. 10 is `softmax` and 11 is `transpose`: as soon as either
    has a kernel, a `scale` request dispatches to it, the buffer count matches,
    both pointers are non-null, and the status is HEXLIB_DSP_OK. The wire
    carries no table version and `skel_dispatch.c` matches on the id alone, so
    nothing else in the system can notice.

    Read off the COMMENT-BLANKED source, so the "== 9" in the explanatory
    comment above the `#define` cannot satisfy this. Mutating either side --
    the `#define` or the Python dict -- fails it.
    """
    from hexlib.runtime.genentry import KIND_ID

    m = re.search(r"#define\s+HEXLIB_KIND_SCALE\s+(\d+)u\b", main)
    assert m, "main.c no longer defines HEXLIB_KIND_SCALE as a decimal literal"
    assert int(m.group(1)) == KIND_ID["scale"], (
        f"main.c dispatches scale as kind {m.group(1)}; genentry.KIND_ID says "
        f"{KIND_ID['scale']}. One of the two copies drifted, and the DSP obeys "
        f"the id it is sent."
    )


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

    assert "hexlib_classify_coherency_lane(" in success_block, (
        "the sentinel must only be read back (and classified) once both "
        "statuses are confirmed OK"
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


def test_coherency_check_treats_negative_zero_as_the_expected_zero_result(main):
    """`x * 0.0f` is -0.0, not +0.0, whenever `x` is negative -- true of many
    lanes of the self-test's own input -- and there is no -ffast-math here
    (toolchain.py) to make that not so. A bit-exact compare of the read-back
    buffer against `(__fp16) 0.0f` would misclassify that HEALTHY result as
    "sentinel unchanged" and report a coherency miss that never happened.

    THIS IS A SOURCE ASSERTION ONLY -- it can confirm the sign-bit mask
    exists in hexlib_classify_coherency_lane(), never that it actually
    classifies 0x8000 as zero. See
    hexlib/tests/test_coherency_lane_classification.py for the genuine,
    compiled-and-run behavioural test of that exact function against that
    exact bit pattern -- the same defect class as the arch-decode fix (see
    test_session_arch_decode.py), closed the same way."""
    classify_body = _function_body(main, "hexlib_classify_coherency_lane")
    assert re.search(r"&\s*0x7[Ff]{3}[Uu]?\b", classify_body), (
        "the expected-zero classification must mask off the sign bit "
        "(0x7FFF), not compare bit-exact equality to +0.0 alone -- see this "
        "function's own header comment on why -0.0 must count as zero"
    )
    body = _function_body(main, "run_coherency_check")
    # THIS NEGATIVE WAS VACUOUS AND IS NOW BOUND TO SOMETHING REAL. It used to
    # be `not re.search(r"memcmp\(&yr\[i\],\s*&zero\b", main)` -- text that
    # has never existed anywhere in main.c, in any revision, so the assertion
    # could not fail no matter what the C did. What it MEANT to forbid is a
    # bit-exact byte compare standing in for the magnitude classification, so
    # forbid that: run_coherency_check's read-back loop must reach its verdict
    # only through hexlib_classify_coherency_lane(). It uses memcpy (to get at
    # raw bits) and never memcmp, so any memcmp appearing in this function is
    # a comparison that has bypassed the classifier -- which is exactly the
    # regression. Verified by mutation: inserting a memcmp here fails this.
    assert "memcmp(" not in body, (
        "run_coherency_check must not compare the read-back buffer with "
        "memcmp -- every lane's verdict goes through "
        "hexlib_classify_coherency_lane(), and a bit-exact byte compare is "
        "how the -0.0 false coherency miss happened the first time"
    )
    assert re.search(r"hexlib_classify_coherency_lane\s*\(", body), (
        "run_coherency_check must classify each lane through "
        "hexlib_classify_coherency_lane(), not reimplement the check inline "
        "-- see test_coherency_lane_classification.py for why that function "
        "must stay the one thing exercised behaviourally"
    )


def test_coherency_check_verifies_the_surviving_bytes_are_really_the_sentinel(main):
    """A buffer that is neither the expected zero result nor the intact
    sentinel (garbled, or partially written) must not be folded into the
    'sentinel_unchanged' / coherency-miss verdict just because it failed the
    zero check -- it is a third, distinct outcome and must be its own
    branch with its own exit code."""
    classify_body = _function_body(main, "hexlib_classify_coherency_lane")
    assert re.search(r"bits\s*==\s*sentinel_bits", classify_body), (
        "the surviving bits must be compared, bit-exact, against the real "
        "sentinel's own bits -- not merely assumed to be the sentinel "
        "because they were not zero"
    )
    assert "HEXLIB_LANE_OTHER" in classify_body, (
        "a lane that is neither zero nor the sentinel must be its own, "
        "third classification -- not folded into either of the other two"
    )

    body = _function_body(main, "run_coherency_check")
    assert "HEXLIB_EXIT_COHERENCY_GARBLED" in main
    assert "exit_code = HEXLIB_EXIT_COHERENCY_GARBLED;" in body
    assert '"COHERENCY buffer_garbled\\n"' in body


def test_coherency_check_documents_its_own_scope_limits(main_comments):
    """Design doc §6.1 (corrected 2026-08-11): the table that makes
    cycles_total load-bearing covers ONLY the DSP-write -> host-read
    direction, for scale_fp16's own write pattern -- not the reverse
    direction, and not every kernel. That caveat must live in this file's
    own comments, not only in the on-device test's docstring, or a future
    reader of just this file could believe a pass here is a general
    coherency proof.

    THE ONE TEST IN THIS FILE THAT TAKES `main_comments`, NOT `main`. Its
    subject IS the comment text -- a prose caveat written for a human reader --
    so blanking comments out would make it assert nothing and it would fail
    immediately. Every other check here must use `main`; see the module
    docstring."""
    main = main_comments
    assert "DSP-write" in main and "host-read" in main
    assert "host-write" in main and "DSP-read" in main
    assert "kernel-independent" in main.lower()

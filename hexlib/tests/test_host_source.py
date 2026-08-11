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

THE THIRD TIGHTENING: STRING LITERALS ARE BLANKED NOW TOO. `csource.code_only`
used to leave string and character literals intact, so the very defect the
second tightening's third bullet describes -- `"dlopen(" in body` satisfied by
driver.c's own error format string -- was still available to every other check
in this file, and to a `}` inside a literal truncating any slice. Fixed in
`csource` (see its module docstring for the three mechanisms and the
mutations). What that means HERE is that `code_only` now hands back text with
no literal payload in it at all, and the handful of checks in this file whose
subject genuinely IS literal text -- a printed line another test asserts on, a
command-line flag string, the dlopen candidate path, the `&_dom=cdsp` spelling
that must NOT appear -- take one of the `*_strings` fixtures below and pass
`keep_strings=True` to the slicer, saying so at the call site. Every other
check keeps the ordinary fixture, because for those a token inside a format
string is evidence of nothing.
"""
import pathlib
import re

import pytest

from hexlib.tests.csource import block_from as _block_from
from hexlib.tests.csource import calls as _calls
from hexlib.tests.csource import code_only as _code_only
from hexlib.tests.csource import code_only_keeping_strings as _code_with_strings
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


# --------------------------------------------------------------------------
# THE `*_strings` FIXTURES: comments blanked, STRING LITERALS INTACT. For the
# few claims whose subject IS literal text. Blanking is length-preserving in
# both views, so an offset found in a `*_strings` slice is valid in the
# ordinary slice of the same function and vice versa -- which is how a test can
# locate a flag string in one view and then brace-slice the block in the other.
# Never use these for a "this guard is here" / "this constant is absent" check;
# that is what the plain fixtures are for. See csource.py's docstring.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def driver_strings():
    return _code_with_strings((H / "driver.c").read_text())


@pytest.fixture(scope="module")
def session_strings():
    return _code_with_strings((H / "session.c").read_text())


@pytest.fixture(scope="module")
def main_strings():
    return _code_with_strings((H / "main.c").read_text())


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


def test_libcdsprpc_is_dlopened_not_linked(driver, driver_strings):
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
    # THE ONE CHECK HERE WHOSE SUBJECT IS A LITERAL: the candidate path is a
    # string, so it has to be looked for in the literal-bearing view. The
    # `handle = dlopen(...)` check above deliberately does NOT -- that is the
    # one the format string used to satisfy.
    body_strings = _function_body(driver_strings, "hexlib_drv_init",
                                  keep_strings=True)
    assert '"libcdsprpc.so"' in body_strings, (
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


def test_the_uri_is_built_not_hardcoded_with_a_domain(session, session_strings):
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
    # LITERAL-BEARING VIEW, DELIBERATELY. This negative is about a STRING
    # SPELLING -- the whole claim is that nobody wrote the domain suffix out by
    # hand -- so it must be checked against text in which literals survive.
    # Against `code_only` text it would pass vacuously (every literal blanked,
    # so no literal can ever be found), which is a weaker check than the one
    # this line was written to make.
    assert '"&_dom=cdsp"' not in session_strings

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


def test_a_size_that_would_truncate_on_the_way_down_is_refused(buffers):
    """`size` is a `size_t` and is narrowed TWICE: `(int) size` for
    rpcmem_alloc and `(uint32_t) size` for hexlib_iface_mmap (qaic's own
    generated signature from the IDL). Neither cast can report a loss, and they
    can disagree with each other -- 2 GiB or more would register a mapping of
    one length for a buffer allocated at another, and `(int) size` can go
    negative outright.

    Unreachable today (every call site passes a plan-computed tensor size) and
    it fails closed on the DSP side if it ever were not (skel_bufs.c's
    `b->size > m->size` check), so the fix is deliberately a guard rather than
    a widening of the wire. What this pins is that the guard runs BEFORE the
    first cast: a check placed after rpcmem_alloc protects nothing, because the
    truncation has already happened by then."""
    body = _function_body(buffers, "hexlib_alloc")
    # `[^{]*` rather than `[^)]*`: the bound is written with a cast in it
    # (`(size_t) INT_MAX`), so the condition legitimately contains parens.
    m = re.search(r"if\s*\([^{]*\bsize\s*>[^{]*\)\s*\{", body)
    assert m, (
        "hexlib_alloc must refuse a size too large for the int/uint32 casts "
        "below it -- neither cast can report the truncation"
    )
    assert m.start() < body.index("hexlib_rpcmem_alloc("), (
        "the size guard must run before the first narrowing cast, not after it"
    )
    guard = _block_from(body, m.end() - 1)
    assert "return -1;" in guard, (
        "an out-of-range size must be refused, not merely logged"
    )
    assert "INT_MAX" in body, (
        "the bound must be the narrower of the two casts (rpcmem_alloc's int), "
        "not uint32's -- a value that fits uint32 can still be negative as an "
        "int"
    )


def test_the_host_never_puts_an_address_on_the_wire(buffers):
    """hexlib_buf_to_desc -- the one place a hexlib_buf_desc is filled in from
    this side -- must zero `base` itself, first (right after the memset, not
    merely somewhere before the struct is used), and nothing in the file may
    derive `base` from the host pointer (`buf->ptr`/`ptr`).

    TWO INDEPENDENT WEAKNESSES, BOTH PROVEN, BOTH FIXED HERE. `d->base =
    (uint64_t)(uintptr_t) buf->ptr;` -- the host's own virtual address on the
    wire, which works perfectly under the simulator's shared address space and
    can only fail on silicon -- passed all 28 tests in this file when written
    two ways at once:

      * the `base = 0;` positive was satisfied by an `fprintf` format string
        containing that text (`csource` used to hand literals back intact; it
        no longer does), and
      * the whole-file `base\\s*=[^;]*\\bptr\\b` negative was satisfied by
        routing the pointer through a temp named `hostaddr`, because the
        forbidden token no longer appeared on the assignment's own line. A
        negative check written as a pattern over the RHS can always be dodged
        by a rename; that is a property of the shape of the check, not of the
        name chosen.

    So the check is inverted into an EXHAUSTIVE one, which a rename cannot
    dodge: enumerate every assignment to `d->base` in this function and require
    that the complete set of right-hand sides is exactly `0`. And require that
    this function never reads the host pointer AT ALL -- no `ptr` token in its
    body -- so there is nothing available to launder through a temp under any
    name. `hexlib_buf_to_desc` legitimately needs only `buf->size` and
    `buf->fd`."""
    body = _function_body(buffers, "hexlib_buf_to_desc")
    assigned = [rhs.strip() for rhs in re.findall(r"d->base\s*=\s*([^;]+);", body)]
    assert assigned == ["0"], (
        f"hexlib_buf_to_desc must assign d->base exactly once, and the value "
        f"must be 0 -- the DSP fills it in from its own mapping table "
        f"(skel_bufs.c). Found right-hand sides {assigned!r}"
    )
    memset_end = body.index(";", body.index("memset(")) + 1
    base_clear = re.search(r"\bbase\s*=\s*0\s*;", body)
    assert base_clear, "base must be explicitly cleared, not left to memset alone"
    assert base_clear.start() < body.index("d->size", memset_end), (
        "base must be cleared before the other fields are filled in"
    )
    assert "ptr" not in body, (
        "hexlib_buf_to_desc must not so much as READ the host pointer -- it "
        "needs buf->size and buf->fd and nothing else, and a function that "
        "cannot see the address cannot put it on the wire under any variable "
        "name"
    )
    # Kept as a belt, and honestly labelled: this is the rename-defeatable
    # form. The exhaustive check above is the one that holds.
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
    # AND NOTHING THAT COULD REGISTER IT UNDER ANOTHER NAME. The named ban above
    # is satisfied by moving the registration into a one-line helper and calling
    # THAT from here, which would silently un-break --unmapped: the fd would be
    # registered after all, hexlib_bufs_map would find it, and the mode would
    # report success for the case it exists to make fail. An exhaustive callee
    # set cannot be dodged by a rename (see csource.calls()).
    assert _calls(skip_block) == {"printf"}, (
        f"the --unmapped branch must do nothing but say so on stdout; found "
        f"calls to {_calls(skip_block)!r}"
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


def test_self_test_prints_cycles_total_after_the_existing_pass_line(main, main_strings):
    """The response header's cycles_total (skel_dispatch.c's PCYCLE bracket
    around the kernel call) must be printed AFTER, never instead of, the
    existing 'PASS (%d values, bit-exact)' line -- so the exact success
    string test_on_device.py's `test_scale_fp16_runs_on_the_dsp_and_is_
    correct` already asserts on stays byte-for-byte intact, and the new
    cycles line is strictly additive.

    BOTH ORDERED THINGS ARE PRINTF FORMAT STRINGS, so this is one of the few
    checks that must run against the literal-bearing view -- the claim is
    literally about what gets printed and in what order. The one non-literal
    half (the value printed comes off the response header, not a constant)
    stays on the ordinary view."""
    body_strings = _function_body(main_strings, "run_self_test", keep_strings=True)
    pass_idx = body_strings.index("PASS (%d values, bit-exact)")
    cycles_idx = body_strings.index("cycles_total=%llu", pass_idx)
    assert pass_idx < cycles_idx
    body = _function_body(main, "run_self_test")
    assert "full_hdr.cycles_total" in body


def test_self_test_flag_parsing_routes_unmapped_and_coherency_correctly(main, main_strings):
    """main()'s --self-test branch must recognize both --unmapped and
    --coherency-check past argv[1], route --coherency-check to
    run_coherency_check(), and thread the --unmapped flag straight into
    run_self_test(unmapped) -- not merely mention both flag strings
    somewhere in the function, which a comment or an unreachable branch
    would also satisfy.

    THE FLAG NAMES ARE STRING LITERALS -- argv is compared against them -- so
    finding them needs the literal-bearing view. The ROUTING half (the branch
    exists and calls the right function) stays on the ordinary, blanked view,
    which is the half a FARF or a usage() line could otherwise satisfy. The two
    views are the same length, so the `"--self-test"` offset found in one is
    the right offset to brace-slice the other from."""
    body = _function_body(main, "main")
    body_strings = _function_body(main_strings, "main", keep_strings=True)
    assert len(body) == len(body_strings)
    self_test_pos = body_strings.index('"--self-test"')
    self_test_block = _block_from(body, self_test_pos)
    self_test_block_strings = _block_from(body_strings, self_test_pos,
                                         keep_strings=True)

    assert '"--unmapped"' in self_test_block_strings
    assert '"--coherency-check"' in self_test_block_strings
    assert re.search(r"run_coherency_check\s*\(\s*\)", self_test_block)
    assert re.search(r"run_self_test\s*\(\s*unmapped\s*\)", self_test_block)

    coherency_if = re.search(r"if\s*\(\s*coherency\s*\)", self_test_block)
    assert coherency_if, "--coherency-check must be checked as its own branch"
    coherency_block = _block_from(self_test_block, coherency_if.end())
    assert re.search(r"run_coherency_check\s*\(\s*\)", coherency_block), (
        "the coherency branch must actually call run_coherency_check(), not "
        "merely check the flag and fall through"
    )


def test_usage_mentions_the_new_self_test_modifiers(main_strings):
    """LITERAL-BEARING VIEW BY NATURE: usage() text is nothing but string
    literals, and what this asserts is that a human running --help is told
    about both modifiers."""
    body = _function_body(main_strings, "usage", keep_strings=True)
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


def test_coherency_check_reads_the_sentinel_only_after_both_statuses_are_ok(
    main, main_strings
):
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
    # The three ORDERED things are printf format strings, so their relative
    # order is a claim about literal text and is checked in the literal-bearing
    # view. Same offsets (blanking preserves length), so the `else` boundary
    # found above is the right one to slice there too.
    body_strings = _function_body(main_strings, "run_coherency_check",
                                 keep_strings=True)
    success_block_strings = _block_from(body_strings, else_pos, keep_strings=True)
    cycles_idx = success_block_strings.index("cycles_total=%llu")
    overwritten_idx = success_block_strings.index('"COHERENCY sentinel_overwritten\\n"')
    unchanged_idx = success_block_strings.index('"COHERENCY sentinel_unchanged\\n"')
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


def test_coherency_check_verifies_the_surviving_bytes_are_really_the_sentinel(
    main, main_strings
):
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
    # The printed verdict line itself -- a literal, so the literal-bearing view.
    body_strings = _function_body(main_strings, "run_coherency_check",
                                 keep_strings=True)
    assert '"COHERENCY buffer_garbled\\n"' in body_strings


def test_caps_reports_a_driver_failure_through_its_exit_code(main, main_strings):
    """`--caps` EXITED 0 WHEN THE DRIVER FAILED TO LOAD. `print_caps()`
    returned `void`, both failure branches printed to stderr and returned, and
    `main()` returned HEXLIB_EXIT_OK regardless -- so on a device whose image
    has no `libcdsprpc.so` for this ABI, `./hexlib_run --caps; echo RC=$?`
    printed "could not load the FastRPC driver" and `RC=0`, which any `set -e`
    wrapper or CI step reads as a pass. It is also the first mode run on
    unfamiliar silicon and the one most likely to fail there.

    Three things are checked, because the defect needed all three to be wrong:
    the function returns int, EVERY early return in it carries a nonzero exit
    constant, and main() actually propagates the value instead of discarding
    it."""
    m = re.search(r"\bstatic\s+int\s+print_caps\s*\(\s*void\s*\)", main)
    assert m, (
        "print_caps must return an exit code, not void -- a void return is "
        "why a driver-load failure exited 0"
    )
    body = _function_body(main, "print_caps")
    returns = re.findall(r"return\s+([^;]+);", body)
    assert returns, "print_caps must return something"
    assert all(r.strip().startswith("HEXLIB_EXIT_") for r in returns), (
        f"every return in print_caps must be a named exit code, got {returns!r}"
    )
    # The two failure branches must NOT return OK; the last (success) one must.
    assert returns[-1].strip() == "HEXLIB_EXIT_OK", (
        f"print_caps's final, success return must be OK, got {returns[-1]!r}"
    )
    for r in returns[:-1]:
        assert r.strip() != "HEXLIB_EXIT_OK", (
            "a print_caps failure branch returns HEXLIB_EXIT_OK -- that is the "
            "original defect, moved rather than fixed"
        )

    # The flag itself is a literal, so it is located in the literal-bearing
    # view; the block's CONTENT is then checked in the blanked one, where
    # neither a comment nor a log line can supply the return this is about.
    main_body = _function_body(main, "main")
    main_body_strings = _function_body(main_strings, "main", keep_strings=True)
    caps_pos = main_body_strings.index('"--caps"')
    caps_block = _block_from(main_body, caps_pos)
    assert re.search(r"return\s+print_caps\s*\(\s*\)\s*;", caps_block), (
        "main() must RETURN print_caps()'s value -- calling it and then "
        "returning HEXLIB_EXIT_OK is exactly the bug"
    )
    assert "HEXLIB_EXIT_OK" not in caps_block, (
        "main()'s --caps branch must not name a constant exit code at all; "
        "the code comes from print_caps()"
    )


# The three places §6.1's coherency table is written down. A doc claiming a
# guarantee the code does not deliver is, on this project, a defect at the same
# weight as a code bug -- so the correction has to land in all three or the
# stale one becomes the one someone reads on the first device job.
_COHERENCY_TABLE_SITES = (
    pathlib.Path("hexlib/runtime/host/main.c"),
    pathlib.Path("docs/superpowers/specs/2026-08-10-silicon-path-runtime-design.md"),
    pathlib.Path("hexlib/device/qdc/test_on_device.py"),
)

# Each element of the correction, and why it must be present in every copy:
#   "unreachable"      -- row 1 (`cycles 0` + sentinel intact -> "a dispatch
#                         bug") cannot happen: reaching the read-back at all
#                         requires both statuses OK, which requires k->fn to
#                         have been called and returned OK, and PCYCLE brackets
#                         exactly that call.
#   "not discriminated" -- the `sentinel_unchanged` row is consistent with a
#                         coherency miss AND with a kernel/entry that returned
#                         OK without writing `y`. Calling it "COHERENCY" is the
#                         misattribution §6.1 exists to prevent.
#   "PCYCLEEN"         -- if the counter does not advance in the unsigned PD,
#                         every row inverts; that is why a zero is asserted
#                         against rather than assumed impossible.
#   "buffer_garbled"   -- the third outcome a previous fix added must appear in
#                         the table too, or the table is still incomplete.
_CORRECTION_ELEMENTS = ("unreachable", "not discriminated", "pcycleen", "buffer_garbled")


# IDS ARE HAND-WRITTEN, NOT DERIVED FROM THE FILENAME. `ids=lambda p: p.name`
# put the literal text `test_on_device.py` into a node id, and
# test_qdc_on_device_is_excluded.py asserts that exact string never appears in
# `pytest --collect-only` output (its way of proving the on-device file is not
# collected) -- so a parametrize id here made THAT test fail, on a file that
# was correctly excluded. Reproduced before this comment existed.
@pytest.mark.parametrize(
    "path", _COHERENCY_TABLE_SITES, ids=("host_main", "design_spec", "device_test")
)
def test_the_coherency_table_correction_landed_everywhere_it_is_written_down(path):
    """READ WITH COMMENTS ON, DELIBERATELY -- unlike every other check in this
    file. The subject IS the prose: §6.1's table is a claim made to a human
    about what the first device job's output will mean, and it was asserting a
    separation the code does not achieve. Two of the three copies are comments
    (main.c's `run_coherency_check` header, test_on_device.py's docstring) and
    the third is a design doc, so blanking comments would make this assert
    nothing.

    Deleting the correction from ANY ONE of the three fails this."""
    text = path.read_text(encoding="utf-8").lower()
    missing = [e for e in _CORRECTION_ELEMENTS if e not in text]
    assert not missing, (
        f"{path} is missing part of §6.1's corrected coherency table: "
        f"{missing!r}. All three copies must say the same thing -- a stale one "
        f"is the copy someone reads while triaging job 1."
    )


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


def test_the_self_test_batch_names_its_layout_instead_of_writing_a_bare_zero(main):
    """`tens[i].layout` must be spelled with the macro, not a literal.

    This was `tens[i].layout = 0;` with a trailing comment naming
    `LAYOUT_ID["row_major"]` -- and unlike the `dtype = 1` literal beside it,
    which genentry's emitted `a->dtype[0] != 1u` guard rejects loudly at run
    time, NOTHING checked layout at all. `grep -c layout` over this file was 0.

    A bare 0 is not wrong today; it is unbound. Insert a layout ahead of
    row_major in `wire.LAYOUT_ID` and `pack_batch` emits 1 while this file
    keeps emitting 0, and `--self-test` still prints `PASS (4100 values,
    bit-exact)` over a buffer the batch declared as something else. The enum
    exists (per hexlib_dsp.h's own header) so that un-repacked weights are a
    plan-time error rather than silent corruption, and `LAYOUT_ID` already
    carries `q4_0_repacked` for the matmul that comes next.

    The macro's VALUE is bound to `wire.LAYOUT_ID` by a compiled probe in
    test_wire_struct_layout.py; this test only pins that main.c goes through
    the macro. Read off comment-blanked source, so the explanatory comment
    cannot satisfy it.
    """
    body = _function_body(main, "build_scale_batch")
    assert "HEXLIB_LAYOUT_ROW_MAJOR" in body, (
        "build_scale_batch must set tens[].layout from HEXLIB_LAYOUT_ROW_MAJOR, "
        "not from a bare integer whose only tie to wire.LAYOUT_ID is a comment"
    )
    assert not re.search(r"\.layout\s*=\s*\d", body), (
        "a numeric literal is being assigned to .layout again; use the macro"
    )

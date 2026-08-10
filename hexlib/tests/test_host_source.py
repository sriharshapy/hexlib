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


def _function_body(src, name):
    """Slice the text of a C function from its signature to its matching
    closing brace, by simple brace-depth counting. Good enough for this
    project's straight-line C; not a general C parser.

    Copied from `test_skel_bufs_source.py` (Task 4), per the coordinator's
    note that a third variant of the same helper is not wanted."""
    m = re.search(rf"\b{re.escape(name)}\s*\([^;{{]*\)\s*\{{", src)
    assert m, f"could not find the definition of {name}() in the source"
    start = m.end() - 1  # position of the opening brace
    depth = 0
    for i in range(start, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
    raise AssertionError(f"unbalanced braces while slicing {name}()")


def _block_from(text, pos):
    """From `pos`, find the next '{' and return the brace-matched block it
    opens (inclusive). Generalizes the closing half of `_function_body` to an
    arbitrary starting offset, so one specific `if (...) { ... }` can be
    isolated instead of just checking "somewhere in the next N characters" --
    which a later, unrelated `return` statement could satisfy by accident."""
    brace = text.index("{", pos)
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace : i + 1]
    raise AssertionError("unbalanced braces while slicing a block")


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
    dlopen() and the candidate path must
    both live in the function that actually loads the driver, not merely
    somewhere in the file (e.g. a comment mentioning both)."""
    body = _function_body(driver, "hexlib_drv_init")
    assert "dlopen(" in body
    assert "libcdsprpc.so" in body


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
    would silently stop tracking either constant if it ever changed."""
    assert re.search(r"hexlib_iface_URI\s+CDSP_DOMAIN\b", session)
    assert '"&_dom=cdsp"' not in session


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
    straight to the control APIs."""
    assert "#include <remote.h>" in session
    assert not re.search(r"=\s*11\b", session)
    # Nothing may pass a literal digit as the request id argument itself --
    # that would dodge the "= 11" check above while still hardcoding a
    # different id the same way.
    assert not re.search(r"remote_(handle|session)_control\s*\(\s*\d", session)


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

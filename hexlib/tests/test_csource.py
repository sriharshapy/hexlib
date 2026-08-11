# hexlib/tests/test_csource.py
"""Tests for the shared comment-aware C source slicer (csource.py) itself.

THE PROPERTY BEING PROVEN: a comment that mentions a function's (or a call's)
name in prose, or that merely contains a stray brace/paren, must not be able
to move any of `function_body`/`block_from`/`block_after_call`'s boundaries
off the real code. Each test below is built so that the OLD, comment-blind
regex (`name\\s*\\([^;{]*\\)\\s*\\{`, matched and brace-counted directly
against the raw source) provably gets it wrong on that exact input -- either
landing on the wrong block, or raising "unbalanced braces" outright -- which
is the failure mode this module exists to close. See the task report for the
manual before/after run that confirms this (reverting `csource.py` to the
comment-blind implementation makes these fail).

THE SECOND PROPERTY, ADDED AFTER A MUTATION FOUND THE GAP: the text these
slicers RETURN must be comment-blanked too, not just their boundaries
comment-aware. The original implementation returned a slice of the raw,
comment-bearing source, so any payload check a consumer ran against it
(`x in body`, `re.search(..., body)`) could be satisfied by a comment INSIDE
the extracted body -- which is exactly how a proven mutation of skel_bufs.c
kept test_skel_bufs_source.py at 8 passed while deleting the branch's central
invariant. `test_*_returns_comment_blanked_text*` below pin that half: each
one fails against the old, non-stripping implementation.
"""
import pathlib
import re

from hexlib.tests.csource import (
    block_after_call,
    block_from,
    code_only,
    function_body,
    strip_comments,
)


def test_strip_comments_blanks_comments_but_preserves_length_and_strings():
    src = (
        '/* block\n comment */int x = 1; // trailing\n'
        'const char *s = "not a // comment or /* one */ either";\n'
    )
    stripped = strip_comments(src)
    assert len(stripped) == len(src)
    assert "block" not in stripped
    assert "trailing" not in stripped
    # the string literal (including its embedded comment-lookalikes) survives
    assert '"not a // comment or /* one */ either"' in stripped
    assert "int x = 1;" in stripped


def test_function_body_is_not_derailed_by_a_comment_naming_it_first():
    """A comment mentions `foo()` (with empty, immediately-closed parens) in
    prose BEFORE foo's real definition, and an unrelated function `bar`'s
    signature+body sits between the comment and foo's real definition, with
    only whitespace between `bar`'s own closing `)` and its opening `{`.

    Under the old comment-blind regex, greedily consuming `[^;{]*` from the
    comment's "foo(" runs straight through the rest of the comment and stops
    at the first `{` it hits at all -- which is `bar`'s, not foo's -- then
    backtracks onto `bar(void)`'s `)` immediately followed by `{`
    (`bar(void) {`), matching THERE instead: the slicer would then return
    `bar`'s body (`{ return 1; }`) for a query about `foo`, silently."""
    src = (
        "/* note: foo() should always return 0 -- see below */\n"
        "int bar(void) { return 1; }\n"
        "\n"
        "int foo(void) {\n"
        "    return 0;\n"
        "}\n"
    )
    body = function_body(src, "foo")
    assert "return 0;" in body
    assert "return 1;" not in body


def test_function_body_is_not_derailed_by_a_comment_mentioning_a_sibling():
    """The reverse shape of the bug that actually broke two tests in
    test_host_source.py: a comment INSIDE one real function mentions a
    DIFFERENT real function's name in prose, ahead of that other function's
    own definition (this is the same shape as driver.c's HEXLIB_DLSYM
    macro-comment relative to hexlib_drv_init -- that file dodges it only by
    luck, via a backslash line-continuation the comment-blind regex cannot
    step over; this case has no such luck: `if (1) {` sits between the
    mention and the nearest disallowed character with nothing but whitespace
    in between, so the old regex has a reachable, wrong `{` to seize on)."""
    src = (
        "int helper(void) {\n"
        "    /* eventually calls target() to finish up */\n"
        "    if (1) { return 1; }\n"
        "}\n"
        "\n"
        "int target(void) {\n"
        "    return 42;\n"
        "}\n"
    )
    body = function_body(src, "target")
    assert "return 42;" in body
    assert "return 1;" not in body


def test_block_from_is_not_derailed_by_a_stray_brace_in_a_comment():
    """Between `pos` (right after the `if`'s condition) and the real guarded
    block, a comment contains a brace character that has nothing to do with
    real control flow. The old comment-blind `text.index("{", pos)` would
    seize on THAT brace as if it opened the guarded block, then either
    mis-slice or -- as here, because a second, real `{` still follows before
    the matching `}` -- raise "unbalanced braces" outright."""
    text = (
        "if (x) /* a comment with a stray { brace in it */ {\n"
        "    return 1;\n"
        "}\n"
    )
    pos = text.index(")")
    block = block_from(text, pos)
    assert block.strip().startswith("{")
    assert "return 1;" in block
    # exactly one open/close pair in the returned slice -- the real block,
    # not a fragment that swallowed the comment's own stray brace as an
    # extra nesting level
    assert block.count("{") == 1 and block.count("}") == 1


def test_block_after_call_is_not_derailed_by_a_comment_between_call_and_guard():
    """A comment between the call and its `if`-guard mentions a DIFFERENT
    call's name and contains its own parenthesized text -- shaped so the old
    comment-blind implementation's own paren-walk (which starts scanning
    right at the real call's own `(`, not at the comment) is not directly
    fooled, but the `if (` window-check and the block-brace search underneath
    it are exercised against live-looking but decorative comment text, which
    must not change what block gets returned."""
    body = (
        "int rc = real_call(a, b);\n"
        "/* note: other_call() is unrelated and has its own { guard } shape */\n"
        "if (rc != 0) {\n"
        "    return HEXLIB_DSP_ERR_INTERNAL;\n"
        "}\n"
    )
    block = block_after_call(body, "real_call")
    assert "HEXLIB_DSP_ERR_INTERNAL" in block
    assert block.count("{") == 1 and block.count("}") == 1


def test_function_body_still_finds_the_real_definition_with_no_comments():
    """Sanity: comment-awareness must not break the ordinary, comment-free
    case it is layered on top of."""
    src = "int plain(void) {\n    return 7;\n}\n"
    assert "return 7;" in function_body(src, "plain")


# ==============================================================================
# The returned text, not just the boundaries. A guard deleted and left behind
# as a comment must not satisfy a payload check run against the slice -- the
# exact hole a mutation of skel_bufs.c walked through.
# ==============================================================================


def test_function_body_returns_comment_blanked_text_by_default():
    """The mutation shape, in miniature: the real `return -1;` is gone and
    survives only as a comment inside the body. `"return -1;" in body` must
    NOT be true, and the code that IS still there must still be visible."""
    src = (
        "int guard(int x) {\n"
        "    if (x < 0) {\n"
        "        /* return -1; */\n"
        "        x = 0;\n"
        "    }\n"
        "    return x;\n"
        "}\n"
    )
    body = function_body(src, "guard")
    assert "return -1;" not in body, (
        "a commented-out return must not satisfy a payload check on the body"
    )
    assert "x = 0;" in body
    assert "return x;" in body
    # length is preserved, so offsets taken from the slice still line up with
    # the same slice of the original text
    assert len(body) == len(function_body(src, "guard", strip=False))


def test_function_body_strip_false_still_returns_the_original_text():
    """The escape hatch is real, and explicit: `strip=False` gives back the
    comment-bearing slice for a caller that means to inspect comments."""
    src = "int guard(void) {\n    /* return -1; */\n    return 0;\n}\n"
    raw = function_body(src, "guard", strip=False)
    assert "/* return -1; */" in raw


def test_block_from_returns_comment_blanked_text_by_default():
    """Same property for `block_from`: the guarded block's payload must be
    code. A commented-out `return -1;` inside the block is not a refusal."""
    text = "if (x) {\n    /* return -1; */\n    log_it();\n}\n"
    block = block_from(text, text.index(")"))
    assert "return -1;" not in block
    assert "log_it();" in block


def test_block_after_call_returns_comment_blanked_text_by_default():
    """Same property for `block_after_call`: the status the caller looks for
    in the checked block must be returned, not merely mentioned in prose."""
    body = (
        "int rc = real_call(a, b);\n"
        "if (rc != 0) {\n"
        "    /* return HEXLIB_DSP_ERR_INTERNAL; */\n"
        "    rc = 0;\n"
        "}\n"
    )
    block = block_after_call(body, "real_call")
    assert not re.search(r"return\s+HEXLIB_DSP_ERR_\w+\s*;", block), (
        "a commented-out error return must not count as propagating a status"
    )
    assert "rc = 0;" in block


def test_no_test_file_carries_its_own_private_copy_of_the_slicer():
    """THE CONSOLIDATION CLAIM, MADE SELF-ENFORCING RATHER THAN PROMISED.
    csource.py's docstring asserted the consolidation was complete while two
    private copies were still live -- a claim about the codebase written in
    prose, which is exactly the kind of thing that rots silently. This checks
    it instead.

    A private copy is a `def` of one of these names in any hexlib/tests module
    other than csource.py itself. An `import ... as _function_body` alias is
    not a copy and is the intended usage, so only `def` is matched.
    `_macro_body` in test_host_source.py is deliberately excluded: it slices a
    backslash-continued `#define`, which brace counting cannot do, and its own
    docstring says why it is a narrowly-scoped sibling rather than a fourth
    slicer."""
    shared = ("strip_comments", "code_only", "function_body", "block_from",
              "block_after_call")
    here = pathlib.Path(__file__).parent
    offenders = []
    for path in sorted(here.glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        for name in shared:
            if re.search(rf"^\s*def\s+_?{name}\s*\(", text, re.M):
                offenders.append(f"{path.name} defines its own {name}()")
    assert not offenders, (
        "private copies of the shared C slicer are back -- import them from "
        "hexlib.tests.csource instead, and see that module's docstring for why "
        "a near-copy is worse than no copy: "
        + "; ".join(offenders)
    )


def test_code_only_is_the_whole_file_form_of_the_same_guarantee():
    """`code_only` is what a whole-file fixture goes through before any
    payload check runs against it -- same blanking, same length, so a
    constant or a call named only in a comment cannot satisfy (or trip) a
    file-wide check."""
    src = '/* calls HAP_mmap() here */\nint f(void) { return 0; }\n'
    assert "HAP_mmap" not in code_only(src)
    assert len(code_only(src)) == len(src)
    assert "int f(void) { return 0; }" in code_only(src)

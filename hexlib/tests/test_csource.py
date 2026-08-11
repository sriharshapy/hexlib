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
"""
import re

from hexlib.tests.csource import block_after_call, block_from, function_body, strip_comments


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

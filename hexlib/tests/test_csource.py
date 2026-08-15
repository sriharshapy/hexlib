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

THE THIRD PROPERTY, AND WHY THIS FILE'S OWN FIRST TEST USED TO ENCODE THE BUG.
Everything above was about COMMENTS, and this file had ZERO coverage of string
literals -- all twelve tests used comments only. Worse, its first test asserted
`'"not a // comment or /* one */ either"' in stripped`: it pinned literal
PRESERVATION as the property to protect. Preservation is right for one narrow
purpose (a literal must be recognized and consumed as one token, so a `//`
inside it is not misread as a comment start) and wrong for the two purposes
that actually matter to every consumer -- brace counting, and the payload text
handed back. A merge-gate review showed literals were the comment hole in a
different vehicle, in three distinct mechanisms, each reproduced against the
then-current code and each pinned by its own test below:

  (A) a positive check satisfied by a FARF/printf format string while the real
      code is deleted -- `test_*_blanks_a_log_lines_payload*`;
  (B) a `}` inside a literal truncating a brace-depth slice, so every negative
      check after it is answered by a fragment -- `test_*_not_truncated_by_a_
      closing_brace_in_a_literal`;
  (C) a `{` inside a literal extending a block past its real end, destroying
      the scoping the whole module exists to provide -- `test_block_from_is_
      not_extended_by_an_opening_brace_in_a_literal`.

Each of those fails against the literal-preserving implementation. The narrow
purpose preservation was right for is pinned separately, in
`test_a_comment_lookalike_inside_a_literal_is_not_a_comment_start`: the
literal's PAYLOAD is blanked, and the code after it still survives, which is
only possible if the `//` inside it was never treated as a comment start.
"""
import pathlib
import re

from hexlib.tests.csource import (
    block_after_call,
    block_from,
    code_only,
    code_only_keeping_strings,
    function_body,
)


def test_code_only_blanks_comments_preserving_length():
    src = (
        '/* block\n comment */int x = 1; // trailing\n'
        'int y = 2;\n'
    )
    stripped = code_only(src)
    assert len(stripped) == len(src)
    assert "block" not in stripped
    assert "trailing" not in stripped
    assert "int x = 1;" in stripped
    assert "int y = 2;" in stripped


def test_a_comment_lookalike_inside_a_literal_is_not_a_comment_start():
    """THE NARROW PROPERTY LITERAL PRESERVATION WAS ACTUALLY FOR, kept, while
    the payload is blanked. This is what this file's first test used to get
    backwards: it asserted the whole literal SURVIVED.

    A `//` inside a string must not be mistaken for a comment start -- if it
    were, everything after it on that line (here, the statement terminator and
    the following line's code) would be blanked away too. So the literal has to
    be recognized and consumed as ONE token. What is written back for that
    token is a separate question, and the answer is: delimiters kept, interior
    blanked. Both halves are asserted here, and the second half fails against
    the implementation that returned literals untouched."""
    src = (
        'const char *s = "not a // comment or /* one */ either";\n'
        'int after = 1;\n'
    )
    out = code_only(src)
    assert len(out) == len(src)
    # Consumed as one token: the code after the literal is still there, which
    # could not be true if the `//` inside it had started a comment.
    assert "int after = 1;" in out
    assert 'const char *s =' in out
    # ... and it is still visibly a literal, so "a string is present here" is
    # still answerable -- only its contents are gone.
    assert out.count('"') == 2
    # The payload, on the other hand, must not be readable as code.
    assert "comment" not in out
    assert "either" not in out


def test_code_only_keeping_strings_is_the_deliberate_escape_hatch():
    """The escape hatch really does return the literal, for the few checks
    whose subject IS literal text -- and it still blanks comments, so it is
    never a way back to raw source."""
    src = '/* a note */ printf("PASS (%d values)"); // trailing\n'
    out = code_only_keeping_strings(src)
    assert len(out) == len(src)
    assert '"PASS (%d values)"' in out
    assert "a note" not in out
    assert "trailing" not in out
    # And the two views are interchangeable in either order, so a fixture built
    # with one can be re-blanked by a slicer using the other.
    assert code_only(out) == code_only(src)
    assert code_only_keeping_strings(code_only(src)) == code_only(src)


def test_an_include_header_name_is_not_a_string_literal_and_survives():
    """`#include "HAP_perf.h"` is a header-name token, not a string literal:
    no escapes, no concatenation, and it cannot be used to hide code because it
    has to name a file that exists for the translation unit to compile. So it
    survives `code_only`, and test_skel_dispatch_source.py's check that the SDK
    header is really INCLUDED (rather than named in a comment) needs no escape
    hatch."""
    src = (
        '#include "HAP_perf.h"\n'
        "#include <string.h>\n"
        '  #  include "skel_internal.h"\n'
        'const char *s = "HAP_perf.h";\n'
    )
    out = code_only(src)
    assert len(out) == len(src)
    assert '#include "HAP_perf.h"' in out
    assert "#include <string.h>" in out
    assert '#  include "skel_internal.h"' in out
    # But a plain literal that merely SPELLS a header name is still blanked --
    # the exemption is for the directive, not for the text.
    assert out.count("HAP_perf.h") == 1


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


# ==============================================================================
# MECHANISM (A): the payload of a log line is not code. `assert "TOKEN" in
# body` must not be satisfiable by `FARF(HIGH, "TOKEN")` with the real code
# deleted. Reproduced on skel_bufs.c (`b->base = 0;` demoted to a FARF printing
# that text, `b->base = m->base;` deleted) and on skel_dispatch.c (the
# total_size guard logging HEXLIB_DSP_ERR_TRUNCATED and falling through).
# ==============================================================================


def test_function_body_blanks_a_log_lines_payload_by_default():
    src = (
        "int guard(struct buf *b) {\n"
        '    FARF(HIGH, "b->base = 0; return HEXLIB_DSP_ERR_UNMAPPED;");\n'
        "    return 0;\n"
        "}\n"
    )
    body = function_body(src, "guard")
    assert "HEXLIB_DSP_ERR_UNMAPPED" not in body, (
        "a status constant named only inside a format string must not satisfy "
        "a payload check on the body"
    )
    assert not re.search(r"return\s+HEXLIB_DSP_ERR_\w+\s*;", body)
    assert not re.search(r"b->base\s*=\s*0\s*;", body), (
        "an assignment spelled out inside a log message is not an assignment"
    )
    # The call itself is still visible -- only its payload is gone, so a check
    # that the LOGGING happens is still possible.
    assert "FARF(HIGH," in body
    assert "return 0;" in body
    assert len(body) == len(function_body(src, "guard", strip=False))


def test_block_from_blanks_a_log_lines_payload_by_default():
    text = (
        "if (hdr.total_size != len) {\n"
        '    FARF(ERROR, "hexlib: HEXLIB_DSP_ERR_TRUNCATED size mismatch");\n'
        "}\n"
    )
    block = block_from(text, text.index(")"))
    assert "HEXLIB_DSP_ERR_TRUNCATED" not in block, (
        "the guard block must not be able to report a status by logging its "
        "name -- that is the fall-through mutation this pins"
    )
    assert "FARF(ERROR," in block


def test_block_after_call_blanks_a_log_lines_payload_by_default():
    body = (
        "int rc = real_call(a, b);\n"
        "if (rc != 0) {\n"
        '    FARF(ERROR, "returning return HEXLIB_DSP_ERR_INTERNAL; now");\n'
        "    rc = 0;\n"
        "}\n"
    )
    block = block_after_call(body, "real_call")
    assert not re.search(r"return\s+HEXLIB_DSP_ERR_\w+\s*;", block), (
        "a status named in a log message must not count as propagating it"
    )
    assert "rc = 0;" in block


# ==============================================================================
# MECHANISM (B): a `}` inside a literal must not truncate a brace-depth slice.
# Reproduced on skel_dispatch.c: `hexlib_read_pcycle` reverted to
# `__asm__("%0 = c15:14")` hidden behind `FARF(HIGH, "pcycle }")`, which made
# every "this must NOT appear here" check in the wrapper look at a fragment
# ending at the literal's brace. 12 passed.
# ==============================================================================


def test_function_body_is_not_truncated_by_a_closing_brace_in_a_literal():
    src = (
        "static uint64_t read_pcycle(void) {\n"
        "    uint64_t v = 0;\n"
        '    FARF(HIGH, "hexlib: pcycle }");\n'
        '    __asm__ __volatile__("%0 = c15:14" : "=r"(v));\n'
        "    return v;\n"
        "}\n"
    )
    body = function_body(src, "read_pcycle")
    assert "__asm__" in body, (
        "code after a literal containing `}` must still be inside the body -- "
        "otherwise every negative check on this function is answered by a "
        "fragment that stops at the literal"
    )
    assert "return v;" in body
    assert body.rstrip().endswith("}")
    assert body.count("{") == 1 and body.count("}") == 1, (
        "the literal's brace must have been blanked, not counted"
    )


def test_a_brace_in_a_char_literal_does_not_truncate_a_body_either():
    """The single-quoted form of the same thing -- and the one shape that can
    turn up in real code without anybody trying (`if (c == '}')`)."""
    src = (
        "int f(char c) {\n"
        "    if (c == '}') return 1;\n"
        "    return 0;\n"
        "}\n"
    )
    body = function_body(src, "f")
    assert "return 0;" in body
    assert body.count("}") == 1


def test_block_from_is_not_truncated_by_a_closing_brace_in_a_literal():
    text = (
        "if (ctx->vtcm_needs_release) {\n"
        '    FARF(HIGH, "reclaim }");\n'
        "    hexlib_vtcm_release(ctx);\n"
        "}\n"
    )
    block = block_from(text, text.index(")"))
    assert "hexlib_vtcm_release(ctx);" in block
    assert block.count("{") == 1 and block.count("}") == 1


# ==============================================================================
# MECHANISM (C): a `{` inside a literal must not extend a block past its real
# end. The mirror image of (B), and the one that destroys scoping: an
# `if`-block check would see the whole rest of the function.
# ==============================================================================


def test_block_from_is_not_extended_by_an_opening_brace_in_a_literal():
    text = (
        "if (x) {\n"
        '    FARF(HIGH, "entering {");\n'
        "    inside_the_block();\n"
        "}\n"
        "after_the_block();\n"
        "if (y) {\n"
        "    return HEXLIB_DSP_ERR_INTERNAL;\n"
        "}\n"
    )
    block = block_from(text, text.index(")"))
    assert "inside_the_block();" in block
    assert "after_the_block();" not in block, (
        "a `{` inside a literal must not make the block swallow the code "
        "after it -- that is scoping destroyed, and the whole reason these "
        "checks are block-scoped rather than function-wide"
    )
    assert "HEXLIB_DSP_ERR_INTERNAL" not in block
    assert block.count("{") == 1 and block.count("}") == 1


def test_block_after_call_is_not_derailed_by_a_literal_naming_the_call():
    """A literal that mentions the call by name, with its own parens and its
    own `{`, sits BEFORE the real call. With literals preserved, the call-site
    search lands inside the format string, the paren-walk closes on the
    literal's own `)`, and the block search picks up the literal's `{` -- so
    the block returned has nothing to do with checking the call's result."""
    body = (
        '    FARF(ERROR, "real_call() failed { ");\n'
        "    int rc = real_call(a, b);\n"
        "    if (rc != 0) {\n"
        "        return HEXLIB_DSP_ERR_INTERNAL;\n"
        "    }\n"
    )
    block = block_after_call(body, "real_call")
    assert "HEXLIB_DSP_ERR_INTERNAL" in block
    assert block.count("{") == 1 and block.count("}") == 1


# ==============================================================================
# The escape hatch, end to end: `keep_strings=True` returns literal text but
# must NOT move a boundary, because boundaries are always found in the fully
# blanked view.
# ==============================================================================


def test_keep_strings_returns_literals_without_moving_any_boundary():
    src = (
        "static void usage(void) {\n"
        '    printf("  --unmapped }\\n");\n'
        "    printf(\"  --coherency-check {\\n\");\n"
        "    return;\n"
        "}\n"
    )
    body = function_body(src, "usage", keep_strings=True)
    assert "--unmapped" in body and "--coherency-check" in body, (
        "the escape hatch must actually hand back the literal text"
    )
    assert "return;" in body, (
        "and the literals' own braces must still not truncate or extend the "
        "slice -- boundaries come from the fully blanked view either way"
    )
    # Same boundaries as the default view, so an offset found in one is valid
    # in the other. This is what lets a consumer locate a flag string in the
    # literal-bearing view and brace-slice the block in the blanked one.
    assert len(body) == len(function_body(src, "usage"))
    assert len(body) == len(function_body(src, "usage", strip=False))


def test_no_test_file_carries_its_own_private_copy_of_the_slicer():
    """THE CONSOLIDATION CLAIM, MADE SELF-ENFORCING RATHER THAN PROMISED.
    csource.py's docstring asserted the consolidation was complete while two
    private copies were still live -- a claim about the codebase written in
    prose, which is exactly the kind of thing that rots silently. This checks
    it instead.

    A private copy is a `def` of one of these names in any hexlib/tests module
    other than csource.py itself. An `import ... as _function_body` alias is
    not a copy and is the intended usage, so only `def` is matched.
    `strip_comments` stays on this list although `csource` no longer exports it:
    it was the name of the comments-only transformation, and a test file
    growing its own `_strip_comments` again is the same regression whether or
    not the shared module still has that name.
    `_macro_body` in test_host_source.py is deliberately excluded: it slices a
    backslash-continued `#define`, which brace counting cannot do, and its own
    docstring says why it is a narrowly-scoped sibling rather than a fourth
    slicer."""
    shared = ("strip_comments", "code_only", "code_only_keeping_strings",
              "calls", "function_body", "block_from", "block_after_call")
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
    constant or a call named only in a comment (or in a log message) cannot
    satisfy (or trip) a file-wide check."""
    src = '/* calls HAP_mmap() here */\nint f(void) { return 0; }\n'
    assert "HAP_mmap" not in code_only(src)
    assert len(code_only(src)) == len(src)
    assert "int f(void) { return 0; }" in code_only(src)

    logged = 'int f(void) { FARF(ERROR, "HAP_mmap failed"); return 0; }\n'
    assert "HAP_mmap" not in code_only(logged), (
        "a whole-file fixture must not be able to satisfy a presence check "
        "with a log message either"
    )
    assert len(code_only(logged)) == len(logged)

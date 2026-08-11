# hexlib/tests/csource.py
"""Shared, comment-aware C source slicing for the source-assertion test
files (test_host_source.py, test_skel_bufs_source.py,
test_skel_vtcm_source.py, and formerly a fourth copy nowhere -- see below).

WHY THIS EXISTS. Four test files independently grew their own copy (or a
near-copy, in test_skel_vtcm_source.py's `_block_after_call`) of a
brace-counting function-body slicer built on the regex
`name\\s*\\([^;{]*\\)\\s*\\{`. That regex is COMMENT-BLIND: it is matched
directly against the raw source text, so if the function's own NAME happens
to appear in a comment -- in prose, e.g. "... fails hexlib_drv_init()
rather than being silently tolerated ..." -- ahead of the real definition,
and that comment sits close enough to some unrelated `{` with nothing but
whitespace in between, the match lands on the WRONG brace and the slicer
returns the WRONG function's body (or a fragment of neither). A test built
on top of that slice would then pass or fail based on code it never
intended to inspect -- an assertion that looks like it verifies a guarantee
while actually checking something else. This bit for real once (see
driver.c's own "PLACED AFTER THE INIT ROUTINE ABOVE, DELIBERATELY" comment,
which documents a comment block being physically MOVED to dodge exactly this
regex, rather than the regex being fixed). This module is the fix: extraction
is comment-aware, so future comments can be placed for readability, not to
avoid confusing a test.

HOW. `strip_comments` produces a same-LENGTH copy of the source with every
`/* ... */` and `// ...` comment blanked out (replaced by spaces, newlines
kept so line numbers do not shift). All matching -- finding a function's
signature, counting brace depth, finding the next `{` from some offset, or
locating a call site -- is done against this blanked copy. Because blanking
preserves length exactly, an offset computed against the blanked copy is
valid against the ORIGINAL source too, so every function here returns a
slice of the REAL text (comments included, for any comment that is
genuinely inside the block being extracted) even though comments could not
influence WHERE that slice's boundaries were found.

STRING LITERAL CAVEAT. `strip_comments` also recognizes `"..."` and `'...'`
literals and leaves them untouched (does not blank them, does not let a
`/*`/`//` INSIDE one be mistaken for a real comment start) -- this matters
because this project's C deliberately has FARF/fprintf format strings
containing things like `%p` and multi-word English sentences, though not,
at present, an actual `//` or `/*` substring inside a string literal
anywhere in the files these tests read. What it does NOT handle: escaped
quotes are handled (a backslash-escaped quote inside a string does not end
it), but a backslash-newline line continuation inside a literal is not, and
a malformed/unterminated literal will make the regex consume everything up
to the next quote of the same kind, wherever that is. Both are exotic enough,
and absent from this project's straight-line C, that handling them is not
worth the complexity here. This is a test helper for known, checked-in
source files, not a general C preprocessor.

NOT A GENERAL C PARSER. No handling of trigraphs, raw string edge cases,
`#if 0`-disabled code (see skel_bufs.c's own `#if __HVX_ARCH__ > 73` --
brace-depth counting still works there because a whole preprocessor
`#if`/`#else`/`#endif` block in this codebase's style always has matching
braces on both sides), or anything else beyond what this project's own
conventionally-formatted C actually does. Good enough for that; nothing more
is claimed.
"""
import re

# Matches, in priority order at any given position: a block comment, a line
# comment, a double-quoted string literal, or a single-quoted character
# literal. `re.sub` scans left to right for the next position at which ANY
# alternative matches, so a `"` or `'` that starts a real literal is matched
# as a literal (and left alone) rather than having some `//`/`/*` inside it
# mistaken for a comment -- the literal is consumed as one token, so nothing
# inside it is considered separately.
_TOKEN = re.compile(
    r"/\*.*?\*/"
    r"|//[^\n]*"
    r'|"(?:\\.|[^"\\])*"'
    r"|'(?:\\.|[^'\\])*'",
    re.DOTALL,
)


def _blank(m):
    text = m.group(0)
    if text[0] in "\"'":
        return text  # a string/char literal: leave it exactly as-is
    # a comment: blank it out, keeping newlines so line numbers don't shift
    return "".join(ch if ch == "\n" else " " for ch in text)


def strip_comments(src):
    """Return a same-length copy of `src` with every `/* ... */` and
    `// ...` comment replaced by whitespace (newlines preserved), and every
    string/char literal left untouched. See the module docstring for the
    string-literal caveat and what this deliberately does not handle.

    Because the result is the same length as `src`, an offset found in the
    result is valid as an offset into `src` too -- that is the whole point:
    callers match against this, then slice the ORIGINAL text."""
    return _TOKEN.sub(_blank, src)


def function_body(src, name):
    """Slice one C function's definition -- from its own signature through
    the matching closing brace -- out of `src`, by simple brace-depth
    counting. Good enough for this project's straight-line C; not a general
    C parser.

    Comment-aware: the signature search and the brace-depth count both run
    against `strip_comments(src)`, so a comment that merely mentions `name`
    in prose, or that contains a stray brace, cannot derail the match onto
    the wrong function. The returned text is sliced from the ORIGINAL
    `src` at the same offsets, so the caller sees real code -- including any
    comment that is genuinely inside the extracted function's own body."""
    matching = strip_comments(src)
    m = re.search(rf"\b{re.escape(name)}\s*\([^;{{]*\)\s*\{{", matching)
    assert m, f"could not find the definition of {name}() in the source"
    start = m.end() - 1  # position of the opening brace
    depth = 0
    for i in range(start, len(matching)):
        if matching[i] == "{":
            depth += 1
        elif matching[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
    raise AssertionError(f"unbalanced braces while slicing {name}()")


def block_from(text, pos):
    """From `pos`, find the next `{` and return the brace-matched block it
    opens (inclusive). Generalizes `function_body`'s closing half to an
    arbitrary starting offset, so one specific `if (...) { ... }` can be
    isolated instead of just checking "somewhere in the rest of the
    function" -- which a later, unrelated `return` statement could satisfy
    by accident.

    Comment-aware for the same reason as `function_body`: the brace search
    and depth count run against `strip_comments(text)`, so a comment between
    `pos` and the real block (or inside it) cannot supply a spurious
    `{`/`}` and throw off the match. `pos` and the returned slice's offsets
    both refer to the ORIGINAL `text`."""
    matching = strip_comments(text)
    brace = matching.index("{", pos)
    depth = 0
    for i in range(brace, len(matching)):
        if matching[i] == "{":
            depth += 1
        elif matching[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace : i + 1]
    raise AssertionError("unbalanced braces while slicing a block")


def block_after_call(body, call_name):
    """Within a function body, find a call to `call_name` and return the
    text of the nearest brace-delimited block that checks its result --
    either the call sits inside an `if` condition (`if (call(...) != 0) {
    ... }`), or an `if` immediately follows the call as a separate statement
    (`x = call(...); if (!x) { ... }`). Both shapes occur in this project's
    skel_vtcm.c.

    Comment-aware for the same reason as `function_body`/`block_from`:
    locating the call, walking its own parens, checking for a preceding
    `if (`, finding the block, and counting its brace depth are ALL done
    against `strip_comments(body)`, so a comment mentioning `call_name`, or
    containing a stray `if (` or brace, cannot be mistaken for the real call
    site or its guard. `body`'s offsets and the returned slice both refer to
    the ORIGINAL `body`.

    Asserts an `if (` appears between the call and the block, so a stray
    block that has nothing to do with checking the call's result cannot be
    picked up by accident."""
    matching = strip_comments(body)
    m = re.search(rf"\b{re.escape(call_name)}\s*\(", matching)
    assert m, f"no call to {call_name}() found in this function"
    call_start = m.start()

    # Walk the call's own parens to find where its argument list ends --
    # none of this file's calls nest parens, but do it properly anyway.
    depth = 0
    call_end = None
    for i in range(m.end() - 1, len(matching)):
        if matching[i] == "(":
            depth += 1
        elif matching[i] == ")":
            depth -= 1
            if depth == 0:
                call_end = i + 1
                break
    assert call_end is not None, f"unbalanced parens in the call to {call_name}()"

    brace_pos = matching.find("{", call_end)
    assert brace_pos != -1, f"no block follows the call to {call_name}()"

    window = matching[max(0, call_start - 80) : brace_pos]
    assert "if" in window and "(" in window, (
        f"{call_name}()'s result does not appear to be checked by an `if` "
        f"before the block that follows it"
    )

    depth = 0
    for i in range(brace_pos, len(matching)):
        if matching[i] == "{":
            depth += 1
        elif matching[i] == "}":
            depth -= 1
            if depth == 0:
                return body[brace_pos : i + 1]
    raise AssertionError(f"unbalanced braces in the block following {call_name}()")

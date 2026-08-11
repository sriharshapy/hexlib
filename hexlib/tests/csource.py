# hexlib/tests/csource.py
"""Shared, comment-aware C source slicing for the source-assertion test
files: test_host_source.py, test_skel_bufs_source.py,
test_skel_vtcm_source.py, test_skel_dispatch_source.py, test_kernels.py,
test_session_arch_decode.py, test_coherency_lane_classification.py,
test_runtime_wire.py, test_wire_struct_layout.py, test_vtcm_contention.py and
test_device_cycles_assertion.py. Those are all of them -- there is no surviving
private copy of this slicer anywhere in hexlib/tests, and adding one is the
thing this module exists to stop.

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

THE CONSOLIDATION CLAIM, STATED HONESTLY. An earlier version of this
docstring said the consolidation was complete. It was not: a private copy
with a weaker `_strip_comments` (comment text DELETED rather than blanked,
so every offset shifted) survived in test_skel_dispatch_source.py, and a
fourth, weaker still (no comment handling at all, `src.index("void " + name
+ "(")`) survived in test_kernels.py. Both have since been migrated here.
The list at the top of this docstring is the enumeration that replaces the
claim: if a new source-assertion test appears and is not on it, the claim is
false again.

HOW. `code_only` produces a same-LENGTH copy of the source with every
`/* ... */` and `// ...` comment blanked out AND the INTERIOR of every string
and character literal blanked out (replaced by spaces, newlines kept so line
numbers do not shift). All matching -- finding a function's signature, counting
brace depth, finding the next `{` from some offset, or locating a call site --
is done against this blanked copy. Because blanking preserves length exactly,
an offset computed against the blanked copy is valid against the ORIGINAL
source too, so a slice taken at those offsets is valid against either text.

COMMENT-AWARE BOUNDARIES ARE ONLY HALF THE JOB -- THE RETURNED TEXT MATTERS
JUST AS MUCH. This module originally returned a slice of the ORIGINAL,
comment-BEARING source, and its docstrings presented that as a feature ("the
caller sees real code -- including any comment that is genuinely inside the
extracted function's own body"). It was a hole, and it was the exact hole
this module was written to close, one level up: consumers run their PAYLOAD
checks (`x in body`, `body.index(...)`, `re.search(..., body)`) against
whatever is returned, so a mutation could delete a guard and leave the
deleted code behind AS A COMMENT and every assertion about it would still
pass. That was proven, not theorised: replacing skel_bufs.c's unmapped-fd
`return HEXLIB_DSP_ERR_UNMAPPED;` with a `continue` that trusts the host's
fd -- precisely the shared-address-space bug the staged gate exists to catch
-- left test_skel_bufs_source.py reporting 8 passed.

CLOSING THE COMMENT VEHICLE LEFT THE LITERAL VEHICLE WIDE OPEN, AND THAT WAS
THIS MODULE'S FAULT. The version of this file that fixed the comment hole
deliberately returned string and character literals UNTOUCHED, and its own
"STRING LITERAL CAVEAT" reasoned only about a `%p` inside a format string and
about a `/*` inside a literal being misread as a comment start. It never
reckoned with the literal being the same hole in a different vehicle. A
merge-gate review demonstrated three distinct mechanisms, each reproduced
here against the then-current code:

  (A) POSITIVE ASSERTIONS DEFEATED BY A LOG STRING. `assert "TOKEN" in body`
      is satisfied by `FARF(HIGH, "TOKEN")` with the real code deleted. This
      is not hypothetical and never was: three of the test files listed above
      recount, in their OWN docstrings, earlier rounds of exactly this
      FARF-string vector being exploited (test_skel_vtcm_source.py's
      HAP_compute_res_query_VTCM error string, test_skel_bufs_source.py's
      `nbytes`, test_host_source.py's `"hexlib: dlopen(%s) failed"`). Fixing
      those three call sites one at a time, while leaving the SLICER handing
      literal text to every payload check, was treating instances of a defect
      whose cause was here. Reproduced: skel_bufs.c's `b->base = 0;` replaced
      by a FARF printing that same text, with the real `b->base = m->base;`
      deleted -- so the DSP hands kernels whatever address the host wrote --
      left test_skel_bufs_source.py at 8 passed.

  (B) NEGATIVE ASSERTIONS DEFEATED BY TRUNCATING THE SLICE. Brace depth was
      counted over text in which literals survived, so a `}` inside a literal
      decremented the count. `FARF(HIGH, "pcycle }")` inside a function made
      `function_body` return everything up to that `}` and nothing after it,
      so every `X not in body` check was answered by a fragment. Reproduced:
      skel_dispatch.c's `hexlib_read_pcycle` reverted to
      `__asm__("%0 = c15:14")` -- unreadable in a user-mode unsigned PD, which
      is the one thing that wrapper exists to avoid -- hidden after such a
      FARF, left test_skel_dispatch_source.py at 12 passed.

  (C) SCOPING DESTROYED BY A LITERAL `{`. The mirror image: an extra `{`
      inside a literal made `block_from`'s depth count swallow the function
      tail, so an `if`-block "scoped" check saw the rest of the function.

So `code_only` now blanks the INTERIOR of every string and character literal
as well, and every slicer counts braces and parens over that text. The
literals' own delimiters are kept, and blanking is still exactly
length-preserving, which is what keeps every offset valid against the original
file -- the property the whole module is built on.

BOUNDARY FINDING STILL NEEDS THE LITERALS WHOLE, WHICH IS WHY THIS IS ONE
TOKENIZER PASS AND NOT TWO REGEX PASSES. A `//` or `/*` inside a string
literal must not be mistaken for a comment start, so the literal has to be
recognized and consumed as a single token BEFORE anything inside it is
considered -- exactly as before. What changed is only what is written back out
for the token once it has been recognized.

THE ESCAPE HATCH, AND THE THREE PLACES IT IS LEGITIMATE.
`code_only_keeping_strings` blanks comments and leaves literals intact. Use it
only where the CLAIM ITSELF IS ABOUT LITERAL TEXT, and say so at the call
site. In this repo that is: a printed line whose exact wording another test
asserts on (main.c's `PASS (%d values, bit-exact)`), a command-line flag
string main() must recognize, and a path or macro spelling that must or must
not appear (`"libcdsprpc.so"`, `"&_dom=cdsp"`). Everything else -- every
"this guard must be here", every "this constant must not be here" -- goes
through `code_only`, because for those a literal is evidence of nothing.
Boundary finding and brace counting use the fully-blanked text either way, so
asking to keep literals never re-opens (B) or (C); it re-opens only (A), and
only for the one check that asked.

`#include "hdr.h"` IS NOT A STRING LITERAL AND IS NOT BLANKED. In C a
header-name is its own token class -- no escape processing, no concatenation
-- and it cannot be written by a mutation trying to hide code, because it has
to name a file that exists for the translation unit to compile at all. So
`#include "HAP_perf.h"` survives `code_only` and
test_skel_dispatch_source.py's check that the SDK header is really included
(not merely referred to in prose) keeps working without an escape hatch.
`#include <hdr.h>` was never affected.

FUNCTION SCOPE IS NOT THE SAME AS REACHABILITY, AND `calls()` IS WHAT THIS
MODULE OFFERS ABOUT THAT. Every negative check built on these slicers is
scoped to one function or one block, so a forbidden construct can be moved one
call level away and the check sees nothing -- no comment and no literal
required. Proven: moving `__asm__("%0 = c15:14")` out of `hexlib_read_pcycle`
into a new `hexlib_raw_pcycle()` helper it calls left
test_skel_dispatch_source.py at 12 passed. `calls()` (see its own docstring)
lets a test state the exhaustive set of callees a block has, so a new helper
is a failure by construction; a whole-FILE negative is the other half, for a
construct that must not exist anywhere at all.

WHAT THIS STILL DOES NOT HANDLE. Escaped quotes are handled (a
backslash-escaped quote inside a literal does not end it), but a
backslash-newline line continuation inside a literal is not, and a
malformed/unterminated literal will make the regex consume everything up to
the next quote of the same kind, wherever that is. Adjacent literals that C
would concatenate are blanked individually, which is the same answer.
Both remaining gaps are exotic enough, and absent from this project's
straight-line C, that handling them is not worth the complexity here. This is
a test helper for known, checked-in source files, not a general C
preprocessor.

NOT A GENERAL C PARSER. No handling of trigraphs, raw string edge cases,
`#if 0`-disabled code (see skel_bufs.c's own `#if __HVX_ARCH__ > 73` --
brace-depth counting still works there because a whole preprocessor
`#if`/`#else`/`#endif` block in this codebase's style always has matching
braces on both sides), or anything else beyond what this project's own
conventionally-formatted C actually does. Good enough for that; nothing more
is claimed.
"""
import re

# Matches, in priority order at any given position: a `#include "hdr.h"`
# header-name (NOT a string literal -- see the docstring), a block comment, a
# line comment, a double-quoted string literal, or a single-quoted character
# literal. `re.sub` scans left to right for the next position at which ANY
# alternative matches, so a `"` or `'` that starts a real literal is matched
# as a literal rather than having some `//`/`/*` inside it mistaken for a
# comment -- the literal is consumed as one token, so nothing inside it is
# considered separately. That is what makes it safe to blank a literal's
# INTERIOR: the decision to blank is made about a token already known to be a
# literal, not about the characters inside one.
_TOKEN = re.compile(
    r'^[ \t]*\#[ \t]*include[ \t]*"[^"\n]*"'
    r"|/\*.*?\*/"
    r"|//[^\n]*"
    r'|"(?:\\.|[^"\\])*"'
    r"|'(?:\\.|[^'\\])*'",
    re.DOTALL | re.MULTILINE,
)


def _spaces(text):
    """Same-length whitespace, newlines kept so line numbers do not shift."""
    return "".join("\n" if ch == "\n" else " " for ch in text)


def _blank(text, blank_strings):
    if text.lstrip().startswith("#"):
        return text  # `#include "hdr.h"` -- a header-name, not a literal
    if text[0] in "\"'":
        if not blank_strings:
            return text
        # Keep the delimiters (so the token is still visibly a literal, and a
        # check that a literal EXISTS at all still works) and blank the
        # interior. Length is preserved either way.
        return text[0] + _spaces(text[1:-1]) + text[-1]
    return _spaces(text)  # a comment


def code_only(text):
    """Return a same-length copy of `text` with every `/* ... */` and `// ...`
    comment blanked out AND the interior of every string and character literal
    blanked out (newlines preserved throughout, so line numbers do not shift).
    `#include "hdr.h"` header-names are left alone -- see the module docstring.

    THIS IS THE ONE TO USE. It produces text that a payload check ("this call
    must be here", "this constant must not be here") can safely be run
    against, because nothing in it came from a comment and nothing in it came
    from a log message, a format string or a CLI-flag string. Both of those
    were proven vehicles for satisfying an assertion while deleting the code it
    was about; see the module docstring for the mutations.

    Because the result is the same length as `text`, an offset found in the
    result is valid as an offset into `text` too -- that is the whole point:
    boundaries found here can be used to slice either version.

    Idempotent, and interchangeable in either order with
    `code_only_keeping_strings`."""
    return _TOKEN.sub(lambda m: _blank(m.group(0), True), text)


def code_only_keeping_strings(text):
    """`code_only`, but string and character literals are left INTACT. The
    deliberate escape hatch, for the few checks whose subject IS literal text:
    a printed line whose exact wording is asserted elsewhere, a command-line
    flag string, a path or macro spelling that must (or must not) appear.

    Say why at the call site. For anything else this is the wrong function: a
    token inside a format string is not evidence that the code it names is
    still there, which is the whole finding this module was rewritten for.

    Same same-length guarantee. Note that the slicers below always count
    braces and parens over `code_only` text regardless, so passing a
    literal-bearing fixture cannot move a boundary."""
    return _TOKEN.sub(lambda m: _blank(m.group(0), False), text)


# Keywords and type specifiers that are followed by `(` in this project's C
# without being a call. `sizeof(T)`, `if (`, `while (`, `for (`, `switch (`,
# `return (x)`, and a cast's own type name in `(uint64_t) (uintptr_t) p`.
_NOT_CALLEES = frozenset("""
    if else for while do switch case return goto sizeof
    void char short int long float double signed unsigned _Bool
    struct union enum const volatile static inline extern register typedef
    defined
""".split())

_CALLEE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def calls(fragment):
    """The set of identifiers that appear as the CALLEE of a call in
    `fragment`, over `code_only` text. `k->fn(&a)` contributes `fn`.

    THIS EXISTS TO CLOSE THE "MOVE IT ONE CALL LEVEL AWAY" ESCAPE. A
    function-scoped negative check -- "this block must NOT call X" -- is
    satisfied by a mutation that puts X in a new one-line helper and calls the
    helper instead. Nothing about the block's own text changed except the name,
    so no `X not in block` check can see it. That was demonstrated on
    skel_dispatch.c: moving `__asm__("%0 = c15:14")` into a new
    `hexlib_raw_pcycle()` left test_skel_dispatch_source.py at 12 passed, with
    no comment or literal trick involved at all.

    Two answers to that, and both are used in this repo. Where the forbidden
    construct must not exist ANYWHERE, pair the scoped negative with a
    whole-FILE one (the raw register read). Where it must not be REACHED from
    one specific block, assert the exhaustive set of things that block calls --
    an unexpected callee is then a failure by construction, whatever it is
    named, because a helper that hides the construct still has to be called
    from somewhere.

    Not a call graph: this is one level, textual, and deliberately so. A
    two-level indirection would defeat it, and the whole-file pairing is what
    covers that case."""
    return {
        m.group(1)
        for m in _CALLEE.finditer(code_only(fragment))
        if m.group(1) not in _NOT_CALLEES
    }


def _view(src, strip, keep_strings):
    """The text a slicer RETURNS a slice of, given its two flags."""
    if not strip:
        return src  # exactly as given, comments and literals included
    if keep_strings:
        return code_only_keeping_strings(src)
    return code_only(src)


def function_body(src, name, strip=True, keep_strings=False):
    """Slice one C function's definition -- from its own opening brace through
    the matching closing brace -- out of `src`, by simple brace-depth
    counting. Good enough for this project's straight-line C; not a general
    C parser.

    Comment- AND literal-aware in BOTH directions. The signature search and the
    brace-depth count run against `code_only(src)`, so a comment that merely
    mentions `name` in prose cannot derail the match onto the wrong function,
    and neither a comment nor a string literal can supply a stray `{`/`}` that
    truncates the body or swallows the next function. And with `strip=True`
    (the default) the text RETURNED is `code_only` text too, so a payload check
    the caller runs against it cannot be satisfied by a comment inside the body
    or by a log message -- see the module docstring for the proven mutations
    behind each half.

    `keep_strings=True` returns the comment-blanked but literal-BEARING slice,
    for a caller whose claim is genuinely about literal text; boundaries are
    unaffected. `strip=False` returns the slice of `src` exactly as given.
    Either way, say why at the call site."""
    matching = code_only(src)
    m = re.search(rf"\b{re.escape(name)}\s*\([^;{{]*\)\s*\{{", matching)
    assert m, f"could not find the definition of {name}() in the source"
    out = _view(src, strip, keep_strings)
    start = m.end() - 1  # position of the opening brace
    depth = 0
    for i in range(start, len(matching)):
        if matching[i] == "{":
            depth += 1
        elif matching[i] == "}":
            depth -= 1
            if depth == 0:
                return out[start : i + 1]
    raise AssertionError(f"unbalanced braces while slicing {name}()")


def block_from(text, pos, strip=True, keep_strings=False):
    """From `pos`, find the next `{` and return the brace-matched block it
    opens (inclusive). Generalizes `function_body`'s closing half to an
    arbitrary starting offset, so one specific `if (...) { ... }` can be
    isolated instead of just checking "somewhere in the rest of the
    function" -- which a later, unrelated `return` statement could satisfy
    by accident.

    Comment- and literal-aware for the same reasons as `function_body`, in both
    directions: the brace search and depth count run against
    `code_only(text)`, so neither a comment nor a string literal between `pos`
    and the real block (or inside it) can supply a spurious `{`/`}` and throw
    off the match, and with `strip=True` (the default) the text returned is
    blanked so a payload check against it cannot be satisfied by a comment or a
    log message inside the block. `pos` is an offset into `text` and is valid
    against every version, since blanking preserves length.

    See `function_body` for `strip` and `keep_strings`."""
    matching = code_only(text)
    out = _view(text, strip, keep_strings)
    brace = matching.index("{", pos)
    depth = 0
    for i in range(brace, len(matching)):
        if matching[i] == "{":
            depth += 1
        elif matching[i] == "}":
            depth -= 1
            if depth == 0:
                return out[brace : i + 1]
    raise AssertionError("unbalanced braces while slicing a block")


def block_after_call(body, call_name, strip=True, keep_strings=False):
    """Within a function body, find a call to `call_name` and return the
    text of the nearest brace-delimited block that checks its result --
    either the call sits inside an `if` condition (`if (call(...) != 0) {
    ... }`), or an `if` immediately follows the call as a separate statement
    (`x = call(...); if (!x) { ... }`). Both shapes occur in this project's
    skel_vtcm.c.

    Comment- and literal-aware for the same reasons as
    `function_body`/`block_from`: locating the call, walking its own parens,
    checking for a preceding `if (`, finding the block, and counting its brace
    depth are ALL done against `code_only(body)`, so neither a comment nor a
    string literal mentioning `call_name`, or containing a stray paren or
    brace, can be mistaken for the real call site or its guard, or truncate the
    block. With `strip=True` (the default) the returned block is blanked too,
    so the `return <status>` a caller then looks for in it can be neither a
    commented-out one nor one named in a FARF.

    Asserts an `if (` appears between the call and the block, so a stray
    block that has nothing to do with checking the call's result cannot be
    picked up by accident.

    See `function_body` for `strip` and `keep_strings`."""
    matching = code_only(body)
    out = _view(body, strip, keep_strings)
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
                return out[brace_pos : i + 1]
    raise AssertionError(f"unbalanced braces in the block following {call_name}()")

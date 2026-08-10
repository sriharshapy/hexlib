"""A verdict with a non-finite error is a verdict, not a missing verdict.

REGRESSION. `maxerr` was matched as a class of digits and exponent punctuation,
which does not match printf's `inf` or `nan`. The whole line then failed to
parse, and `run_sim` reported "no verdict recovered: the harness never printed
HEXLIB_VERDICT, so nothing was actually checked" -- about a run where the
harness had printed a perfectly good verdict saying the kernel was WRONG.

Found by a real near-miss: adding fp16 bit patterns as 16-bit integers overflows
to inf, and the near-miss was scored INCONCLUSIVE instead of correctly rejected.
It failed safe (never a pass) but it misattributed the cause, and a genuine
kernel that overflowed would have been reported as not having run rather than as
incorrect.
"""
from hexlib import sim


def test_an_inf_error_still_parses_as_a_failing_verdict():
    text = "HEXLIB_VERDICT correct=0 wrong=3539 maxerr=inf\n"
    parsed = sim.parse_verdict(text)
    assert parsed is not None, "an inf error was read as no verdict at all"
    correct, n_wrong, max_err = parsed
    assert correct is False
    assert n_wrong == 3539
    assert max_err == float("inf")


def test_a_nan_error_still_parses():
    parsed = sim.parse_verdict("HEXLIB_VERDICT correct=0 wrong=7 maxerr=nan\n")
    assert parsed is not None
    correct, n_wrong, max_err = parsed
    assert correct is False
    assert n_wrong == 7
    assert max_err != max_err or max_err == float("inf")  # NaN or coerced


def test_an_inf_verdict_is_counted_as_one_verdict():
    """run_sim requires EXACTLY one verdict. If the inf line did not match the
    counter, a single run would look like zero verdicts."""
    assert sim.count_verdicts("HEXLIB_VERDICT correct=0 wrong=1 maxerr=inf\n") == 1


def test_ordinary_finite_errors_are_unaffected():
    for value, expect in (("0", 0.0), ("0.00195312", 0.00195312), ("1.5e-07", 1.5e-07)):
        parsed = sim.parse_verdict(
            f"HEXLIB_VERDICT correct=1 wrong=0 maxerr={value}\n"
        )
        assert parsed is not None, value
        assert parsed[0] is True
        assert parsed[1] == 0
        assert parsed[2] == expect


def test_a_genuinely_absent_verdict_is_still_None():
    """The fail-closed behaviour this must not weaken: a run that printed no
    verdict has checked nothing, and that is a failure."""
    assert sim.parse_verdict("Done!\n\tTotal: Insns=123 Pcycles=456\n") is None
    assert sim.parse_verdict("") is None


def test_an_unparseable_error_field_does_not_lose_the_verdict():
    """correct= and wrong= are what decide the gate. Losing the whole line over
    an unreadable magnitude is the bug this file exists for."""
    parsed = sim.parse_verdict("HEXLIB_VERDICT correct=0 wrong=2 maxerr=garbage\n")
    assert parsed is not None
    assert parsed[0] is False
    assert parsed[1] == 2

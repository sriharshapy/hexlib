"""`movement_only` changes what counts as proof of acceleration. It must not
become a way to opt out of being checked.

The default rule -- vector ARITHMETIC must appear, or bytes merely passed through
the vector unit while the real work stayed scalar -- is right for a compute
kernel and wrong for a permutation, which has no arithmetic to vectorise. The
encoder has 49 such ops. These tests pin the narrow shape of the exception:

  * a movement-only kernel must still prove it used the vector unit;
  * a scalar one still fails;
  * and the flag does nothing to any of the other gate conditions.
"""
from hexlib.anticheat import AccelProof
from hexlib.verify import NEARMISS_INCONCLUSIVE, NEARMISS_REJECTED, VerifyReport


def _report(**kw) -> VerifyReport:
    base = dict(
        task_id="t",
        correct=True,
        kernel_cycles=100,
        accel=AccelProof(used_hvx=True, used_hvx_compute=True, used_hmx=False),
        nearmiss={"nearmiss_a.c": NEARMISS_REJECTED},
        toolchain_version="19.0.04",
        sdk_version="6.4.0.2",
        host="h",
        timestamp="t",
    )
    base.update(kw)
    return VerifyReport(**base)


_VECTOR_NO_ARITH = AccelProof(used_hvx=True, used_hvx_compute=False, used_hmx=False)
_SCALAR = AccelProof(used_hvx=False, used_hvx_compute=False, used_hmx=False)


def test_vector_moves_without_arithmetic_fail_by_default():
    """The original rule, unchanged for every compute kernel."""
    assert _report(accel=_VECTOR_NO_ARITH).gate_passed() is False


def test_vector_moves_without_arithmetic_pass_when_declared_movement_only():
    assert _report(accel=_VECTOR_NO_ARITH, movement_only=True).gate_passed() is True


def test_a_scalar_movement_kernel_still_fails():
    """The flag drops the arithmetic requirement, NOT the vector requirement. A
    transpose written as a scalar copy loop is not accelerated and must not
    pass."""
    assert _report(accel=_SCALAR, movement_only=True).gate_passed() is False


def test_movement_only_does_not_excuse_being_wrong():
    assert (
        _report(accel=_VECTOR_NO_ARITH, movement_only=True, correct=False).gate_passed()
        is False
    )


def test_movement_only_does_not_excuse_a_missing_near_miss():
    assert (
        _report(accel=_VECTOR_NO_ARITH, movement_only=True, nearmiss={}).gate_passed()
        is False
    )


def test_movement_only_does_not_excuse_an_inconclusive_near_miss():
    assert (
        _report(
            accel=_VECTOR_NO_ARITH,
            movement_only=True,
            nearmiss={"nearmiss_a.c": f"{NEARMISS_INCONCLUSIVE}: did not run"},
        ).gate_passed()
        is False
    )


def test_the_claim_is_printed_so_a_reviewer_sees_it():
    """A weaker requirement that is invisible in the result table is a weaker
    requirement nobody reviews."""
    table = _report(accel=_VECTOR_NO_ARITH, movement_only=True).to_table()
    assert "movement-only" in table
    assert "movement-only" not in _report().to_table()


def test_a_compute_kernel_is_unaffected_by_the_flags_existence():
    assert _report().gate_passed() is True
    assert (
        _report(
            accel=AccelProof(used_hvx=True, used_hvx_compute=False, used_hmx=True)
        ).gate_passed()
        is True
    )

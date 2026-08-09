import pytest

from hexlib.result import Err, Measurements, Ok, is_ok


def test_ok_requires_measurements():
    """An Ok cannot be built without measurements — this is the structural form
    of the QDC false-pass fix: 'job completed, no results' cannot become success."""
    with pytest.raises(TypeError):
        Ok(None)


def test_ok_carries_measurements():
    m = Measurements(
        kernel_cycles=2311,
        toolchain_version="19.0.04",
        sdk_version="6.4.0.2",
        host="testbox",
        timestamp="2026-08-09T00:00:00Z",
    )
    r = Ok(m)
    assert is_ok(r)
    assert r.unwrap().kernel_cycles == 2311


def test_err_is_not_ok_and_unwrap_raises():
    r = Err("no results recovered", "sim produced no HEXLIB_KCYCLES line")
    assert not is_ok(r)
    with pytest.raises(RuntimeError, match="no results recovered"):
        r.unwrap()


def test_measurements_reject_missing_cycles():
    with pytest.raises(TypeError):
        Measurements(
            kernel_cycles=None,
            toolchain_version="19.0.04",
            sdk_version="6.4.0.2",
            host="testbox",
            timestamp="2026-08-09T00:00:00Z",
        )

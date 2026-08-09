# hexlib/tests/test_contract.py
import json
import os
import pathlib

from hexlib import contract, kerneldir as kd

GOOD_TABLE = """### hexlib verify — rmsnorm_fp16

| gate | result |
|---|---|
| correct | PASS |
| kernel_cycles | 9798 |
| accel (ELF-proven) | hvx, hvx-compute |
| near-miss `nearmiss_x.c` | correctly rejected |
| **gate** | **PASS** |

target `v75` · toolchain `19.0.04` · SDK `6.4.0.2` · host `me@box` · `2026-08-09T00:00:00Z`
"""


def _kernel(tmp_path, table=GOOD_TABLE, name="rmsnorm_fp16"):
    d = tmp_path / name
    d.mkdir()
    for f in kd.REQUIRED_FILES:
        (d / f).write_text("")
    (d / "nearmiss_x.c").write_text("")
    (d / "spec.json").write_text(json.dumps({"task_id": name, "dtype": "fp16"}))
    if table is not None:
        # Explicit UTF-8: GOOD_TABLE contains U+00B7 and U+2014, and
        # contract.py reads RESULT.md as UTF-8 (matching verify.py's own write
        # encoding). Path.write_text()'s default is the locale encoding, which
        # on Windows (cp1252) mangles those characters on write, not read.
        (d / "RESULT.md").write_text(table, encoding="utf-8")
    return str(d)


def test_parses_a_good_table():
    parsed = contract.parse_result_table(GOOD_TABLE)
    assert parsed["toolchain"] == "19.0.04"
    assert parsed["gate"] == "PASS"


def test_valid_kernel_with_result_passes():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path
        assert contract.check_kernel_contract(_kernel(Path(tmp))) == []


def test_missing_result_file_is_rejected(tmp_path):
    problems = contract.check_kernel_contract(_kernel(tmp_path, table=None))
    assert any("RESULT.md" in p for p in problems)


def test_unparseable_result_is_rejected(tmp_path):
    problems = contract.check_kernel_contract(_kernel(tmp_path, table="looks fine to me"))
    assert any("could not be parsed" in p for p in problems)


def test_wrong_toolchain_version_is_rejected(tmp_path):
    """Cycle numbers from a different toolchain are not comparable."""
    bad = GOOD_TABLE.replace("19.0.04", "18.0.00")
    problems = contract.check_kernel_contract(_kernel(tmp_path, table=bad))
    assert any("18.0.00" in p and "19.0.04" in p for p in problems)


def test_failing_gate_is_rejected(tmp_path):
    bad = GOOD_TABLE.replace("| **gate** | **PASS** |", "| **gate** | **FAIL** |")
    problems = contract.check_kernel_contract(_kernel(tmp_path, table=bad))
    assert any("gate" in p.lower() and "FAIL" in p for p in problems)


def test_structural_problems_are_also_reported(tmp_path):
    d = _kernel(tmp_path)
    os.remove(os.path.join(d, "baseline.c"))
    assert any("baseline.c" in p for p in contract.check_kernel_contract(d))


def test_wrong_target_is_rejected(tmp_path):
    bad = GOOD_TABLE.replace("v75", "v73")
    problems = contract.check_kernel_contract(_kernel(tmp_path, table=bad))
    assert any("v73" in p and "v75" in p for p in problems)


def test_binary_garbage_result_is_rejected(tmp_path):
    """A RESULT.md that is pure binary garbage must be rejected, not raise."""
    d = tmp_path / "rmsnorm_fp16"
    d.mkdir()
    for f in kd.REQUIRED_FILES:
        (d / f).write_text("")
    (d / "nearmiss_x.c").write_text("")
    (d / "spec.json").write_text(json.dumps({"task_id": "rmsnorm_fp16", "dtype": "fp16"}))
    (d / "RESULT.md").write_bytes(bytes([0xFF, 0xFE, 0x00, 0x80, 0x81]))
    problems = contract.check_kernel_contract(str(d))
    assert problems != []


def test_result_from_another_kernel_is_rejected(tmp_path):
    """A table copied from a different kernel is evidence about that kernel,
    not this one."""
    problems = contract.check_kernel_contract(
        _kernel(tmp_path, name="softmax_fp16")
    )
    assert any("rmsnorm_fp16" in p and "softmax_fp16" in p for p in problems)


def test_nearmiss_not_in_the_table_is_rejected(tmp_path):
    d = _kernel(tmp_path)
    (pathlib.Path(d) / "nearmiss_added_later.c").write_text("")
    problems = contract.check_kernel_contract(d)
    assert any("nearmiss_added_later.c" in p for p in problems)


def test_wrongly_accepted_nearmiss_in_the_table_is_rejected(tmp_path):
    bad = GOOD_TABLE.replace("correctly rejected", "WRONGLY ACCEPTED")
    problems = contract.check_kernel_contract(_kernel(tmp_path, table=bad))
    assert any("nearmiss_x.c" in p for p in problems)


def test_corrupted_prose_outside_matched_regions_is_rejected(tmp_path):
    """A RESULT.md can be a syntactically valid, correctly-toolchained PASS
    table -- both regex-anchored regions parse cleanly -- and still be a
    corrupted file: invalid UTF-8 bytes landing in the surrounding prose,
    outside the conditions line and the gate row, must not be silently
    repaired into a pass. Written as raw bytes so the corruption is real,
    not something Python would re-encode away.
    """
    d = tmp_path / "rmsnorm_fp16"
    d.mkdir()
    for f in kd.REQUIRED_FILES:
        (d / f).write_text("")
    (d / "nearmiss_x.c").write_text("")
    (d / "spec.json").write_text(json.dumps({"task_id": "rmsnorm_fp16", "dtype": "fp16"}))
    corrupted = (
        GOOD_TABLE.encode("utf-8")
        + "\nMeasured on the hexagon simulator under the pinned bus model.".encode(
            "utf-8"
        )
        + b"\xff\xfe"
        + " Not a silicon measurement.".encode("utf-8")
    )
    (d / "RESULT.md").write_bytes(corrupted)
    problems = contract.check_kernel_contract(str(d))
    assert problems != []

import json
import os

import pytest

from hexlib import kerneldir as kd


def _write(d, name, text=""):
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def _good_spec():
    return {
        "task_id": "rmsnorm_fp16",
        "dtype": "fp16",
        "caps": [],
        "mechanisms": ["hvx"],
        "params": {"R": 6, "C": 80, "eps": 0.001},
        "expert_kernel_cycles": 9798,
        "tolerance": "hexlib_close_f16",
        "tags": ["rmsnorm", "hvx"],
    }


def _make_kernel(tmp_path, spec=None):
    d = tmp_path / "rmsnorm_fp16"
    d.mkdir()
    for f in kd.REQUIRED_FILES:
        _write(str(d), f)
    _write(str(d), "nearmiss_no_eps.c")
    with open(d / "spec.json", "w", encoding="utf-8") as f:
        json.dump(spec or _good_spec(), f)
    return str(d)


def test_valid_directory_has_no_problems(tmp_path):
    assert kd.validate_dir(_make_kernel(tmp_path)) == []


def test_missing_file_is_reported_by_name(tmp_path):
    d = _make_kernel(tmp_path)
    os.remove(os.path.join(d, "baseline.c"))
    problems = kd.validate_dir(d)
    assert any("baseline.c" in p for p in problems)


def test_a_nearmiss_is_required(tmp_path):
    """A kernel PR must include a plausible-but-wrong variant the harness catches.
    Without one, the harness has not been shown to discriminate."""
    d = _make_kernel(tmp_path)
    os.remove(os.path.join(d, "nearmiss_no_eps.c"))
    assert any("nearmiss" in p for p in kd.validate_dir(d))


def test_nearmiss_files_are_found(tmp_path):
    d = _make_kernel(tmp_path)
    _write(d, "nearmiss_wrong_axis.c")
    assert sorted(os.path.basename(p) for p in kd.nearmiss_files(d)) == [
        "nearmiss_no_eps.c",
        "nearmiss_wrong_axis.c",
    ]


def test_spec_loads(tmp_path):
    spec = kd.load_spec(_make_kernel(tmp_path))
    assert spec.task_id == "rmsnorm_fp16"
    assert spec.caps == []
    assert spec.params["C"] == 80


def test_spec_rejects_unknown_cap(tmp_path):
    bad = _good_spec()
    bad["caps"] = ["hmx", "tensorcore"]
    d = _make_kernel(tmp_path, bad)
    assert any("tensorcore" in p for p in kd.validate_dir(d))


def test_spec_rejects_task_id_not_matching_dir(tmp_path):
    bad = _good_spec()
    bad["task_id"] = "something_else"
    d = _make_kernel(tmp_path, bad)
    assert any("task_id" in p for p in kd.validate_dir(d))


def test_scaffold_creates_a_valid_directory(tmp_path):
    d = kd.scaffold(str(tmp_path), "softmax_fp16")
    assert kd.validate_dir(d) == []
    assert kd.load_spec(d).task_id == "softmax_fp16"


def test_scaffold_refuses_to_overwrite(tmp_path):
    kd.scaffold(str(tmp_path), "softmax_fp16")
    with pytest.raises(FileExistsError):
        kd.scaffold(str(tmp_path), "softmax_fp16")


@pytest.mark.parametrize(
    "dtype",
    [
        "int8", "uint8", "int16", "int32", "uint16",
        "i8", "u8", "i8->i8", "i8->i32",
        "int8->int32", "int32->int8", "uint8xint8->int32",
        "int8+int8->int8", "int8*int8->int8", "int8->int32->int8",
    ],
)
def test_integer_dtypes_are_detected(dtype):
    assert kd.is_integer_dtype(dtype)


@pytest.mark.parametrize(
    "dtype",
    ["fp16", "fp32", "f16", "f32", "float", "bf16", "int8->fp16", "fp16->int8"],
)
def test_float_dtypes_are_not_integer(dtype):
    """A mixed pipeline touching float anywhere may need tolerance."""
    assert not kd.is_integer_dtype(dtype)


def test_integer_kernel_must_be_bit_exact(tmp_path):
    """An i8 path has no reordering and no representation error, so a tolerance
    there hides wrong results rather than accommodating the hardware."""
    spec = _good_spec()
    spec["dtype"] = "int8->int32"
    spec["tolerance"] = "hexlib_close_f16"
    problems = kd.validate_dir(_make_kernel(tmp_path, spec))
    assert any("bit-exact" in p and "int8->int32" in p for p in problems)


def test_integer_kernel_with_exact_tolerance_is_fine(tmp_path):
    spec = _good_spec()
    spec["dtype"] = "int8"
    spec["tolerance"] = kd.EXACT_TOLERANCE
    assert kd.validate_dir(_make_kernel(tmp_path, spec)) == []


def test_float_kernel_may_use_a_tolerance(tmp_path):
    spec = _good_spec()
    spec["dtype"] = "fp16"
    spec["tolerance"] = "hexlib_close_f16"
    assert kd.validate_dir(_make_kernel(tmp_path, spec)) == []


def test_float_kernel_may_also_be_exact(tmp_path):
    """Nothing forbids an fp kernel from being bit-exact if it genuinely is."""
    spec = _good_spec()
    spec["dtype"] = "fp32"
    spec["tolerance"] = kd.EXACT_TOLERANCE
    assert kd.validate_dir(_make_kernel(tmp_path, spec)) == []

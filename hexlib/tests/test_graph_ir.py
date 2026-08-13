from __future__ import annotations

import pytest

from hexlib.graph.ir import DTYPES, Graph, Op, Tensor, nbytes


def _t(name, shape, dtype="fp32", const=False):
    return Tensor(name=name, dtype=dtype, shape=shape, const=const)


def test_tensor_is_frozen():
    t = _t("x", (4, 8))
    with pytest.raises(Exception):
        t.name = "y"


def test_op_coerces_list_inputs_to_tuple():
    """Op coerces list inputs/outputs to tuple and is immune to list mutation."""
    inputs_list = ["x", "y"]
    outputs_list = ["z"]
    op = Op(id=0, kind="add", inputs=inputs_list, outputs=outputs_list, attrs={})

    # Original lists can be mutated without affecting the op
    inputs_list.append("mutated")
    outputs_list[0] = "also_mutated"

    # Op still has the original values as a tuple
    assert op.inputs == ("x", "y")
    assert op.outputs == ("z",)


def test_graph_coerces_list_fields_to_tuple():
    """Graph coerces ops/inputs/outputs lists to tuples and is immune to list mutation."""
    ops_list = [Op(id=0, kind="scale", inputs=("x",), outputs=("y",), attrs={"factor": 2.0})]
    inputs_list = ["x"]
    outputs_list = ["y"]

    x = _t("x", (4,))
    y = _t("y", (4,))
    g = Graph(
        tensors={"x": x, "y": y},
        ops=ops_list,
        inputs=inputs_list,
        outputs=outputs_list,
    )

    # Original lists can be mutated without affecting the graph
    ops_list.append(Op(id=1, kind="scale", inputs=("y",), outputs=("z",), attrs={}))
    inputs_list.append("mutated")
    outputs_list[0] = "also_mutated"

    # Graph still has the original values as tuples
    assert len(g.ops) == 1
    assert g.inputs == ("x",)
    assert g.outputs == ("y",)


def test_tensor_rejects_unknown_dtype():
    with pytest.raises(ValueError) as e:
        _t("x", (4,), dtype="bfloat16")
    assert "bfloat16" in str(e.value)
    assert "x" in str(e.value)


def test_tensor_rejects_nonpositive_dim():
    with pytest.raises(ValueError) as e:
        _t("x", (4, 0))
    assert "x" in str(e.value)


def test_dtypes_are_exactly_these():
    """AN EXHAUSTIVE PIN, DELIBERATELY. A dtype added here has to be carried
    through `wire.DTYPE_ID`, `runner.WIRE_DTYPE` or `WIRE_RAW`, and
    `genentry._CTYPE` before anything can use it -- `test_runtime_wire.py` binds
    those three -- so a silent addition is a dtype the graph accepts and the
    wire cannot carry. Failing here is the intended way to be reminded.

    q8_0 was added on 2026-08-13 and this test caught it, which is the check
    working rather than the check being in the way."""
    assert DTYPES == frozenset({"fp32", "fp16", "int32", "q4_0", "q8_0"})


def test_nbytes_dense():
    assert nbytes((8, 128), "fp32") == 4096
    assert nbytes((8, 128), "fp16") == 2048
    assert nbytes((8, 128), "int32") == 4096


def test_nbytes_q4_0_is_18_bytes_per_32_elements():
    # 32 four-bit values + one fp16 scale = 18 bytes. Spec 5.1.1.
    assert nbytes((1, 32), "q4_0") == 18
    assert nbytes((8, 128), "q4_0") == 8 * 4 * 18


def test_nbytes_q4_0_rejects_unblocked_length():
    with pytest.raises(ValueError) as e:
        nbytes((8, 100), "q4_0")
    assert "32" in str(e.value)


def test_graph_reports_dangling_input():
    g = Graph(
        tensors={"x": _t("x", (4,))},
        ops=(Op(id=0, kind="add", inputs=("x", "missing"), outputs=("y",), attrs={}),),
        inputs=("x",),
        outputs=("y",),
    )
    problems = g.problems()
    assert any("missing" in p for p in problems)


def test_graph_reports_undeclared_output_tensor():
    g = Graph(
        tensors={"x": _t("x", (4,))},
        ops=(Op(id=0, kind="scale", inputs=("x",), outputs=("y",), attrs={"factor": 2.0}),),
        inputs=("x",),
        outputs=("y",),
    )
    assert any("y" in p for p in g.problems())


def test_graph_reports_double_write():
    g = Graph(
        tensors={"x": _t("x", (4,)), "y": _t("y", (4,))},
        ops=(
            Op(id=0, kind="scale", inputs=("x",), outputs=("y",), attrs={"factor": 2.0}),
            Op(id=1, kind="scale", inputs=("x",), outputs=("y",), attrs={"factor": 3.0}),
        ),
        inputs=("x",),
        outputs=("y",),
    )
    assert any("written twice" in p for p in g.problems())


def test_graph_reports_duplicate_op_id():
    g = Graph(
        tensors={"x": _t("x", (4,)), "y": _t("y", (4,)), "z": _t("z", (4,))},
        ops=(
            Op(id=0, kind="scale", inputs=("x",), outputs=("y",), attrs={"factor": 2.0}),
            Op(id=0, kind="scale", inputs=("y",), outputs=("z",), attrs={"factor": 3.0}),
        ),
        inputs=("x",),
        outputs=("z",),
    )
    assert any("id 0" in p for p in g.problems())


def test_graph_reports_use_before_write():
    g = Graph(
        tensors={"x": _t("x", (4,)), "y": _t("y", (4,)), "z": _t("z", (4,))},
        ops=(
            Op(id=0, kind="scale", inputs=("z",), outputs=("y",), attrs={"factor": 2.0}),
            Op(id=1, kind="scale", inputs=("x",), outputs=("z",), attrs={"factor": 3.0}),
        ),
        inputs=("x",),
        outputs=("y",),
    )
    assert any("before it is written" in p for p in g.problems())


def test_valid_graph_has_no_problems():
    g = Graph(
        tensors={"x": _t("x", (4,)), "y": _t("y", (4,))},
        ops=(Op(id=0, kind="scale", inputs=("x",), outputs=("y",), attrs={"factor": 2.0}),),
        inputs=("x",),
        outputs=("y",),
    )
    assert g.problems() == []


def test_empty_graph_is_not_a_pass():
    # A validator that found nothing to check must not report success.
    # This is the "absence read as success" hazard in CONTRIBUTING.md.
    g = Graph(tensors={}, ops=(), inputs=(), outputs=())
    assert g.problems() != []


def test_op_attrs_reject_unserializable_value():
    with pytest.raises(TypeError) as e:
        Op(id=0, kind="scale", inputs=("x",), outputs=("y",), attrs={"f": object()})
    assert "f" in str(e.value)

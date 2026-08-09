from __future__ import annotations

import json

import pytest

from hexlib.graph.ir import Op
from hexlib.graph.plan import (
    Plan,
    Slot,
    Step,
    Tiling,
    Transfer,
    from_json,
    render,
    to_json,
)
from hexlib.result import Err


def _plan(**overrides):
    base = dict(
        steps=(
            Step(
                op=Op(id=0, kind="matmul", inputs=("x", "w"), outputs=("y",), attrs={}),
                dma_in=(Transfer(id=0, tensor="w", direction="in", vtcm_offset=0, nbytes=576),),
                dma_wait=(0,),
                dma_out=(),
                tiling=Tiling(axis="n", tile_elements=32, count=96, buffers=2),
            ),
        ),
        vtcm=(Slot(tensor="y", offset=1152, size=4096, first_use=0, last_use=1),),
        vtcm_high_water=5248,
        predicted_bytes_moved=55296,
        vtcm_budget=8388608,
        unimplemented=("matmul",),
        target="hexagon-v75",
    )
    base.update(overrides)
    return Plan(**base)


def test_a_plan_with_no_steps_cannot_be_constructed():
    with pytest.raises(ValueError) as e:
        _plan(steps=())
    assert "step" in str(e.value).lower()


def test_high_water_must_be_an_int():
    with pytest.raises(TypeError) as e:
        _plan(vtcm_high_water=None)
    assert "vtcm_high_water" in str(e.value)


def test_predicted_bytes_moved_must_be_an_int():
    # Spec 8: a plan whose predicted cost cannot be computed is an error, not
    # a plan with a zero cost field.
    with pytest.raises(TypeError) as e:
        _plan(predicted_bytes_moved=None)
    assert "predicted_bytes_moved" in str(e.value)


def test_high_water_over_budget_cannot_be_constructed():
    with pytest.raises(ValueError) as e:
        _plan(vtcm_high_water=9_000_000, vtcm_budget=8_388_608)
    assert "9000000" in str(e.value).replace(",", "") or "9_000_000" in str(e.value)


def test_transfer_direction_must_be_in_or_out():
    with pytest.raises(ValueError):
        Transfer(id=0, tensor="w", direction="sideways", vtcm_offset=0, nbytes=4)


def test_transfer_nbytes_must_be_positive():
    with pytest.raises(ValueError):
        Transfer(id=0, tensor="w", direction="in", vtcm_offset=0, nbytes=0)


def test_tiling_requires_at_least_one_iteration_and_one_buffer():
    with pytest.raises(ValueError):
        Tiling(axis="n", tile_elements=32, count=0, buffers=2)
    with pytest.raises(ValueError):
        Tiling(axis="n", tile_elements=32, count=4, buffers=0)


def test_slot_last_use_may_not_precede_first_use():
    with pytest.raises(ValueError):
        Slot(tensor="y", offset=0, size=16, first_use=5, last_use=2)


def test_json_round_trip_is_lossless():
    plan = _plan()
    back = from_json(to_json(plan))
    assert not isinstance(back, Err)
    assert back == plan


def test_json_is_actually_json_and_has_no_pointers():
    text = to_json(_plan())
    parsed = json.loads(text)
    assert parsed["vtcm_budget"] == 8388608
    assert "0x" not in text, "an address leaked into the plan; offsets are numbers"


def test_from_json_on_garbage_is_an_err_not_an_exception():
    out = from_json("{not json")
    assert isinstance(out, Err)


def test_from_json_missing_a_required_field_is_an_err_naming_it():
    out = from_json(json.dumps({"steps": [], "vtcm": []}))
    assert isinstance(out, Err)
    assert "vtcm_high_water" in out.detail


def test_target_survives_the_json_round_trip_and_is_rendered():
    # Two plans for one graph are only comparable if each says what it targets.
    plan = _plan(target="adreno")
    back = from_json(to_json(plan))
    assert not isinstance(back, Err)
    assert back.target == "adreno"
    assert "adreno" in render(plan)


def test_a_plan_with_an_empty_target_cannot_be_constructed():
    with pytest.raises(ValueError):
        _plan(target="")


def test_from_json_missing_the_target_is_an_err():
    import json as _json

    raw = _json.loads(to_json(_plan()))
    del raw["target"]
    out = from_json(_json.dumps(raw))
    assert isinstance(out, Err)
    assert "target" in out.detail


def test_render_shows_the_budget_the_high_water_and_the_unimplemented_kinds():
    text = render(_plan())
    assert "8388608" in text.replace(",", "")
    assert "5248" in text.replace(",", "")
    # kernel: None is reported explicitly as "not implemented", never treated
    # as a no-op (spec 8).
    assert "matmul" in text
    assert "not implemented" in text.lower()


def test_render_says_so_when_nothing_is_unimplemented():
    text = render(_plan(unimplemented=()))
    assert "all op kinds have kernels" in text.lower()


def test_render_shows_the_tiling_loop_rather_than_unrolling_it():
    text = render(_plan())
    assert "96" in text  # the iteration count
    assert text.count("Transfer") < 10, "the render unrolled the loop"

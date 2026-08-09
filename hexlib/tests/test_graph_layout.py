from __future__ import annotations

import pytest

import hexlib.graph.opdefs  # noqa: F401 - registers all op kinds as a side effect
from hexlib.graph.layout import (
    ACCEPTED_LAYOUTS,
    HMX_TILE,
    Buffer,
    Layout,
    Placement,
    check_layouts,
    layout_nbytes,
)
from hexlib.graph.ops import all_kinds


def test_hmx_fp16_tile_is_32x32_and_2048_bytes():
    # matmul-ops.h:16-19. Symmetric across both operands in fp16 mode.
    assert HMX_TILE.rows == 32
    assert HMX_TILE.cols == 32
    assert HMX_TILE.elements == 1024
    assert HMX_TILE.nbytes_f16 == 2048


def test_dense_layout_nbytes_matches_the_plain_calculation():
    assert layout_nbytes((8, 128), "fp16", Layout.DENSE) == 2048


def test_q4_0_blocked_is_18_bytes_per_32():
    assert layout_nbytes((1, 32), "q4_0", Layout.Q4_0_BLOCKED) == 18


def test_q4_0_repacked_holds_the_same_bytes_as_blocked():
    # Repacking reorders; it does not resize. If these ever disagree, one of
    # them is wrong and the DMA byte count silently follows it.
    shape = (768, 3072)
    assert layout_nbytes(shape, "q4_0", Layout.Q4_0_REPACKED) == layout_nbytes(
        shape, "q4_0", Layout.Q4_0_BLOCKED
    )


def test_hmx_tile_f16_layout_rounds_up_to_whole_tiles():
    # 40x40 fp16 needs 2x2 tiles = 4 * 2048 bytes, not 40*40*2.
    assert layout_nbytes((40, 40), "fp16", Layout.HMX_TILE_F16) == 4 * 2048


def test_hmx_tile_f16_rejects_a_non_fp16_dtype():
    with pytest.raises(ValueError) as e:
        layout_nbytes((32, 32), "q4_0", Layout.HMX_TILE_F16)
    assert "fp16" in str(e.value)


def test_placement_is_frozen_and_offset_is_a_number_not_a_pointer():
    p = Placement(layout=Layout.DENSE, perm=(0, 1), buffer=Buffer.VTCM, offset=4096)
    assert isinstance(p.offset, int)
    with pytest.raises(Exception):
        p.offset = 0


def test_placement_rejects_a_negative_offset():
    with pytest.raises(ValueError):
        Placement(layout=Layout.DENSE, perm=(0,), buffer=Buffer.VTCM, offset=-1)


def test_placement_rejects_an_invalid_permutation():
    with pytest.raises(ValueError) as e:
        Placement(layout=Layout.DENSE, perm=(1, 2), buffer=Buffer.VTCM, offset=0)
    assert "permutation" in str(e.value)


def test_matmul_weight_must_be_repacked_not_blocked():
    blocked = (
        Placement(Layout.DENSE, (0, 1), Buffer.VTCM, 0),
        Placement(Layout.Q4_0_BLOCKED, (0, 1), Buffer.DDR, 0),
    )
    problems = check_layouts("matmul", blocked)
    assert problems
    assert "Q4_0_REPACKED" in problems[0]
    assert "Q4_0_BLOCKED" in problems[0]


def test_matmul_accepts_a_repacked_weight():
    ok = (
        Placement(Layout.DENSE, (0, 1), Buffer.VTCM, 0),
        Placement(Layout.Q4_0_REPACKED, (0, 1), Buffer.VTCM, 0),
    )
    assert check_layouts("matmul", ok) == []


def test_check_layouts_on_an_unknown_kind_is_a_problem_not_a_pass():
    # An op kind with no declared layouts must not silently accept anything.
    assert check_layouts("not_a_real_kind", ()) != []


def test_check_layouts_rejects_the_wrong_operand_count():
    problems = check_layouts("matmul", (Placement(Layout.DENSE, (0, 1), Buffer.VTCM, 0),))
    assert problems
    assert "1" in problems[0] and "2" in problems[0]


def test_every_accepted_layouts_entry_is_a_tuple_of_tuples_of_layouts():
    assert ACCEPTED_LAYOUTS, "table is empty; this test would pass vacuously"
    for kind, per_input in ACCEPTED_LAYOUTS.items():
        assert isinstance(per_input, tuple), kind
        for choices in per_input:
            assert isinstance(choices, tuple) and choices, kind
            for layout in choices:
                assert isinstance(layout, Layout), kind


def test_int8_hmx_layouts_exist_and_are_asymmetric():
    # Retained because docs/hardware/hmx-int8.md records the geometry, and
    # because the asymmetry is the thing four probe rounds got backwards.
    # Unused: hexlib uses HMX fp16 mode (spec 4.3).
    assert Layout.HMX_ACT_TILE_I8 is not Layout.HMX_WGT_TILE_I8
    assert layout_nbytes((32, 32), "int32", Layout.HMX_ACT_TILE_I8) == 2048
    assert layout_nbytes((32, 32), "int32", Layout.HMX_WGT_TILE_I8) == 1024


def test_every_registered_op_kind_has_accepted_layouts_entry():
    # Ensures no registered op is silently accepted without layout checking.
    # Direction: registered_kinds ⊆ table_keys (not equality).
    # matmul_epilogue is in the table but not yet registered (Task 3);
    # an equality check would fail for the wrong reason.
    registered = all_kinds()
    assert registered, "registered op kinds list is empty; this test would pass vacuously"
    for kind in registered:
        assert kind in ACCEPTED_LAYOUTS, (
            f"op kind {kind!r} is registered but has no entry in ACCEPTED_LAYOUTS; "
            "check_layouts would call it an unknown kind and report a problem"
        )

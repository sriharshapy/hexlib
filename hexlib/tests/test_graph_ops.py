from __future__ import annotations

import numpy as np
import pytest

from hexlib.graph import ops as opsmod
from hexlib.graph.ir import Tensor


def _dummy(kind="dummy"):
    return opsmod.OpDef(
        kind=kind,
        infer=lambda inputs, attrs: (((4,), "fp32"),),
        working_set=lambda inputs, outputs, attrs: 16,
        reference=lambda inputs, attrs: (inputs[0] * 2,),
    )


def test_opdef_requires_all_three_callables():
    for missing in ("infer", "working_set", "reference"):
        kwargs = {
            "kind": "x",
            "infer": lambda inputs, attrs: (),
            "working_set": lambda inputs, outputs, attrs: 0,
            "reference": lambda inputs, attrs: (),
        }
        kwargs[missing] = None
        with pytest.raises(TypeError) as e:
            opsmod.OpDef(**kwargs)
        assert missing in str(e.value)


def test_opdef_kernel_defaults_to_none_and_that_is_legal():
    d = _dummy()
    assert d.kernel is None


def test_register_then_get_round_trips():
    reg = opsmod.Registry()
    d = _dummy("mine")
    reg.register(d)
    assert reg.get("mine") is d


def test_duplicate_registration_is_an_error():
    reg = opsmod.Registry()
    reg.register(_dummy("mine"))
    with pytest.raises(ValueError) as e:
        reg.register(_dummy("mine"))
    assert "mine" in str(e.value)


def test_get_unknown_kind_names_the_kind_and_the_known_ones():
    reg = opsmod.Registry()
    reg.register(_dummy("mine"))
    with pytest.raises(KeyError) as e:
        reg.get("nope")
    msg = str(e.value)
    assert "nope" in msg
    assert "mine" in msg


def test_all_kinds_is_sorted():
    reg = opsmod.Registry()
    reg.register(_dummy("zebra"))
    reg.register(_dummy("apple"))
    assert reg.all_kinds() == ("apple", "zebra")


def test_empty_registry_all_kinds_is_empty_tuple():
    assert opsmod.Registry().all_kinds() == ()


def test_infer_signature_is_shape_dtype_pairs():
    reg = opsmod.Registry()
    reg.register(_dummy("mine"))
    out = reg.get("mine").infer((Tensor("a", "fp32", (4,)),), {})
    assert out == (((4,), "fp32"),)


def test_reference_receives_arrays_and_returns_a_tuple():
    reg = opsmod.Registry()
    reg.register(_dummy("mine"))
    out = reg.get("mine").reference((np.array([1.0, 2.0], dtype=np.float32),), {})
    assert isinstance(out, tuple)
    np.testing.assert_allclose(out[0], [2.0, 4.0])


def test_module_registry_is_populated_and_not_empty():
    # Importing the definitions package must actually register something. A
    # registry that registered nothing, passing "every OpDef is complete"
    # vacuously, is the "absence read as success" hazard CONTRIBUTING.md names.
    import hexlib.graph.opdefs  # noqa: F401

    assert len(opsmod.REGISTRY.all_kinds()) >= 11


def test_every_registered_opdef_is_complete():
    import hexlib.graph.opdefs  # noqa: F401

    kinds = opsmod.REGISTRY.all_kinds()
    assert kinds, "registry is empty; this test would otherwise pass vacuously"
    for kind in kinds:
        d = opsmod.REGISTRY.get(kind)
        assert callable(d.infer), kind
        assert callable(d.working_set), kind
        assert callable(d.reference), kind
        assert d.kind == kind

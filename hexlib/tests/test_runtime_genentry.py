# hexlib/tests/test_runtime_genentry.py
"""Entry-point generation. Pure text in, pure text out — no SDK, no device."""
import pytest

from hexlib.exec import runner as rn
from hexlib.runtime import genentry as ge


def test_scale_entry_unpacks_the_declared_signature():
    src = ge.emit_entry("scale", rn.SPECS["scale"])
    assert "int scale_fp16_entry(const hexlib_args *a)" in src
    assert "scale_fp16(" in src
    assert "(const hexlib_hf *) a->buf[0]" in src
    assert "(hexlib_hf *) a->buf[1]" in src
    assert "HEXLIB_DSP_OK" in src


def test_entry_checks_buffer_count_before_dereferencing_any():
    src = ge.emit_entry("scale", rn.SPECS["scale"])
    idx = src.index("a->n_buf")
    assert idx < src.index("scale_fp16("), "the count check must come first"
    assert "HEXLIB_DSP_ERR_INVAL_PARAMS" in src


def test_numel_scalar_comes_from_ne_not_from_a_host_promise():
    """`n` is derived on the DSP from the tensor extent it was given, so a host
    that lied about the length cannot make the kernel run past the buffer."""
    src = ge.emit_entry("scale", rn.SPECS["scale"])
    assert "a->ne[0][0]" in src


def test_attr_scalar_comes_from_params_blob():
    src = ge.emit_entry("scale", rn.SPECS["scale"])
    assert "a->params" in src


def test_perm_requirement_is_documented_as_unenforceable_not_faked():
    """transpose covers three signatures. Handing a perm (0,2,1) op to the perm
    (1,0,2) kernel returns a correctly-shaped, silently WRONG layout that every
    downstream shape check accepts -- but `hexlib_args` has no field that
    carries a permutation (only per-buffer buf/ne/dtype/layout, plus n_buf,
    vtcm, params, n_threads), so no `if` written here could ever fail on a
    perm mismatch. Enforcement lives only on the host, in
    `RunnerSpec.check_requires`, before the op is ever put on the wire. This
    gap must be documented plainly, not papered over with a check that cannot
    fail -- a check that cannot fail is indistinguishable from no check at all
    except that it looks like protection."""
    src = ge.emit_entry("transpose", rn.SPECS["transpose"])
    assert "NOT VERIFIED ON THE DSP" in src
    assert "perm" in src, "the gap should name the key it cannot verify"
    checks = [line for line in src.splitlines() if line.strip().startswith("if (")]
    assert not any("perm" in line for line in checks), (
        "a reachable `if` mentioning perm would be a check that cannot fail, "
        "i.e. exactly the decorative check this generator must not emit"
    )


def test_cast_requires_fp16_dtype():
    """Unlike `perm`, `dtype` maps onto a real per-buffer field
    (`hexlib_args.dtype[]`), so this one must be a reachable check, not just a
    documented gap -- asserting only the status-code string would still pass
    if the real check were downgraded to a comment, which is the one thing
    this test exists to catch."""
    src = ge.emit_entry("cast", rn.SPECS["cast"])
    assert "HEXLIB_DSP_ERR_REQUIRES" in src
    checks = [line for line in src.splitlines() if line.strip().startswith("if (")]
    assert any("a->dtype[" in line for line in checks), (
        "the dtype requirement must be a reachable `if (a->dtype[...] ...)` "
        "check, not only documented"
    )


def test_every_input_buffers_dtype_is_checked_before_it_is_cast():
    """THE FINDING. `a->dtype[i]` is carried per BUFFER and was read for exactly
    one of them -- the output, and only when `requires` happened to mention it.
    Meanwhile `emit_entry`'s own docstring names the hazard: "casting an fp32
    buffer to hexlib_hf* would halve every stride silently". A batch declaring a
    `scale` input as fp32 with ne[0] = 4100 -- reachable from `run_raw`, from
    main.c's hand-built blob, or from a future planner -- had 8200 of its 16400
    bytes read at half stride and got HEXLIB_DSP_OK back.

    So every buffer's declared dtype is now checked against the dtype the entry
    is about to cast it to. Asserted as a reachable `if` per buffer INDEX, not
    as a substring: a check on buf[0] only would still leave `add`'s second
    operand and every output unguarded."""
    src = ge.emit_entry("cast", rn.SPECS["cast"])
    checks = [ln for ln in src.splitlines() if ln.strip().startswith("if (")]
    assert any("a->dtype[0] !=" in ln for ln in checks), (
        "the fp32 INPUT's dtype must be checked before it is cast to float*"
    )
    assert any("a->dtype[1] !=" in ln for ln in checks), (
        "the fp16 OUTPUT's dtype must be checked before it is cast to hexlib_hf*"
    )
    # add has two inputs: the second one is the operand a single-buffer check
    # would miss.
    add = ge.emit_entry("add", rn.SPECS["add"])
    add_checks = [ln for ln in add.splitlines() if ln.strip().startswith("if (")]
    for i in range(3):
        assert any(f"a->dtype[{i}] !=" in ln for ln in add_checks), (
            f"buffer {i} of add is cast without its dtype being checked"
        )


def test_the_dtype_check_uses_the_wire_id_for_that_buffers_own_dtype():
    """The VALUE, bound to the buffer. `cast` is fp32 in, fp16 out, so the two
    checks must compare against DIFFERENT ids -- a generator that used the
    output's dtype for every buffer would emit two identical checks and refuse
    every legitimate cast batch, and one that used the input's would let the
    halved-stride write through."""
    from hexlib.runtime.wire import DTYPE_ID

    src = ge.emit_entry("cast", rn.SPECS["cast"])
    assert f"a->dtype[0] != {DTYPE_ID['fp32']}u" in src
    assert f"a->dtype[1] != {DTYPE_ID['fp16']}u" in src


def test_the_buffer_count_and_null_checks_still_come_before_any_dtype_check():
    """`a->dtype[i]` is only meaningful for a buffer the batch actually
    supplied, so the count check has to stay first."""
    src = ge.emit_entry("cast", rn.SPECS["cast"])
    assert src.index("a->n_buf") < src.index("a->dtype[")
    assert src.index("!a->buf[0]") < src.index("a->dtype[")
    assert src.index("a->dtype[") < src.index("cast_f32_f16(")


def test_a_dtype_requirement_that_disagrees_with_the_declared_dtype_is_refused():
    """THE RELATED MINOR, MADE LOUD. `_requires_check` hardcoded `out_idx` for
    every key: correct for `cast` today, and silently wrong for any future
    dtype requirement about an INPUT, which would have inspected the OUTPUT's
    dtype instead. There is no way to say which buffer a `requires` entry is
    about, so the generator refuses the ambiguous case rather than guessing --
    a spec whose dtype requirement is not its own declared output dtype is
    either about an input (unexpressible) or a contradiction (it would refuse
    every batch this serializer can build)."""
    bad = rn.RunnerSpec(
        kind="castish", kernel_dir="kernels/castish", inputs=("fp32",),
        out_dtype="fp16", scalars=(rn.Scalar("numel:0", "int"),),
        requires=(("dtype", "fp32"),),
    )
    with pytest.raises(ge.GenError, match="dtype"):
        ge.emit_entry("castish", bad)


def test_every_shipped_dtype_requirement_restates_its_own_declared_out_dtype():
    """Stated as a test because it is the reason the DSP-side dtype check
    cannot fail through `hexlib/exec/dsp.py`: that serializer stamps the output
    tensor's dtype from `spec.out_dtype`, which is the same value the
    requirement holds, so the generated `if` compares the spec with itself.
    The requirement is really enforced on the host, against the CALLER'S attr
    (`RunnerSpec.check_requires`). The generated check is still worth having --
    it is reachable from a hand-built batch (main.c, `run_raw`) -- but its reach
    should not be overstated, and this pins the fact."""
    for name, spec in rn.SPECS.items():
        for key, want in spec.requires:
            if key == "dtype":
                assert want == spec.out_dtype, name


def test_table_is_sorted_and_terminated():
    src = ge.emit_table({"scale": rn.SPECS["scale"], "add": rn.SPECS["add"]})
    assert "hexlib_kernel_table[]" in src
    assert "hexlib_kernel_table_len" in src
    assert src.index('"add"') < src.index('"scale"'), "sorted, so diffs are stable"


def test_kind_ids_are_stable_across_runs():
    """A kind id crossing the wire must not depend on dict ordering — a renumber
    silently sends every op to the wrong kernel."""
    assert ge.KIND_ID == dict(sorted(ge.KIND_ID.items(), key=lambda kv: kv[1]))
    ids = list(ge.KIND_ID.values())
    assert ids == sorted(ids) and len(set(ids)) == len(ids)


def test_every_spec_has_a_kind_id():
    for name in rn.SPECS:
        assert name in ge.KIND_ID, f"{name} has no wire id"


# The ids as shipped. ON-WIRE IDS ARE AN ABI: `skel_dispatch.c` matches an op by
# id alone, the blob carries no table version, and nothing in a response would
# reveal a mismatch except the id itself. So a new kind is APPENDED and an
# existing one never moves. test_kind_ids_are_stable_across_runs below asserts
# only sortedness and uniqueness, which a wholesale renumber preserves -- and a
# renumber that moved `scale` from 9 onto `transpose`'s 11 would dispatch every
# scale op to transpose_th_fp16_entry with a matching buffer count, two non-null
# pointers, and HEXLIB_DSP_OK. This literal is what makes that fail.
SHIPPED_KIND_IDS = {
    "add": 1, "cast": 2, "layernorm": 3, "matmul": 4, "matmul_epilogue": 5,
    "patchify": 6, "reshape": 7, "rope_2d": 8, "scale": 9, "softmax": 10,
    "transpose": 11,
}


def test_the_shipped_kind_ids_never_move():
    for name, want in SHIPPED_KIND_IDS.items():
        assert ge.KIND_ID.get(name) == want, (
            f"{name} was id {want} on the wire and is now "
            f"{ge.KIND_ID.get(name)}. Append new kinds; never renumber."
        )
    new = set(ge.KIND_ID) - set(SHIPPED_KIND_IDS)
    assert all(ge.KIND_ID[n] > max(SHIPPED_KIND_IDS.values()) for n in new), (
        f"{sorted(new)} must take ids above {max(SHIPPED_KIND_IDS.values())}"
    )


def test_every_kind_a_COMPILED_PLAN_can_contain_has_a_wire_id():
    """THE COVERAGE CLAIM, CHECKED AGAINST THE REGISTRY RATHER THAN ASSUMED.
    `KIND_ID` holds 11 entries and the op registry holds 13. The two absentees
    are `gelu_tanh` and `gelu_erf`, and both are in `fuse.FUSABLE_ACTS`: fusion
    absorbs them into `matmul_epilogue`'s `act` attr, so neither can appear as a
    standalone plan step and there is no live wire gap today.

    That is a claim about a PASS, though, not about the table, and it is exactly
    the claim that stops being true the moment a new op kind is registered
    without an id -- at which point `dsp.py` raises KeyError on a graph that
    compiles fine. So the registry is compared here rather than trusted, and a
    new kind that is neither fusable nor given an id fails this."""
    import hexlib.graph.opdefs  # noqa: F401  -- registers the op defs
    from hexlib.graph.fuse import FUSABLE_ACTS
    from hexlib.graph.ops import REGISTRY

    dispatchable = set(REGISTRY.all_kinds()) - set(FUSABLE_ACTS)
    missing = sorted(dispatchable - set(ge.KIND_ID))
    assert not missing, (
        f"{missing} can appear as a plan step and has no wire id; dsp.py would "
        f"raise KeyError on a graph that compiled cleanly"
    )
    assert set(ge.KIND_ID) <= set(REGISTRY.all_kinds()), (
        f"{sorted(set(ge.KIND_ID) - set(REGISTRY.all_kinds()))} has a wire id "
        f"but is not an op kind at all"
    )


def test_generated_entry_includes_the_kernel_api_header():
    src = ge.emit_entry("scale", rn.SPECS["scale"])
    assert '#include "kernel_api.h"' in src
    assert '#include "hexlib_dsp.h"' in src


def test_input_dtype_not_output_dtype_decides_the_input_cast():
    """cast is fp32 in, fp16 out. Casting the input to hexlib_hf* would halve
    every stride and read the wrong half of the buffer — a plausible wrong
    answer, not a crash."""
    src = ge.emit_entry("cast", rn.SPECS["cast"])
    assert "(const float *) a->buf[0]" in src
    assert "(hexlib_hf *) a->buf[1]" in src


def test_the_function_name_is_the_basename_of_the_kernel_dir():
    """spec.kernel_dir is a PATH, 'kernels/scale_fp16'. Using it verbatim would
    emit `kernels/scale_fp16_entry`, which is not an identifier."""
    assert rn.SPECS["scale"].kernel_dir == "kernels/scale_fp16"
    src = ge.emit_entry("scale", rn.SPECS["scale"])
    assert "int scale_fp16_entry(" in src
    assert "kernels/" not in src.split("Source of truth")[1]


def test_a_hand_written_dsp_entry_wins(tmp_path):
    """The escape hatch is real and its use is visible: a kernel whose argument
    mapping is not expressible declaratively ships its own dsp_entry.c, and the
    generator must not overwrite or shadow it.

    `expect=("scale",)` states what this one-kernel tmp tree actually claims to
    hold -- without it, `generate` refuses the tree as a partial checkout, which
    is the point of the test below."""
    kdir = tmp_path / "kernels" / "scale_fp16"
    kdir.mkdir(parents=True)
    (kdir / "dsp_entry.c").write_text("/* hand written */\n")
    written = ge.generate(str(tmp_path), str(tmp_path / "out"), expect=("scale",))
    assert not any("scale_fp16_entry.c" in w for w in written)
    assert any("hexlib_kernel_table.c" in w for w in written)


def test_generate_takes_the_REPO_root_not_a_kernels_root(tmp_path):
    """kernel_dir is repo-relative. Passing `<repo>/kernels` would look for
    `kernels/kernels/scale_fp16`, find nothing, and emit an EMPTY dispatch table
    -- a build that links and then reports 'no kernel for kind 9' at run time."""
    (tmp_path / "kernels" / "scale_fp16").mkdir(parents=True)
    ok = ge.generate(str(tmp_path), str(tmp_path / "out"), expect=("scale",))
    assert any("scale_fp16_entry.c" in w for w in ok)
    with pytest.raises(ge.GenError, match="no kernel"):
        ge.generate(str(tmp_path / "kernels"), str(tmp_path / "out2"),
                    expect=("scale",))


def test_a_PARTIAL_kernel_tree_is_an_error_not_a_partial_dispatch_table(tmp_path):
    """THE MINOR. `generate` raised only when NO kernel directory was found. A
    tree missing SOME of them emitted a dispatch table missing those rows, with
    no error at all -- so the ops answered HEXLIB_DSP_ERR_NO_KERNEL at run time,
    which reads as "this kernel is broken" rather than "the generator was
    pointed at an incomplete tree". Absence reported as partial success is the
    same shape as absence reported as success, one step down.

    By default every kind with a `RunnerSpec` must have its directory; `expect`
    narrows that for a caller that genuinely holds a subset (only the tests in
    this file, today)."""
    (tmp_path / "kernels" / "scale_fp16").mkdir(parents=True)
    with pytest.raises(ge.GenError) as exc:
        ge.generate(str(tmp_path), str(tmp_path / "out"))
    # The MISSING list, not merely the names somewhere in the message: the one
    # kernel that IS present must not be reported as absent.
    assert "for ['add', 'cast', 'transpose']" in str(exc.value)
    assert not (tmp_path / "out" / "hexlib_kernel_table.c").exists(), (
        "a partial dispatch table must not be left behind for a build to link"
    )


def test_an_expect_naming_a_kind_with_no_spec_is_an_error(tmp_path):
    """`expect` narrows a claim; it cannot invent one. A typo'd or stale name
    would otherwise quietly narrow nothing."""
    (tmp_path / "kernels" / "scale_fp16").mkdir(parents=True)
    with pytest.raises(ge.GenError, match="scal"):
        ge.generate(str(tmp_path), str(tmp_path / "out"), expect=("scal",))


def test_unknown_scalar_source_is_an_error_not_a_zero():
    bad = rn.RunnerSpec(
        kind="bogus", kernel_dir="bogus_fp16", inputs=("fp16",), out_dtype="fp16",
        scalars=(rn.Scalar(source="wat:1"),),
    )
    with pytest.raises(ge.GenError, match="wat:1"):
        ge.emit_entry("bogus", bad)

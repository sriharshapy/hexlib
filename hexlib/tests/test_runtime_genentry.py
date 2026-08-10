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
    generator must not overwrite or shadow it."""
    kdir = tmp_path / "kernels" / "scale_fp16"
    kdir.mkdir(parents=True)
    (kdir / "dsp_entry.c").write_text("/* hand written */\n")
    written = ge.generate(str(tmp_path), str(tmp_path / "out"))
    assert not any("scale_fp16_entry.c" in w for w in written)
    assert any("hexlib_kernel_table.c" in w for w in written)


def test_generate_takes_the_REPO_root_not_a_kernels_root(tmp_path):
    """kernel_dir is repo-relative. Passing `<repo>/kernels` would look for
    `kernels/kernels/scale_fp16`, find nothing, and emit an EMPTY dispatch table
    -- a build that links and then reports 'no kernel for kind 9' at run time."""
    (tmp_path / "kernels" / "scale_fp16").mkdir(parents=True)
    ok = ge.generate(str(tmp_path), str(tmp_path / "out"))
    assert any("scale_fp16_entry.c" in w for w in ok)
    with pytest.raises(ge.GenError, match="no kernel"):
        ge.generate(str(tmp_path / "kernels"), str(tmp_path / "out2"))


def test_unknown_scalar_source_is_an_error_not_a_zero():
    bad = rn.RunnerSpec(
        kind="bogus", kernel_dir="bogus_fp16", inputs=("fp16",), out_dtype="fp16",
        scalars=(rn.Scalar(source="wat:1"),),
    )
    with pytest.raises(ge.GenError, match="wat:1"):
        ge.emit_entry("bogus", bad)

# hexlib/tests/test_anticheat.py
from hexlib import anticheat as ac


def test_v_register_operand_counts_as_hvx():
    disasm = "   4c:\t01 c0 00 28\t\tv1 = vmem(r0++#1)\n"
    assert ac.disasm_has_hvx(disasm)


def test_scalar_gpr_pair_with_hvx_looking_mnemonic_does_not_count():
    """vaddw/vaddub on rN registers are SCALAR GPR-pair ops that merely
    disassemble with HVX-looking mnemonics. Keying on the mnemonic would grant
    accel credit to a kernel that never touched the vector unit — so the
    discriminator is the V-register/vmem OPERAND, never the mnemonic."""
    disasm = "   40:\t00 c0 00 f3\t\tr1:0 = vaddw(r3:2, r5:4)\n"
    assert not ac.disasm_has_hvx(disasm)


def test_symbol_header_lines_are_ignored():
    """Lines without a tab have no instruction column; a token in a symbol name
    must never grant credit."""
    assert not ac.disasm_has_hvx("0000004c <vmem_lookalike_symbol>:\n")


def test_load_only_hvx_is_not_compute():
    """A kernel that moves data through the vector unit but computes in scalar
    registers is not genuinely accelerated."""
    disasm = (
        "   4c:\t01 c0 00 28\t\tv1 = vmem(r0++#1)\n"
        "   50:\t02 c0 00 28\t\tvmem(r1++#1) = v2\n"
    )
    assert ac.disasm_has_hvx(disasm)
    assert not ac.disasm_has_hvx_compute(disasm)


def test_vector_arithmetic_is_compute():
    disasm = "   54:\t03 c0 00 1f\t\tv3.h = vadd(v1.h, v2.h)\n"
    assert ac.disasm_has_hvx_compute(disasm)


def test_hmx_requires_both_activation_and_weight():
    """A lone mxmem is matrix-memory movement, not a multiply. Requiring the
    operand PAIR closes the hole where a stray mxmem flipped used_hmx."""
    act_only = "   60:\t00 00 00 00\t\tactivation.ub = mxmem(r4, r7)\n"
    assert not ac.disasm_has_hmx(act_only)

    both = act_only + "   64:\t00 00 00 00\t\tweight.b = mxmem(r3, r8)\n"
    assert ac.disasm_has_hmx(both)


def test_empty_disassembly_grants_nothing():
    """Fail closed: uncertainty never grants credit."""
    assert not ac.disasm_has_hvx("")
    assert not ac.disasm_has_hvx_compute("")
    assert not ac.disasm_has_hmx("")


def test_disassemble_returns_empty_when_objdump_is_missing(tmp_path):
    """The documented contract: '' on any failure, which denies credit.
    Before finding 4's fix, a missing objdump binary raised FileNotFoundError
    straight out of subprocess.run instead of coming back as a failing rc."""
    assert ac.disassemble("nosuch.o", str(tmp_path / "no_such_bin")) == ""


def test_prove_accel_grants_nothing_when_objdump_is_missing(tmp_path):
    proof = ac.prove_accel("nosuch.o", str(tmp_path / "no_such_bin"))
    assert not proof.used_hvx and not proof.used_hvx_compute and not proof.used_hmx

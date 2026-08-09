"""ELF-level proof that a kernel genuinely used the vector or matrix unit.

Acceleration credit is NEVER granted from source text or a self-reported flag.
The compiled object is disassembled and the instruction operands are inspected.

THE DISCRIMINATOR IS THE OPERAND, NEVER THE MNEMONIC. Scalar GPR-pair
instructions disassemble with HVX-looking mnemonics — `r1:0 = vaddw(r3:2, r5:4)`
is a scalar op — so keying on `vaddw` would grant vector credit to a kernel that
never touched the vector unit. A real HVX instruction references a V register
(v0..v31, including pairs v1:0) or a vector memory op.

WHAT THIS CAN AND CANNOT ANSWER. Every function here answers exactly one
question: is the instruction PRESENT in the disassembly? None can answer "did it
execute?" — a static disassembly contains no such information. A kernel whose
HVX work sits in a never-taken branch will still be reported as using HVX.
Closing that gap needs per-packet commit profiling; treat these figures as upper
bounds.

Adapted from hexbench's `env/anticheat.py` (same author), whose detector logic
is frozen. Fail closed: uncertainty denies credit.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from hexlib import toolchain as tc

# A real HVX instruction references an HVX vector register (v0..v31, incl. pairs
# v1:0) or an HVX vector memory op.
_HVX_INSN = re.compile(r"\bv\d+(?::\d+)?\b|\bvmemu?\b|\bvgather\b|\bvscatter\b")

# HVX *memory* ops. A kernel that only moves data through the vector unit but
# does its arithmetic in scalar registers is not genuinely accelerated, so these
# are excluded when asking "did the vector unit do real compute?".
_HVX_MEM = re.compile(r"\bvmemu?\b|\bvgather\b|\bvscatter\b")

# A genuine HMX matmul loads BOTH matrix operands: an `activation.` operand AND a
# `weight.` operand — the defining act of a multiply. Real objdump forms, from
# compiling HMX helpers with -mv75 -mhvx -mhmx -O2: int8
# `activation.ub = mxmem(r4,r7)` / `weight.b = mxmem(r3,r8)`; fp16
# `activation.hf = mxmem(...)` / `weight.hf = mxmem(...)`.
_HMX_ACT = re.compile(r"\bactivation\.")
_HMX_WGT = re.compile(r"\bweight\.")


def _asm_lines(disasm_text: str):
    """Yield just the assembly column. Lines without a tab are symbol headers or
    blanks and have no instruction column, so a token inside a symbol name can
    never grant credit."""
    for line in disasm_text.splitlines():
        if "\t" not in line:
            continue
        yield line.rsplit("\t", 1)[-1]


def disasm_has_hvx(disasm_text: str) -> bool:
    """True iff any instruction references an HVX vector register or vector
    memory op. Pure (no I/O)."""
    return any(_HVX_INSN.search(asm) for asm in _asm_lines(disasm_text))


def disasm_has_hvx_compute(disasm_text: str) -> bool:
    """True iff there is a genuine HVX vector *compute* op — a V-register
    instruction that is not a pure vector memory op. Pure (no I/O)."""
    return any(
        _HVX_INSN.search(asm) and not _HVX_MEM.search(asm)
        for asm in _asm_lines(disasm_text)
    )


def disasm_has_hmx(disasm_text: str) -> bool:
    """True iff BOTH an `activation.` and a `weight.` matrix operand are loaded.
    Matrix-memory movement alone is not a multiply and does not count.

    KNOWN LIMITATION: the pair is matched over the whole text, not per packet,
    so an activation load in one function and a weight load in another would
    satisfy it. Pure (no I/O)."""
    saw_act = saw_wgt = False
    for asm in _asm_lines(disasm_text):
        if _HMX_ACT.search(asm):
            saw_act = True
        if _HMX_WGT.search(asm):
            saw_wgt = True
        if saw_act and saw_wgt:
            return True
    return False


@dataclass(frozen=True)
class AccelProof:
    used_hvx: bool
    used_hvx_compute: bool
    used_hmx: bool


def disassemble(obj: str, bin_dir: str) -> str:
    """Disassemble an object with hexagon-llvm-objdump. Returns '' on any
    failure, which denies credit rather than assuming it."""
    objdump = os.path.join(bin_dir, tc.exe("hexagon-llvm-objdump"))
    env = tc.toolchain_env(bin_dir)
    rc, out, err, timed_out = tc.run(
        [objdump, "-d", obj], env, timeout=tc.SIM_TIMEOUT_S
    )
    if timed_out or rc != 0:
        return ""
    return out


def prove_accel(obj: str, bin_dir: str) -> AccelProof:
    """Disassemble the kernel-only object and report what it genuinely used.
    Fail-closed: an unreadable disassembly grants nothing."""
    disasm = disassemble(obj, bin_dir)
    return AccelProof(
        used_hvx=disasm_has_hvx(disasm),
        used_hvx_compute=disasm_has_hvx_compute(disasm),
        used_hmx=disasm_has_hmx(disasm),
    )

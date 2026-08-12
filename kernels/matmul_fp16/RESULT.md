### hexlib verify — matmul_fp16

| gate | result |
|---|---|
| correct | PASS |
| max abs error | 0.000976562 (n_wrong 0) |
| kernel_cycles | 995714 |
| accel (ELF-proven) | hvx, hvx-compute |
| near-miss `nearmiss_fp16_accumulate.c` | correctly rejected |
| near-miss `nearmiss_transposed_operand.c` | correctly rejected |
| near-miss `nearmiss_wrong_batch_stride.c` | correctly rejected |
| **gate** | **PASS** |

target `v75` · toolchain `19.0.04` · SDK `6.4.0.2` · host `sriha@Heathcliff` · `2026-08-12T09:35:31Z`

Measured on the hexagon simulator under the pinned bus model (buspenalty 75, busratio 2). The simulator is cycle-approximate; these numbers are reproducible, not silicon measurements.

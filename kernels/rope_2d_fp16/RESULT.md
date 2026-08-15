### hexlib verify — rope_2d_fp16

| gate | result |
|---|---|
| correct | PASS |
| max abs error | 6.10352e-05 (n_wrong 0) |
| kernel_cycles | 1212 |
| accel (ELF-proven) | hvx, hvx-compute |
| near-miss `nearmiss_adjacent_pairing.c` | correctly rejected |
| near-miss `nearmiss_fp16_accumulate.c` | correctly rejected |
| near-miss `nearmiss_negated_term.c` | correctly rejected |
| near-miss `nearmiss_partial_rotation.c` | correctly rejected |
| near-miss `nearmiss_swapped_cos_sin.c` | correctly rejected |
| near-miss `nearmiss_table_indexed_by_head.c` | correctly rejected |
| **gate** | **PASS** |

target `v75` · toolchain `19.0.04` · SDK `6.4.0.2` · host `sriha@Heathcliff` · `2026-08-11T21:14:27Z`

Measured on the hexagon simulator under the pinned bus model (buspenalty 75, busratio 2). The simulator is cycle-approximate; these numbers are reproducible, not silicon measurements.

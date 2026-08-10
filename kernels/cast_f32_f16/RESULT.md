### hexlib verify — cast_f32_f16

| gate | result |
|---|---|
| correct | PASS |
| max abs error | 0.5 (n_wrong 0) |
| kernel_cycles | 1176 |
| accel (ELF-proven) | hvx, hvx-compute |
| near-miss `nearmiss_forgets_the_deal.c` | correctly rejected |
| near-miss `nearmiss_reads_one_vector_per_output.c` | correctly rejected |
| near-miss `nearmiss_swapped_halves.c` | correctly rejected |
| **gate** | **PASS** |

target `v75` · toolchain `19.0.04` · SDK `6.4.0.2` · host `sriha@Heathcliff` · `2026-08-10T11:07:06Z`

Measured on the hexagon simulator under the pinned bus model (buspenalty 75, busratio 2). The simulator is cycle-approximate; these numbers are reproducible, not silicon measurements.

### hexlib verify — scale_fp16

| gate | result |
|---|---|
| correct | PASS |
| max abs error | 0 (n_wrong 0) |
| kernel_cycles | 886 |
| accel (ELF-proven) | hvx, hvx-compute |
| near-miss `nearmiss_drops_the_tail.c` | correctly rejected |
| near-miss `nearmiss_splats_the_numeric_value.c` | correctly rejected |
| **gate** | **PASS** |

target `v75` · toolchain `19.0.04` · SDK `6.4.0.2` · host `sriha@Heathcliff` · `2026-08-10T09:46:39Z`

Measured on the hexagon simulator under the pinned bus model (buspenalty 75, busratio 2). The simulator is cycle-approximate; these numbers are reproducible, not silicon measurements.

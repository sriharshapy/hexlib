### hexlib verify — softmax_fp16

| gate | result |
|---|---|
| correct | PASS |
| max abs error | 0 (n_wrong 0) |
| kernel_cycles | 11292 |
| accel (ELF-proven) | hvx, hvx-compute |
| near-miss `nearmiss_no_max_subtraction.c` | correctly rejected |
| near-miss `nearmiss_sum_fp16.c` | correctly rejected |
| near-miss `nearmiss_wrong_axis.c` | correctly rejected |
| **gate** | **PASS** |

target `v75` · toolchain `19.0.04` · SDK `6.4.0.2` · host `sriha@Heathcliff` · `2026-08-11T21:02:03Z`

Measured on the hexagon simulator under the pinned bus model (buspenalty 75, busratio 2). The simulator is cycle-approximate; these numbers are reproducible, not silicon measurements.

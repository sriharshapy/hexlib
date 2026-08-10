### hexlib verify — layernorm_fp16

| gate | result |
|---|---|
| correct | PASS |
| max abs error | 0.000488281 (n_wrong 0) |
| kernel_cycles | 111088 |
| accel (ELF-proven) | hvx, hvx-compute |
| near-miss `nearmiss_permuted_affine_lanes.c` | correctly rejected |
| near-miss `nearmiss_unbiased_variance.c` | correctly rejected |
| **gate** | **PASS** |

target `v75` · toolchain `19.0.04` · SDK `6.4.0.2` · host `sriha@Heathcliff` · `2026-08-10T11:19:02Z`

Measured on the hexagon simulator under the pinned bus model (buspenalty 75, busratio 2). The simulator is cycle-approximate; these numbers are reproducible, not silicon measurements.

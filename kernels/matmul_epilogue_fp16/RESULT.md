### hexlib verify — matmul_epilogue_fp16

| gate | result |
|---|---|
| correct | PASS |
| max abs error | 0.000244141 (n_wrong 0) |
| kernel_cycles | 14310406 |
| accel (ELF-proven) | hvx, hvx-compute |
| near-miss `nearmiss_bias_after_activation.c` | correctly rejected |
| near-miss `nearmiss_fp16_accumulation.c` | correctly rejected |
| near-miss `nearmiss_gelu_swap.c` | correctly rejected |
| near-miss `nearmiss_scale_off_by_one_block.c` | correctly rejected |
| near-miss `nearmiss_swapped_nibble_order.c` | correctly rejected |
| **gate** | **PASS** |

target `v75` · toolchain `19.0.04` · SDK `6.4.0.2` · host `sriha@Heathcliff` · `2026-08-12T05:28:34Z`

Measured on the hexagon simulator under the pinned bus model (buspenalty 75, busratio 2). The simulator is cycle-approximate; these numbers are reproducible, not silicon measurements.

### hexlib verify — patchify_fp32

| gate | result |
|---|---|
| correct | PASS |
| max abs error | 0 (n_wrong 0) |
| kernel_cycles | 2018331 |
| accel (ELF-proven) | hvx · movement-only (no arithmetic) |
| near-miss `nearmiss_channel_temporal_swap.c` | correctly rejected |
| near-miss `nearmiss_merge_ignored.c` | correctly rejected |
| near-miss `nearmiss_patch_interior_swap.c` | correctly rejected |
| **gate** | **PASS** |

target `v75` · toolchain `19.0.04` · SDK `6.4.0.2` · host `sriha@Heathcliff` · `2026-08-11T21:47:01Z`

Measured on the hexagon simulator under the pinned bus model (buspenalty 75, busratio 2). The simulator is cycle-approximate; these numbers are reproducible, not silicon measurements.

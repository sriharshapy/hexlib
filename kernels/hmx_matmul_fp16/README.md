# hmx_matmul_fp16 — committed, and it does NOT pass its gate

**There is no `RESULT.md` in this directory, deliberately.** Every other kernel here has
one, because `RESULT.md` is the tool-generated record of a kernel that cleared gates 3
and 5. This one does not clear them: the simulator faults before the harness prints
`HEXLIB_VERDICT`, and the gate reports that honestly —

> no verdict recovered from the simulator: the harness never printed HEXLIB_VERDICT, so
> nothing was actually checked. This is a failure, not a pass.

Writing a `RESULT.md` by hand would make an ungated kernel indistinguishable from a
gated one in exactly the place a reader goes to check. The record lives here instead.

The kernel source is committed anyway because **what it found on the way is worth more
than the code**, and because the code is probably fine — it was being run somewhere it
can never work.

## What it is

The fp16 HMX tile sequence at the smallest shape whose scalar reference is trivially
checkable: `M=32, N=32, K=64`, two 2048-byte dot tiles, `range = 2048 * n_dot_tiles - 1`.
Deliberately *before* weight repacking, deep-K accumulation or `matmul_epilogue`
integration are built on top of it. Operands are symmetric in fp16 mode — both are
2048-byte tiles — unlike the int8 path, where the activation is masked at 2047 and the
weight at 1023.

## Two real defects, found and fixed

Both would have cost far more later, and both were found by reading upstream rather than
by guessing at the fault.

1. **`HMX_SET_BIAS` does not take a bias tile.** It takes a **256-byte** area, not a
   32×32 one. `hmx_init_column_scales` (llama.cpp `hmx-utils.h:19-23`) writes one HVX
   vector of per-**column** packed 32-bit words, then one zero vector — and each word is
   an fp16 *pair*: low half a multiplicative scale, high half an additive bias. That is
   what upstream's `Q6_V_vsplat_R(0x3c00)` means when its comment reads "scale: 1.0,
   bias: 0.0 in FP16". The first version passed a 2048-byte 32×32 array, which is a
   different operand entirely.

   `docs/hardware/hmx-int8.md` had recorded "the bias tile is a scale, not an additive
   bias" from the earlier int8 probing. That was half the story: it is **both**, one pair
   per column.

2. **Tile-sized alignment.** `mxmem` addresses a 2048-byte tile; this project's standard
   `HEXLIB_ALIGN` is 128. The first fault was at `badva0=04114a68`, whose low 11 bits are
   not zero.

## The wall, and it is structural

With both fixed it still faults, and the fault **moved to a stack address**
(`badva0=04115868`) while the tiles are now 2048-aligned — so it is no longer the
operands. Exception code `0x18` in `ssr=80740018`, `ccr=00130000`.

The reason is not a hypothesis any more. Reading
`../llama.cpp/ggml/src/ggml-hexagon/htp` gives the whole protocol, and **a standalone
simulator ELF — which is how every other kernel here is gated — has none of its four
parts**:

| | what HMX needs | upstream | in hexlib |
|---|---|---|---|
| 1 | **Power**, separate from HVX's own request and guarded on `__HVX_ARCH__ >= 75`: `HAP_power_set_HMX_v2` with all three DCVS corners at `VCORNER_MAX`, perf mode `HAP_CLK_PERF_HIGH` | `main.c:473-492` | **does not exist** |
| 2 | **Acquisition** in the *same* compute-res attr as VTCM: `HAP_compute_res_attr_set_hmx_param(&attr, 1)` before `HAP_compute_res_acquire` | `main.c:259-291` | `skel_vtcm.c:102` — written, commented *"REVISIT THIS when an HMX kernel first lands"*, and **never once executed** |
| 3 | **An explicit lock** around every use: `HAP_compute_res_hmx_lock` / `_hmx_unlock` | `hmx-queue.c:17-30` | **does not exist** |
| 4 | **A dedicated thread** owning that lock — upstream queues all HMX work to `hmx_queue_thread`, created only `if (n_hmx)` | `main.c:386-394` | n/a |

The pieces map almost one-to-one onto code that already exists here, which is why this
is a short list rather than a redesign: llama.cpp's
`htp_iface_start(..., n_hvx, n_hmx, max_vmem)` is the *same* signature hexlib ported, and
`n_hmx` was always meant for exactly this — `session.c` passes `0` unconditionally.

**Consequence: an HMX kernel belongs on the QuRT-hosted batch path, not on `hexlib
test`'s standalone-ELF gate path.** That is a structural difference from all twelve
gated kernels in this repository. They are gated by a program that boots, computes and
prints a verdict with no protection domain around it, and HMX cannot run in one.

## Open, and deliberately not guessed

**Is `HAP_compute_res_hmx_lock` thread-scoped?** That decides whether hexlib's
single-threaded skel can hold it inline or has to stand up a queue thread the way
upstream does. Writing the four steps into hexlib before knowing this would be building
on the same kind of assumption that produced both defects above.

**The cheapest next move** is to wire `n_hmx=1`, add the power request and an inline lock
in `skel_dispatch.c`, and run this existing tile through the **batch** path on the
simulator. If it computes, HMX is unblocked.

## Why `nearmiss_split_packet.c` is kept

Nothing can run it yet. It is kept because it preserves the failure
`docs/hardware/hmx-int8.md` spent four probe rounds on: an activation load and a weight
load issued as **two packets instead of one** does not degrade the accumulator, it
**clears** it. That reads as a hardware limitation rather than a coding error, and it is
the kind of thing that is expensive to rediscover.

## What HMX is and is not for

It is a **speed** lever, not an accuracy one. The encoder's accuracy problem is the
weight format — see the accuracy table in the top-level `README.md`, where q4_0 costs
0.13 of cosine and fp16 activations cost 9e-06. No amount of HMX moves that.

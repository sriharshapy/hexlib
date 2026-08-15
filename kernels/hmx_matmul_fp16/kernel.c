/* kernels/hmx_matmul_fp16/kernel.c
 *
 * One 32x32 fp16 output tile through the HMX tile engine. See kernel_api.h for
 * the sequence and for why the one-packet rule is the whole point.
 */
#include "kernel_api.h"

/* Transcribed from ../llama.cpp/ggml/src/ggml-hexagon/htp/hmx-utils.h:207-220
 * (MIT; see ATTRIBUTION.md), not paraphrased -- the brace placement is load
 * bearing and a reformatting is a behaviour change. */
#define HMX_LOAD_MPY_DEEP_F16(act, wt, range) \
    "{\n" \
    "    activation.hf = mxmem(" act ", " range "):deep\n" \
    "    weight.hf = mxmem(" wt ", " range ")\n" \
    "}\n"

#define HMX_STORE_AFTER_F16(out, scale_reg) \
    "mxmem(" out ", " scale_reg "):after.hf = acc\n"

#define HMX_SET_BIAS(scales) \
    "bias = mxmem2(" scales ")\n"

#define HMX_CLRACC_F16() \
    "mxclracc.hf\n"

void hmx_matmul_fp16(const hexlib_hf *act, const hexlib_hf *wt,
                     const unsigned int *scales, hexlib_hf *out) {
    /* `range` spans the WHOLE dot depth in one instruction: the `:deep` variant
     * walks n_dot_tiles consecutive tiles itself rather than needing a loop, so
     * long as n_dot_tiles <= 32 (llama.cpp's own bound, matmul-ops.h and the
     * `__builtin_assume(n_dot_tiles <= 32)` in core_dot_chunk_fp16_short). The
     * encoder's K of 768 and 3072 are 24 and 96 dot tiles, so 3072 will need the
     * outer accumulation loop core_dot_chunk_fp16 already shows -- that is the
     * next kernel, not this one. */
    const unsigned int range = (unsigned int) (HMX_TILE_BYTES * HMX_MM_DOT_TILES - 1);

    /* BIAS IS SET ONCE, BEFORE the accumulator is cleared and before the
     * multiply -- llama.cpp sets it outside both of its tile loops. Setting it
     * after mxclracc, or per tile, is a different operation. */
    asm volatile(HMX_SET_BIAS("%0") :: "r"(scales));

    asm volatile(HMX_CLRACC_F16());
    asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0")
                 :: "r"(range), "r"(act), "r"(wt));
    asm volatile(HMX_STORE_AFTER_F16("%0", "%1")
                 :: "r"(out), "r"(0) : "memory");
}

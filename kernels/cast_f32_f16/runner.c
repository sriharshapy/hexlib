/* kernels/cast_f32_f16/runner.c -- the executor's entry point.
 *
 * PROTOCOL, little-endian, matching hexlib/exec/runner.py's spec for `cast`:
 *
 *   hexlib_in.bin    int32 n
 *                    float32 x[n]     <- fp32 IN, unlike the other kernels
 *   hexlib_out.bin   fp16    y[n]
 *
 * The input payload is 4 bytes per element and the output 2. That asymmetry is
 * the whole point of the op and the one thing to get right on both sides.
 */
#include <stdio.h>

#include "hexlib/hexlib_harness.h"
#include "kernel_api.h"

/* Encoder shape is 256*1536 = 393216. */
#define RUNNER_MAX_N 524288

static float X[RUNNER_MAX_N]     HEXLIB_ALIGN;
static hexlib_hf Y[RUNNER_MAX_N] HEXLIB_ALIGN;

int main(void) {
    FILE *in = fopen("hexlib_in.bin", "rb");
    if (!in) {
        printf("RUNNER error=no_input\n");
        return 2;
    }
    int n = 0;
    if (fread(&n, sizeof(int), 1, in) != 1) {
        printf("RUNNER error=short_header\n");
        fclose(in);
        return 3;
    }
    if (n <= 0 || n > RUNNER_MAX_N) {
        printf("RUNNER error=bad_n n=%d max=%d\n", n, RUNNER_MAX_N);
        fclose(in);
        return 4;
    }
    if (fread(X, sizeof(float), (size_t) n, in) != (size_t) n) {
        printf("RUNNER error=short_payload n=%d\n", n);
        fclose(in);
        return 5;
    }
    fclose(in);

    unsigned long long kcyc = 0;
    HEXLIB_TIME_KERNEL(kcyc, cast_f32_f16(X, Y, n));

    FILE *out = fopen("hexlib_out.bin", "wb");
    if (!out) {
        printf("RUNNER error=no_output\n");
        return 6;
    }
    if (fwrite(Y, sizeof(hexlib_hf), (size_t) n, out) != (size_t) n) {
        printf("RUNNER error=short_write\n");
        fclose(out);
        return 7;
    }
    fclose(out);
    printf("RUNNER ok n=%d cycles=%llu\n", n, kcyc);
    return 0;
}

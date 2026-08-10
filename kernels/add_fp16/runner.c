/* kernels/add_fp16/runner.c -- the executor's entry point, not the gate's.
 *
 * See kernels/scale_fp16/runner.c for why this is a separate binary from
 * harness.c. In short: the harness must not be able to read host-supplied data,
 * and this must not print the gate's verdict line.
 *
 * PROTOCOL, little-endian, matching hexlib/exec/runner.py's spec for `add`:
 *
 *   hexlib_in.bin    int32 n
 *                    fp16  a[n]
 *                    fp16  b[n]
 *   hexlib_out.bin   fp16  y[n]
 *
 * Both payloads are the same length and that length is n, so there is exactly
 * one way to parse this and no ambiguity for the two sides to disagree about.
 */
#include <stdio.h>

#include "hexlib/hexlib_harness.h"
#include "kernel_api.h"

/* The encoder's adds are [256, 768] = 196608. Sized with headroom so a
 * different resolution does not require recompiling. */
#define RUNNER_MAX_N 262144

static hexlib_hf A[RUNNER_MAX_N] HEXLIB_ALIGN;
static hexlib_hf B[RUNNER_MAX_N] HEXLIB_ALIGN;
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
    if (fread(A, sizeof(hexlib_hf), (size_t) n, in) != (size_t) n
        || fread(B, sizeof(hexlib_hf), (size_t) n, in) != (size_t) n) {
        printf("RUNNER error=short_payload n=%d\n", n);
        fclose(in);
        return 5;
    }
    fclose(in);

    unsigned long long kcyc = 0;
    HEXLIB_TIME_KERNEL(kcyc, add_fp16(A, B, Y, n));

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

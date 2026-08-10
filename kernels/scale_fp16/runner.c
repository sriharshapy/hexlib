/* kernels/scale_fp16/runner.c -- the executor's entry point, not the gate's.
 *
 * WHY A SECOND main() EXISTS. `harness.c` decides whether this kernel is
 * CORRECT: it builds its own inputs, compares against the scalar baseline, and
 * prints a verdict. This file instead lets the kernel be CALLED with data the
 * host chose, so a plan executor can dispatch a real op to it and get the
 * values back. The two are deliberately separate binaries: the harness must not
 * be able to read host-supplied data (its whole value is that it generates its
 * own inputs and cannot be fed a passing answer), and the runner must not print
 * a verdict (nothing has been checked).
 *
 * PROTOCOL, little-endian, matching hexlib/exec/hexagon.py:
 *
 *   hexlib_in.bin    int32   n
 *                    float32 factor
 *                    fp16    x[n]
 *   hexlib_out.bin   fp16    y[n]
 *
 * A short read or a length past the buffer is a hard failure with a distinct
 * exit code, never a partial computation: the host is going to read
 * hexlib_out.bin either way, so a truncated run must not leave a file that
 * looks like a result.
 */
#include <stdio.h>

#include "hexlib/hexlib_harness.h"
#include "kernel_api.h"

/* The encoder's largest scale is [12, 256, 64] = 196608 elements. Sized with
 * headroom rather than exactly, so a different resolution does not require
 * recompiling this file. */
#define RUNNER_MAX_N 262144

static hexlib_hf X[RUNNER_MAX_N] HEXLIB_ALIGN;
static hexlib_hf Y[RUNNER_MAX_N] HEXLIB_ALIGN;

int main(void) {
    FILE *in = fopen("hexlib_in.bin", "rb");
    if (!in) {
        printf("RUNNER error=no_input\n");
        return 2;
    }

    int n = 0;
    float factor = 0.0f;
    if (fread(&n, sizeof(int), 1, in) != 1
        || fread(&factor, sizeof(float), 1, in) != 1) {
        printf("RUNNER error=short_header\n");
        fclose(in);
        return 3;
    }
    if (n <= 0 || n > RUNNER_MAX_N) {
        printf("RUNNER error=bad_n n=%d max=%d\n", n, RUNNER_MAX_N);
        fclose(in);
        return 4;
    }
    if (fread(X, sizeof(hexlib_hf), (size_t) n, in) != (size_t) n) {
        printf("RUNNER error=short_payload n=%d\n", n);
        fclose(in);
        return 5;
    }
    fclose(in);

    unsigned long long kcyc = 0;
    HEXLIB_TIME_KERNEL(kcyc, scale_fp16(X, Y, n, factor));

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

    /* Deliberately NOT hexlib_report's format. Nothing here checked anything,
     * and a line the gate's parser recognises must never come from a binary
     * that only computed. */
    printf("RUNNER ok n=%d cycles=%llu\n", n, kcyc);
    return 0;
}

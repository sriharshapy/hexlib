/* kernels/transpose_th_fp16/runner.c -- the executor's entry point.
 *
 * PROTOCOL, little-endian, matching hexlib/exec/runner.py's spec for `transpose`:
 *
 *   hexlib_in.bin    int32 T, int32 H, int32 D
 *                    fp16  x[T*H*D]
 *   hexlib_out.bin   fp16  y[T*H*D]
 *
 * Three dimensions rather than one element count, because the permutation cannot
 * be performed without knowing all three -- unlike the elementwise kernels,
 * where n is enough.
 */
#include <stdio.h>

#include "hexlib/hexlib_harness.h"
#include "kernel_api.h"

/* Encoder shape is 256*12*64 = 196608. */
#define RUNNER_MAX_N 262144

static hexlib_hf X[RUNNER_MAX_N] HEXLIB_ALIGN;
static hexlib_hf Y[RUNNER_MAX_N] HEXLIB_ALIGN;

int main(void) {
    FILE *in = fopen("hexlib_in.bin", "rb");
    if (!in) {
        printf("RUNNER error=no_input\n");
        return 2;
    }
    int T = 0, H = 0, D = 0;
    if (fread(&T, sizeof(int), 1, in) != 1
        || fread(&H, sizeof(int), 1, in) != 1
        || fread(&D, sizeof(int), 1, in) != 1) {
        printf("RUNNER error=short_header\n");
        fclose(in);
        return 3;
    }
    if (T <= 0 || H <= 0 || D <= 0) {
        printf("RUNNER error=bad_dims T=%d H=%d D=%d\n", T, H, D);
        fclose(in);
        return 4;
    }
    const long n = (long) T * H * D;
    if (n > RUNNER_MAX_N) {
        printf("RUNNER error=too_big n=%ld max=%d\n", n, RUNNER_MAX_N);
        fclose(in);
        return 4;
    }
    if (fread(X, sizeof(hexlib_hf), (size_t) n, in) != (size_t) n) {
        printf("RUNNER error=short_payload n=%ld\n", n);
        fclose(in);
        return 5;
    }
    fclose(in);

    unsigned long long kcyc = 0;
    HEXLIB_TIME_KERNEL(kcyc, transpose_th_fp16(X, Y, T, H, D));

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
    printf("RUNNER ok n=%ld cycles=%llu\n", n, kcyc);
    return 0;
}

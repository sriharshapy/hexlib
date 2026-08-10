/* kernels/add_fp16/harness.c
 *
 * Builds inputs, runs the baseline for reference, times ONLY the kernel call,
 * compares with tolerance, and prints the two lines the driver parses.
 *
 * THE INPUTS ARE CHOSEN TO MAKE THE Vh / Vhf CONFUSION FAIL LOUDLY. One
 * near-miss adds the fp16 bit patterns as 16-bit integers instead of as floats.
 * That produces a wrong answer for essentially any input, but negative values
 * make it unmistakable: fp16 stores sign-magnitude, so an integer add of a
 * negative and a positive value does not even have the right sign. Both signs
 * therefore appear throughout, and neither array is a constant.
 */
#include "hexlib/hexlib_harness.h"
#include "kernel_api.h"

void add_fp16_baseline(const hexlib_hf *, const hexlib_hf *, hexlib_hf *, int);

static hexlib_hf A[ADD_N]   HEXLIB_ALIGN;
static hexlib_hf B[ADD_N]   HEXLIB_ALIGN;
static hexlib_hf Y[ADD_N]   HEXLIB_ALIGN;
static hexlib_hf REF[ADD_N] HEXLIB_ALIGN;

int main(void) {
    for (int i = 0; i < ADD_N; ++i) {
        /* Both signs, a spread of magnitudes, and the two arrays out of phase so
         * no lane ever adds a value to itself. */
        A[i] = (hexlib_hf) ((float) ((i % 29) - 14) * 0.5f);
        B[i] = (hexlib_hf) ((float) ((i % 17) - 8) * 0.25f);
    }

    /* Poison, so a kernel that writes nothing or skips the tail cannot pass. */
    for (int i = 0; i < ADD_N; ++i) {
        Y[i] = (hexlib_hf) 12345.0f;
    }

    add_fp16_baseline(A, B, REF, ADD_N);

    unsigned long long kcyc = 0;
    HEXLIB_TIME_KERNEL(kcyc, add_fp16(A, B, Y, ADD_N));

    int n_wrong = 0;
    double max_err = 0.0;
    for (int i = 0; i < ADD_N; ++i) {
        if (!hexlib_close_f16((float) Y[i], (float) REF[i], 0.02f, 1e-3f)) {
            ++n_wrong;
        }
        double d = (double) (float) Y[i] - (double) (float) REF[i];
        if (d < 0.0) d = -d;
        if (d > max_err) max_err = d;
    }

    hexlib_report(n_wrong == 0, n_wrong, max_err, kcyc);
    return 0;
}

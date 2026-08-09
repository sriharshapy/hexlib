/* include/hexlib/hexlib_harness.h
 *
 * Everything a kernel harness needs to produce a machine-readable verdict:
 * kernel-only cycle timing, 128-byte alignment, fp16/fp32 tolerance compares,
 * and the two output lines the driver parses.
 *
 * WHY THE HARNESS TIMES ONLY THE KERNEL. Harness and CRT startup cost roughly
 * 155,000-190,000 cycles and is nearly constant, so whole-program cycles scale
 * inversely with kernel size and produce spurious 4x-38x differences between
 * runs that differ in nothing that matters. The pcycle delta taken immediately
 * around the kernel call is the only number worth comparing.
 *
 * WHY THE HARNESS ALSO CHECKS CORRECTNESS. One harness then works identically
 * on the simulator and on a device, with no golden-vector transfer and no
 * second code path.
 */
#ifndef HEXLIB_HARNESS_H
#define HEXLIB_HARNESS_H

#include <stdint.h>
#include <stdio.h>

#define HEXLIB_ALIGN __attribute__((aligned(128)))

/* The standalone runtime sets SYSCFG.PCYCLEEN so pcyclelo/hi advance; we set it
 * defensively too, which is harmless if it is already on. */
static inline void hexlib_enable_pcycle(void) {
    unsigned t;
    __asm__ volatile("%0=syscfg\n\t %0=setbit(%0,#5)\n\t syscfg=%0\n\t isync\n\t"
                     : "=&r"(t)); /* bit 5 = PCYCLEEN */
}

static inline unsigned long long hexlib_rdpcyc(void) {
    unsigned lo, hi;
    __asm__ volatile("%0=pcyclelo\n\t %1=pcyclehi\n\t" : "=r"(lo), "=r"(hi));
    return ((unsigned long long) hi << 32) | lo;
}

/* Time STMT into OUT (unsigned long long). */
#define HEXLIB_TIME_KERNEL(OUT, STMT)                    \
    do {                                                 \
        hexlib_enable_pcycle();                          \
        unsigned long long _c0 = hexlib_rdpcyc();        \
        STMT;                                            \
        unsigned long long _c1 = hexlib_rdpcyc();        \
        (OUT) = _c1 - _c0;                               \
    } while (0)

/* ---- tolerance compares ----
 * HVX float is the non-IEEE qf16 path and float operations reorder, so fp16
 * results are compared with a tolerance, never bit-exactly.
 */
static inline int hexlib_close_f16(__fp16 a, __fp16 b, float rel, float abs_tol) {
    float fa = (float) a, fb = (float) b;
    float d = fa - fb;
    if (d < 0.0f) d = -d;
    if (d <= abs_tol) return 1;
    float m = fa < 0.0f ? -fa : fa;
    float mb = fb < 0.0f ? -fb : fb;
    if (mb > m) m = mb;
    return d <= rel * m;
}

static inline int hexlib_close_f32(float a, float b, float rel, float abs_tol) {
    float d = a - b;
    if (d < 0.0f) d = -d;
    if (d <= abs_tol) return 1;
    float m = a < 0.0f ? -a : a;
    float mb = b < 0.0f ? -b : b;
    if (mb > m) m = mb;
    return d <= rel * m;
}

/* ---- the two lines the driver parses ----
 * Both are printed unconditionally. A run that produces neither is a failure,
 * never a pass — the driver has no code path that turns silence into success.
 */
static inline void hexlib_report(int correct, int n_wrong, double max_err,
                                 unsigned long long kernel_cycles) {
    printf("HEXLIB_VERDICT correct=%d wrong=%d maxerr=%.9g\n",
           correct ? 1 : 0, n_wrong, max_err);
    printf("HEXLIB_KCYCLES kernel=%llu\n", kernel_cycles);
    fflush(stdout);
}

#endif /* HEXLIB_HARNESS_H */

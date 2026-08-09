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

/* Time STMT into OUT (unsigned long long).
 *
 * The temporaries are reserved-prefixed because STMT is expanded INSIDE this
 * block: a caller with its own local named `_c0` would have it silently
 * shadowed within the timed statement, substituting a cycle counter for the
 * caller's variable with no compile error. */
#define HEXLIB_TIME_KERNEL(OUT, STMT)                          \
    do {                                                       \
        hexlib_enable_pcycle();                                \
        unsigned long long __hexlib_c0 = hexlib_rdpcyc();      \
        STMT;                                                  \
        unsigned long long __hexlib_c1 = hexlib_rdpcyc();      \
        (OUT) = __hexlib_c1 - __hexlib_c0;                     \
    } while (0)

/* ---- tolerance compares ----
 * HVX float is the non-IEEE qf16 path and float operations reorder, so fp16
 * results are compared with a tolerance, never bit-exactly.
 *
 * WHY THESE TAKE float AND NOT __fp16. hexagon-clang rejects __fp16 as a
 * by-value parameter outright:
 *
 *     error: parameters cannot have __fp16 type; did you forget * ?
 *
 * and it fires on the DECLARATION, so a header with such a signature cannot
 * even be included. Callers pass __fp16 values and the promotion to float
 * happens at the call site, which is what the compiler wants; the body
 * converted to float on its first line anyway, so nothing about the comparison
 * changes. This is also why every fp16 kernel in the v6 corpus passes __fp16 by
 * POINTER and never by value.
 *
 * The alternative -- adding -Xclang -fnative-half-arguments-and-returns -- was
 * rejected: it changes the pinned flag set, and every recorded cycle number was
 * measured without it.
 */
static inline int hexlib_close_f16(float a, float b, float rel, float abs_tol) {
    float d = a - b;
    if (d < 0.0f) d = -d;
    if (d <= abs_tol) return 1;
    float m = a < 0.0f ? -a : a;
    float mb = b < 0.0f ? -b : b;
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

/* hexlib/runtime/simhost/sim_shims.c -- symbols this SDK's local test library
 * does not provide, needed only to link and run under the simulator.
 *
 * HAP_mmap2/HAP_munmap2 (declared in HAP_mem.h for every Hexagon target) are
 * what skel_bufs.c calls, deliberately preferred over the older int-length
 * HAP_mmap/HAP_munmap for size_t safety on large buffers (see
 * task-4-report.md). On real silicon they are backed by QuRT. This SDK's
 * utils/sim_utils/src/test_utils.c -- built into test_util.a, the local-test
 * transport this qexe links against instead of a real device driver --
 * predates the "2" variants and defines only the int-length pair. `nm` across
 * every prebuilt .a in this SDK confirms HAP_mmap2/HAP_munmap2 are defined
 * nowhere for a Hexagon target build.
 *
 * These thin wrappers exist ONLY so the simulator qexe links and runs; they do
 * not change skel_bufs.c's own size_t-safe call, and they are never linked
 * into anything that runs on silicon (a real device skel links against the
 * real QuRT-backed HAP_mmap2, not this file). hexlib_q's own buffers are
 * bounded by MAX_BLOB (16 MiB, see simhost.c), well inside `int` range, so the
 * narrowing here is safe for what this harness actually exercises.
 */
#include "HAP_mem.h"

void *HAP_mmap2(void *addr, size_t len, int prot, int flags, int fd, long offset) {
    return HAP_mmap(addr, (int) len, prot, flags, fd, offset);
}

int HAP_munmap2(void *addr, size_t len) {
    return HAP_munmap(addr, (int) len);
}

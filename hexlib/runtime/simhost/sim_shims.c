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
 *
 * WHAT THIS SHIM MAKES UNTESTABLE UNDER THE SIMULATOR -- read this before
 * trusting a simulator pass on the buffer-mapping path. The HAP_mmap this
 * wraps is test_util.c's, and its ENTIRE body (utils/sim_utils/src/
 * test_utils.c:59) is `return (void *)(uintptr_t)fd;`. It never fails for any
 * nonzero fd, and the "address" it returns IS the fd's bit pattern, not a
 * real mapping of anything.
 *
 *   1. hexlib_bufs_register's own failure path (HEXLIB_DSP_ERR_MMAP_FAILED,
 *      in skel_bufs.c, taken when HAP_mmap2 returns 0 or -1) is UNREACHABLE
 *      under the simulator for any realistic nonzero fd. A simulator pass
 *      proves nothing about that path; only silicon, where HAP_mmap2 talks to
 *      a real mapper that can genuinely fail, can exercise it.
 *
 *   2. Because the fd doubles as the "address" here, the simulator cannot
 *      tell "resolved the fd through hexlib_bufs_register's table" apart from
 *      "happened to use the fd as an address" by comparing VALUES alone --
 *      both would produce the same base pointer for a registered fd. What
 *      still discriminates the two is the table LOOKUP itself, not the
 *      value: hexlib_bufs_map (skel_bufs.c) consults ctx->mmap, and that
 *      table is populated ONLY by hexlib_bufs_register. A never-registered fd
 *      has no entry regardless of what HAP_mmap would have returned for it,
 *      so hexlib_bufs_map still refuses it (HEXLIB_DSP_ERR_UNMAPPED). That is
 *      why the --unmapped test (see simhost.c) remains a real discriminator
 *      even with this fd-as-address stand-in underneath it: it is testing
 *      whether the lookup happened at all, not what value came out of it.
 */
#include "HAP_mem.h"

void *HAP_mmap2(void *addr, size_t len, int prot, int flags, int fd, long offset) {
    return HAP_mmap(addr, (int) len, prot, flags, fd, offset);
}

int HAP_munmap2(void *addr, size_t len) {
    return HAP_munmap(addr, (int) len);
}

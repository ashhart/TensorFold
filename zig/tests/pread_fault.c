// Inject bounded delays and partial reads into the test process only.
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

static ssize_t (*system_pread)(int, void *, size_t, off_t);
static pthread_once_t ready = PTHREAD_ONCE_INIT;
static _Atomic uint64_t calls;
static unsigned delay_us;
static size_t read_limit;
static off_t eof_at;
static int failure_kind;

static void resolve(void) {
    system_pread = dlsym(RTLD_NEXT, "pread");
    if (!system_pread) abort();
}

void tf_probe_config(unsigned delay, size_t limit, int inject_eof) {
    pthread_once(&ready, resolve);
    delay_us = delay;
    read_limit = limit;
    eof_at = inject_eof ? 32768 : 0;
    failure_kind = inject_eof;
    atomic_store(&calls, 0);
}

uint64_t tf_probe_calls(void) { return atomic_load(&calls); }

ssize_t pread(int fd, void *buffer, size_t size, off_t offset) {
    pthread_once(&ready, resolve);
    atomic_fetch_add(&calls, 1);
    if (delay_us) usleep(delay_us);
    if (eof_at && offset >= eof_at) {
        if (failure_kind == 1) return 0;
        errno = failure_kind == 2 ? EIO : EINTR;
        return -1;
    }
    if (read_limit && size > read_limit) size = read_limit;
    return system_pread(fd, buffer, size, offset);
}

// Replace the test file bytes after the loader returns.
int tf_probe_poison(const char *path) {
    int fd = open(path, O_RDWR);
    if (fd < 0) return -1;
    off_t length = lseek(fd, 0, SEEK_END);
    if (length < 0) { close(fd); return -1; }
    char zeros[4096] = {0};
    for (off_t at = 0; at < length; ) {
        size_t size = (size_t)(length - at);
        if (size > sizeof zeros) size = sizeof zeros;
        ssize_t n = pwrite(fd, zeros, size, at);
        if (n <= 0) { close(fd); return -1; }
        at += n;
    }
    int result = fsync(fd);
    close(fd);
    return result;
}

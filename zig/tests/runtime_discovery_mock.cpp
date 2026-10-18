#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif
static int ready;
static char marker[4096];
static void record(const char *kind) {
    if (!ready) _Exit(72);
    FILE *out = fopen(marker, "ab");
    if (!out) _Exit(73);
    fputs(kind, out);
    fclose(out);
}
static void exit_callback(void) { record("atexit\n"); }
__attribute__((constructor)) static void loaded(void) {
    if (atexit(exit_callback) != 0) _Exit(74);
}
__attribute__((destructor)) static void unloaded(void) { record("destructor\n"); }
struct ThreadExit {
    ~ThreadExit() { record("thread_local\n"); }
};
static thread_local ThreadExit thread_exit;
extern "C" EXPORT int tf_probe_ready(const char *path) {
    const size_t n = strlen(path);
    if (n >= sizeof(marker)) return -1;
    memcpy(marker, path, n + 1);
    (void)&thread_exit;
    ready = 1;
    return 0;
}

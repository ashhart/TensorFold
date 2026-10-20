// Owned ABI fixture for the Linux loader layout test; no FFmpeg, model or device.
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    int64_t frames;
    int rate_num, rate_den, width, height, counted;
} Info;
typedef int (*FrameFn)(void *, int, int, const uint8_t *, int, int, int);

const char *tf_video_version(void) { return "9.0.2"; }

int tf_video_open(const uint8_t *bytes, size_t len, void **handle, Info *info) {
    static int container;
    *handle = NULL;
    if (len != 4 || memcmp(bytes, "stub", 4)) return -1;
    *handle = &container;
    *info = (Info){1, 1, 1, 1, 1, 0};
    return 0;
}

#ifndef TFVIDEO_STUB_NO_FRAMES
int tf_video_frames(void *handle, const int *indices, int n, FrameFn take, void *ctx) {
    static const uint8_t rgb[] = {1, 2, 3};
    (void)handle;
    for (int k = 0; k < n; ++k) {
        if (indices[k] != 0) return -6;
        if (take(ctx, k, indices[k], rgb, 1, 1, 3)) return -7;
    }
    return n;
}
#endif

void tf_video_close(void *handle) { (void)handle; }

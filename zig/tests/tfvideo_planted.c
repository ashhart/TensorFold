// Planted for check-video-exports-linux: a C table that x86_64 asm reads PC-relative, as FFmpeg's assembly does.
#include <stdint.h>

const uint16_t tfvideo_planted_table[4] = {1, 2, 3, 4};

uint32_t tfvideo_planted_read(void) {
    uint32_t v;
#if defined(__x86_64__)
    __asm__("movzwl tfvideo_planted_table(%%rip), %0" : "=r"(v));
#else
    v = tfvideo_planted_table[0];
#endif
    return v;
}

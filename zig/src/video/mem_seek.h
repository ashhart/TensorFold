#ifndef TF_MEM_SEEK_H
#define TF_MEM_SEEK_H

#include <stdint.h>
#include <stddef.h>

// Checked base + off in the buffer's [0, len] range, without signed overflow.
static int64_t tf_mem_seek_at(size_t len, int64_t off, int64_t base) {
    if (off > 0 && (uint64_t)off > (uint64_t)((int64_t)len - base)) return -1;
    if (off < 0 && (uint64_t)(-(off + 1)) + 1 > (uint64_t)base) return -1;
    return base + off;
}

#endif

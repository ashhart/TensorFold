#ifndef TENSORFOLD_OP_LAUNCH_SHAPE_H
#define TENSORFOLD_OP_LAUNCH_SHAPE_H

#include <stdint.h>
#include <stdbool.h>

static inline uint32_t tf_launch_blocks(uint64_t count, uint32_t threads) {
    if (threads == 0) return 0;
    const uint64_t blocks = count / threads + (count % threads != 0);
    return (uint32_t)(blocks > 65535 ? 65535 : blocks);
}

static inline bool tf_matrix_count(uint64_t rows, uint64_t columns, uint64_t element_bytes, uint64_t* count) {
    if (rows == 0) {
        *count = 0;
        return true;
    }
    if (element_bytes == 0 || columns == 0 || columns > UINT64_MAX / rows) return false;
    *count = rows * columns;
    return *count <= UINT64_MAX / element_bytes;
}

#endif

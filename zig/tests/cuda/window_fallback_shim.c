// The host test calls the production header's predicate through this C ABI.
#include "window_fallback.h"

int tf_wrows_test_head_affected(const uint16_t* out, int width, int first, int stride) {
    return tf_wrows_head_affected(out, width, first, stride);
}

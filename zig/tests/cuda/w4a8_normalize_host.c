#include "../../kernels/cuda/w4a8_normalize.h"
float tf_w4a8_normalize_host(float v, float amax) {
    const float inv = amax > 0.0f ? 127.0f / amax : 1.0f;
    return tf_w4a8_normalized(v, amax, inv);
}

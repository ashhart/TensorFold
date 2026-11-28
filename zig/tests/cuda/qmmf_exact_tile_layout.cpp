// Compile the scale-address helper used by the CUDA body, with no CUDA instructions.
#include "qmmf_fp4_wide_layout.h"
extern "C" size_t tf_qmmf_wide_scale_offset(size_t tile, int groups, int group, int chunk, bool wrong) {
    if (wrong) return (tile * groups + group) * 512 + chunk * 16;
    return tf_qmmf_wide_layout::offset(tile, groups, group, chunk, 256);
}
extern "C" int tf_qmmf_wide_shared() { return tf_qmmf_wide_layout::shared_bytes; }
extern "C" int tf_qmmf_wide_order_witness() {
    volatile float ascending = 0.0f, reverse = 0.0f;
    const float terms[4] = {16777216.0f, 1.0f, -16777216.0f, -1.0f};
    for (int i = 0; i < 4; ++i) ascending = ascending + terms[i];
    for (int i = 3; i >= 0; --i) reverse = reverse + terms[i];
    return ascending == -1.0f && reverse == 1.0f;
}

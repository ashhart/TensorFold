// The two original 64-column scale planes, retained in their packed order.
#pragma once
#include <stddef.h>
#ifdef __CUDACC__
#define TF_QMMF_LAYOUT_HD __host__ __device__
#else
#define TF_QMMF_LAYOUT_HD
#endif
namespace tf_qmmf_wide_layout {
TF_QMMF_LAYOUT_HD constexpr size_t offset(size_t tile64, int groups, int group, int chunk16, int plane_bytes) {
    const int per = plane_bytes / 16;
    return ((tile64 + chunk16 / per) * groups + group) * plane_bytes + (chunk16 % per) * 16;
}
constexpr int shared_bytes = 4 * (16 * 64 * 2 + 128 * 64 / 2 + 128 * 4);
}
#undef TF_QMMF_LAYOUT_HD

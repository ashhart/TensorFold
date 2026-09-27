// Grouped EXL3 expert GEMV instances for codebook 1 (mcg).
#include "experts_grouped.cuh"

namespace tf_exl3x {
template void grouped_launch<1>(const GroupedArgs&, cudaStream_t);
template void dequant_launch<1>(const uint32_t*, half*, int, int, int, cudaStream_t);
}  // namespace tf_exl3x

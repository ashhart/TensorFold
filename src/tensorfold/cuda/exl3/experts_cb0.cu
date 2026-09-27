// Grouped EXL3 expert GEMV instances for codebook 0 (3inst).
#include "experts_grouped.cuh"

namespace tf_exl3x {
template void grouped_launch<0>(const GroupedArgs&, cudaStream_t);
template void dequant_launch<0>(const uint32_t*, half*, int, int, int, cudaStream_t);
}  // namespace tf_exl3x

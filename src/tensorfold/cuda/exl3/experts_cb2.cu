// Grouped EXL3 expert GEMV instances for codebook 2 (mul1).
#include "experts_prompt.cuh"

namespace tf_exl3x {
template void grouped_launch<2>(const GroupedArgs&, cudaStream_t);
template void prompt_launch<2>(const PromptArgs&, int, cudaStream_t);
template void prompt_down_launch<2>(const DownArgs&, int, cudaStream_t);
template void dequant_launch<2>(const uint32_t*, half*, int, int, int, cudaStream_t);
}  // namespace tf_exl3x

// Each mHC control omits one operation and must fail its shape gate.
#define MHC_CONTROL
#define qwen35_mhc_enter tf_ctl_qwen35_mhc_enter
#define qwen35_mhc_route tf_ctl_qwen35_mhc_route
#define qwen35_mhc_merge tf_ctl_qwen35_mhc_merge
#define qwen35_mhc_leave tf_ctl_qwen35_mhc_leave
namespace tf_ctl_mhc {
#include "qwen35_mhc.cu"
}
#undef qwen35_mhc_enter
#undef qwen35_mhc_route
#undef qwen35_mhc_merge
#undef qwen35_mhc_leave
#undef MHC_CONTROL

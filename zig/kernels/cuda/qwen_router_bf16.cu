// ModelOpt-only expert-first router; the standard MoE translation unit stays unchanged.
#include "moe.cu"
#define TF_MOE_ROUTER_CONTROLS
#include "moe_router_expert.cuh"

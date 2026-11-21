// ModelOpt-only prompt recurrence; the standard GDN translation unit stays unchanged.
#include "gdn_prefill.cu"
#define TF_GDN_CHAIN_CONTROLS
#include "gdn_prefill_norm.cuh"

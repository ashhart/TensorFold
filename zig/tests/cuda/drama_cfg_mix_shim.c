// The host test runs the production d3_cfg_mix element body through this C ABI.
#include "drama3_cfg.h"

void d3_cfg_mix_test_at(float* conditional, const float* unconditional, int n, float gamma, int i) {
    d3_cfg_mix_at(conditional, unconditional, n, gamma, i);
}

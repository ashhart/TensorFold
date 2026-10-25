#include <metal_stdlib>
using namespace metal;

struct D3CfgArgs { uint count; float gamma; };

// Candidate logits include semantic tokens and end-of-speech; residual codec heads remain conditional.
[[max_total_threads_per_threadgroup(256)]] kernel void d3_cfg_mix(device float* conditional [[buffer(0)]],
                       device const float* unconditional [[buffer(1)]],
                       constant D3CfgArgs& args [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= args.count || args.gamma == 1.0f) return;
    float c = conditional[i];
    conditional[i] = c + (args.gamma - 1.0f) * (c - unconditional[i]);
}

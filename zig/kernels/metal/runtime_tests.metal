// Small kernels for the Metal runtime's own GPU tests and dispatch benchmarks.
#include <metal_stdlib>
using namespace metal;

kernel void tf_vadd(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                    device float* c [[buffer(2)]], constant uint& n [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
  if (i < n) c[i] = a[i] + b[i];
}

// out[i] = a[i] ^ (b[n - 1 - i] * 2654435761): reads both inputs far from its own index
kernel void tf_mix(device const uint* a [[buffer(0)]], device const uint* b [[buffer(1)]],
                   device uint* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
  if (i < n) out[i] = a[i] ^ (b[n - 1 - i] * 2654435761u);
}

// y[i] = x[i] * p[0] + p[1]; arguments come from buffers so indirect commands can change them
kernel void tf_affine(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                      device const float* p [[buffer(2)]], device const uint* n [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
  if (i < n[0]) y[i] = x[i] * p[0] + p[1];
}

// one dependent step: every thread updates its own word in place
kernel void tf_step(device uint* x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  x[i] = x[i] * 3u + 1u;
}

// a host/GPU round: out[i] = in[i] iterated `iters` times through an LCG-like map (GPU work to overlap)
kernel void tf_round(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]],
                     constant uint& n [[buffer(2)]], constant uint& iters [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  uint v = in[i];
  for (uint k = 0; k < iters; k++) v = v * 1664525u + 1013904223u;
  out[i] = v;
}

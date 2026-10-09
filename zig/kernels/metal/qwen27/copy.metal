// Copies preserve BF16 bits while projection stacks become independent row-major views.
#include <metal_stdlib>
using namespace metal;

struct Slice { uint rows, width, stride, offset; };

kernel void q27_slice(device const ushort* X [[buffer(0)]], constant Slice& p [[buffer(1)]],
                      device ushort* Y [[buffer(2)]], uint2 t [[thread_position_in_grid]]) {
  if (t.x < p.width && t.y < p.rows) Y[t.y * p.width + t.x] = X[t.y * p.stride + p.offset + t.x];
}

struct Add { uint rows, width; };

kernel void q27_add(device const bfloat* X [[buffer(0)]], device const bfloat* R [[buffer(1)]],
                    constant Add& p [[buffer(2)]], device bfloat* Y [[buffer(3)]],
                    uint2 t [[thread_position_in_grid]]) {
  if (t.x < p.width && t.y < p.rows) {
    const uint i = t.y * p.width + t.x;
    Y[i] = bfloat(float(X[i]) + float(R[i]));
  }
}

struct Gather { uint rows, width, capacity, planes; };

kernel void q27_tap_gather(device const ushort* X [[buffer(0)]], device const uint* kept [[buffer(1)]],
                           constant Gather& p [[buffer(2)]], device ushort* Y [[buffer(3)]],
                           uint2 t [[thread_position_in_grid]]) {
  if (t.x >= p.width * p.planes || t.y >= p.rows || kept[t.y] >= p.capacity) return;
  const uint plane = t.x / p.width, column = t.x % p.width;
  Y[t.y * (p.width * p.planes) + t.x] = X[(plane * p.capacity + kept[t.y]) * p.width + column];
}

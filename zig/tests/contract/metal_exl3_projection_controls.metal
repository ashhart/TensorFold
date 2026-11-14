#include <metal_stdlib>
using namespace metal;

struct Exl3ContractArgs { uint k, n, rows, stress, fault; float base; };

inline float exl3_contract_weight(const device half* tiles, constant Exl3ContractArgs& a, uint i, uint j) {
  const uint cell = (i % 16) * 16 + j % 16;
  if (a.stress) return float(tiles[(j / 16) * 256 + cell]);
  const uint kb = i / 16;
  uint special = 3;
  if (kb == 0) special = 0;
  else if (kb == a.k / 32) special = 1;
  else if (kb == a.k / 16 - 1) special = 2;
  return special < 3 ? float(tiles[(special * (a.n / 16) + j / 16) * 256 + cell]) : a.base;
}

kernel void exl3_projection_contract_fault(device half* x [[buffer(0)]],
                                           const device half* tiles [[buffer(1)]],
                                           device float* y [[buffer(2)]],
                                           constant Exl3ContractArgs& a [[buffer(3)]],
                                           uint tid [[thread_position_in_grid]]) {
  if (tid >= a.rows * a.n) return;
  if (a.fault == 2 && tid == 0) x[0] += half(1);
  const uint r = tid / a.n, j = tid % a.n;
  const uint end = a.fault == 3 ? a.k - 16 : a.k;
  float sum = 0;
  half small = 0;
  for (uint i = 0; i < end; ++i) {
    float v = float(x[r * a.k + i]);
    if (a.fault == 1 && !isfinite(v)) v = 0;
    const float w = exl3_contract_weight(tiles, a, i, j);
    if (a.fault == 4) small = half(float(small) + v * w);
    else sum = fma(v, w, sum);
  }
  y[tid] = a.fault == 4 ? float(small) : sum;
}

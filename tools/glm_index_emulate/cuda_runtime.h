// Host stand-ins for the CUDA names glm_index_*.cuh use: a block's threads are OS threads, __syncthreads a barrier.
#pragma once
#include <algorithm>
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstring>
#define __global__
#define __device__
#define __forceinline__ inline
#define __restrict__
#define __launch_bounds__(x)
#define __shared__ static
#define __align__(x) __attribute__((aligned(x)))
struct dim3 { unsigned x, y, z; };
struct float4 { float x, y, z, w; };
inline thread_local dim3 threadIdx, blockIdx;
namespace emu {
inline std::barrier<>* block_bar;
inline std::barrier<>* warp_bar[32];
inline uint32_t xbuf[1024];
inline const void* pbuf[1024];
}  // namespace emu
inline void __syncthreads() { emu::block_bar->arrive_and_wait(); }
inline void warp_sync() { emu::warp_bar[threadIdx.x >> 5]->arrive_and_wait(); }
template <class T>
inline T shuffle(T v, int src_of(int lane)) {
  const int t = (int)threadIdx.x, l = t & 31, base = t - l;
  uint32_t w;
  std::memcpy(&w, &v, 4);
  emu::xbuf[t] = w;
  warp_sync();
  const int s = src_of(l);
  std::memcpy(&v, &emu::xbuf[base + (s >= 0 ? s : l)], 4);
  warp_sync();
  return v;
}
inline int __shfl_up_sync(unsigned, int v, int d) {
  static thread_local int delta;
  delta = d;
  return shuffle(v, [](int l) { return l >= delta ? l - delta : -1; });
}
inline float __shfl_xor_sync(unsigned, float v, int o) {
  static thread_local int mask;
  mask = o;
  return shuffle(v, [](int l) { return l ^ mask; });
}
inline int atomicAdd(int* p, int v) { return std::atomic_ref<int>(*p).fetch_add(v); }
inline int atomicMax(int* p, int v) {
  std::atomic_ref<int> a(*p);
  int old = a.load();
  while (old < v && !a.compare_exchange_weak(old, v)) {}
  return old;
}
inline int __float_as_int(float f) { int i; std::memcpy(&i, &f, 4); return i; }
inline float __fadd_rn(float a, float b) { return a + b; }
inline float __fmul_rn(float a, float b) { return a * b; }
inline float __fmaf_rn(float a, float b, float c) { return std::fma(a, b, c); }
using std::fmaxf;
using std::max;
using std::min;

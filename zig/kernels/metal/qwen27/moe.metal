// The routed block: route, the sort for the tensor units, the experts' SwiGLU and the combine (bf16 as the reference).
#include <metal_stdlib>
using namespace metal;

struct RouteArgs { int experts, topk, ld, renorm; }; // PICK, WEIGHT [rows, topk + 1]: the picks, then the shared expert (id experts, its sigmoid)
struct SortArgs { int slots, experts; };
struct GluArgs { int width; };
struct CombineArgs { int hidden, topk, shared_in_out, pad; };

// A row a threadgroup: fp32 softmax (to bf16) over LOGITS[row, 0..experts), top-k by logit (lower id on ties).
[[kernel]] void q27_route(const device bfloat* LOGITS [[buffer(0)]], device uint* PICK [[buffer(1)]],
                          device float* WEIGHT [[buffer(2)]], constant RouteArgs& a [[buffer(3)]],
                          uint r [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint nsg [[simdgroups_per_threadgroup]]) {
  threadgroup float part[32];
  threadgroup float best_v[32];
  threadgroup int best_i[32];
  threadgroup int sel[16];
  threadgroup float w[16];
  const device bfloat* z = LOGITS + size_t(r) * a.ld;
  const bool mine = int(t) < a.experts;
  const float x = mine ? float(z[t]) : -INFINITY;
  // the row's max, then the sum of exp
  float m = simd_max(x);
  if (lane == 0) part[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = part[0];
  for (uint g = 1; g < nsg; g++) m = max(m, part[g]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float ex = mine ? precise::exp(x - m) : 0.0f;
  float s = simd_sum(ex);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  s = 0.0f;
  for (uint g = 0; g < nsg; g++) s += part[g];
  const float p = float(bfloat(ex / s));
  float key = x;
  for (int k = 0; k < a.topk; k++) {
    float v = key;
    int id = mine ? int(t) : 0x7fffffff;
    for (ushort o = 16; o > 0; o >>= 1) {
      const float ov = simd_shuffle_down(v, o);
      const int oi = simd_shuffle_down(id, o);
      if (ov > v || (ov == v && oi < id)) {
        v = ov;
        id = oi;
      }
    }
    if (lane == 0) {
      best_v[sg] = v;
      best_i[sg] = id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      float bv = best_v[0];
      int bi = best_i[0];
      for (uint g = 1; g < nsg; g++)
        if (best_v[g] > bv || (best_v[g] == bv && best_i[g] < bi)) {
          bv = best_v[g];
          bi = best_i[g];
        }
      sel[k] = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(t) == sel[k]) {
      w[k] = p;
      key = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (t == 0) {
    float sum = 0.0f;
    for (int k = 0; k < a.topk; k++) sum += w[k];
    const float norm = float(bfloat(sum));
    const int stride = a.topk + 1;
    for (int k = 0; k < a.topk; k++) {
      PICK[size_t(r) * stride + k] = uint(sel[k]);
      WEIGHT[size_t(r) * stride + k] = a.renorm ? float(bfloat(w[k] / norm)) : w[k];
    }
    const float g = float(z[a.experts]);
    PICK[size_t(r) * stride + a.topk] = uint(a.experts);
    WEIGHT[size_t(r) * stride + a.topk] = float(bfloat(1.0f / (1.0f + precise::exp(-g))));
  }
}

// Slots (id < experts) sorted by expert, stably, a thread an expert (at most 256): OFFS (starts, end), ORDER, WS.
[[kernel]] void q27_route_sort(const device uint* PICK [[buffer(0)]], const device float* WEIGHT [[buffer(1)]],
                               device int* OFFS [[buffer(2)]], device int* ORDER [[buffer(3)]],
                               device float* WS [[buffer(4)]], constant SortArgs& a [[buffer(5)]],
                               uint e [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]]) {
  threadgroup atomic_int counts[256]; // counted by all threads together
  const bool expert = int(e) < a.experts;
  atomic_store_explicit(counts + e, 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = int(e); i < a.slots; i += int(n))
    if (PICK[i] < uint(a.experts)) atomic_fetch_add_explicit(counts + PICK[i], 1, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!expert) return;
  int at = 0;
  for (int j = 0; j < int(e); j++) at += atomic_load_explicit(counts + j, memory_order_relaxed);
  OFFS[e] = at;
  if (e == 0) {
    int total = 0;
    for (int j = 0; j < a.experts; j++) total += atomic_load_explicit(counts + j, memory_order_relaxed);
    OFFS[a.experts] = total;
  }
  for (int i = 0; i < a.slots; i++) {
    if (PICK[i] != e) continue;
    ORDER[at] = i;
    WS[at] = WEIGHT[i];
    at++;
  }
}

// Sorted slot i's input row: X[ORDER[i] / wk.y], wk.y = topk + 1 (a row's slots: its picks, then the shared expert's).
[[kernel]] void q27_route_gather(const device bfloat* X [[buffer(0)]], const device int* ORDER [[buffer(1)]],
                                 device bfloat* Y [[buffer(2)]], constant int2& wk [[buffer(3)]],
                                 uint2 g [[thread_position_in_grid]]) {
  Y[size_t(g.y) * wk.x + g.x] = X[size_t(ORDER[g.y] / wk.y) * wk.x + g.x];
}

// act = silu(gate) * up in fp32, rounded once, as the dense block's q27_mlp.
[[kernel]] void q27_expert_glu(const device bfloat* G [[buffer(0)]], const device bfloat* U [[buffer(1)]],
                               device bfloat* OUT [[buffer(2)]], constant GluArgs& a [[buffer(3)]],
                               uint2 g [[thread_position_in_grid]]) {
  const size_t i = size_t(g.y) * a.width + g.x;
  const float gate = float(G[i]);
  OUT[i] = bfloat(gate / (1.0f + exp(-gate)) * float(U[i]));
}

// OUT[r] = bf16(bf16(sum_k bf16(Y[k] * w_k)) + bf16(sigma * shared)); shared: OUT[r] (shared_in_out) or Y's slot topk.
[[kernel]] void q27_expert_combine(const device bfloat* Y [[buffer(0)]], const device float* WEIGHT [[buffer(1)]],
                                   device bfloat* OUT [[buffer(2)]], constant CombineArgs& a [[buffer(3)]],
                                   uint2 g [[thread_position_in_grid]]) {
  const size_t r = g.y, j = g.x, stride = a.topk + 1;
  float routed = 0.0f;
  for (int k = 0; k < a.topk; k++)
    routed += float(bfloat(float(Y[(r * stride + k) * a.hidden + j]) * WEIGHT[r * stride + k]));
  const float sigma = WEIGHT[r * stride + a.topk];
  const float row = a.shared_in_out ? float(OUT[r * a.hidden + j]) : float(Y[(r * stride + a.topk) * a.hidden + j]);
  OUT[r * a.hidden + j] = bfloat(float(bfloat(routed)) + float(bfloat(sigma * row)));
}

#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
struct RouteArgs {
  int experts;
  int topk;
  float scale;
  int trace_layer;
};

// The better of two (value, id) pairs in ds_route's order: the higher value, the lower id on ties.
inline bool route_better(float ov, int oi, float v, int id) { return ov > v || (ov == v && oi < id); }

// Routing a threadgroup a row: top-k of sqrt(softplus) + bias (lower id on ties, bias_vl on images), expert order.

// Each simdgroup takes its own top-k in that order without the threadgroup; simdgroup 0 then takes the top-k of

// those candidates (nsg * topk <= 96), which are the row's in the same order, so the weights sum in pick order.

// Numeric FP32 ordering expressed as unsigned bits, with both zeros equal.
inline uint tf_route_order(float value) {
  uint u = as_type<uint>(value);
  if ((u & 0x7fffffffu) == 0) u = 0;
  return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}
inline int tf_route_rank(float value, int id) {
  const uint key = tf_route_order(value);
  const uint top = simd_max(key);
  return simd_min(key == top ? id : 0x7fffffff);
}

[[kernel]] void tf_route_reduce(const device float* LOGITS [[buffer(0)]], const device float* BIAS [[buffer(1)]],
                         device uint* PICK [[buffer(2)]], device float* WEIGHT [[buffer(3)]],
                         constant RouteArgs& a [[buffer(4)]], const device int* IMG [[buffer(5)]],
                         const device float* BIAS_VL [[buffer(6)]], device uint* TRACE [[buffer(7)]], uint r [[threadgroup_position_in_grid]],
                         uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
  threadgroup uint bad[32];
  threadgroup float cv[32][16];
  threadgroup int ci[32][16];
  threadgroup float cs[32][16];
  const device float* z = LOGITS + size_t(r) * a.experts;
  float score = 0.0f, key = -INFINITY;
  if (int(t) < a.experts) {
    const float x = z[t];
    score = precise::sqrt(x > 20.0f ? x : precise::log(1.0f + precise::exp(x)));
    key = score + (IMG[r] ? BIAS_VL[t] : BIAS[t]);  // an image span's rows steer by their own bias
  }
  const int myid = int(t) < a.experts ? int(t) : 0x7fffffff;
  const bool ordinary = simd_all(isfinite(key));
  if (lane == 0) bad[sg] = !ordinary;
  for (int k = 0; k < a.topk; k++) {
    float v = key;
    int id = myid;
    if (ordinary) {
      id = tf_route_rank(v, id);
      v = simd_shuffle(v, ushort(id & 31));
    } else {
    for (ushort o = 16; o > 0; o >>= 1) {
      const float ov = simd_shuffle_down(v, o);
      const int oi = simd_shuffle_down(id, o);
      if (route_better(ov, oi, v, id)) {
        v = ov;
        id = oi;
      }
    }
    }
    v = simd_broadcast(v, 0);
    id = simd_broadcast(id, 0);
    const float s = simd_broadcast(id == myid ? score : 0.0f, ushort(id & 31));
    if (lane == 0) {
      cv[sg][k] = v;
      ci[sg][k] = id;
      cs[sg][k] = s;
    }
    if (id == myid) key = -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  const bool all_ordinary = simd_all(lane >= nsg || !bad[lane]);
  // simdgroup 0: lane l holds candidates l, l + 32 and l + 64
  const int total = int(nsg) * a.topk;
  float lv[3], ls[3];
  int li[3];
  for (int j = 0; j < 3; j++) {
    const int c = int(lane) + 32 * j;
    const bool live = c < total;
    lv[j] = live ? cv[c / a.topk][c % a.topk] : -INFINITY;
    li[j] = live ? ci[c / a.topk][c % a.topk] : 0x7fffffff;
    ls[j] = live ? cs[c / a.topk][c % a.topk] : 0.0f;
  }
  int s2[16];
  float w2[16];
  for (int k = 0; k < a.topk; k++) {
    float v = lv[0];
    int id = li[0];
    for (int j = 1; j < 3; j++)
      if (route_better(lv[j], li[j], v, id)) {
        v = lv[j];
        id = li[j];
      }
    if (all_ordinary) {
      id = tf_route_rank(v, id);
    } else {
    for (ushort o = 16; o > 0; o >>= 1) {
      const float ov = simd_shuffle_down(v, o);
      const int oi = simd_shuffle_down(id, o);
      if (route_better(ov, oi, v, id)) {
        v = ov;
        id = oi;
      }
    }
    }
    id = simd_broadcast(id, 0);
    float mine = 0.0f;
    bool has = false;
    for (int j = 0; j < 3; j++)
      if (li[j] == id) {
        mine = ls[j];
        has = true;
        lv[j] = -INFINITY;
        li[j] = 0x7fffffff;
      }
    s2[k] = id;
    w2[k] = simd_broadcast(mine, ushort(simd_min(has ? lane : 32u)));
  }
  if (lane == 0) {
    float sum = 0.0f;
    for (int k = 0; k < a.topk; k++) sum += w2[k];
    for (int k = 0; k < a.topk; k++) w2[k] = w2[k] / (sum + 1e-20f) * a.scale;
    for (int i = 1; i < a.topk; i++)
      for (int j = i; j > 0 && s2[j - 1] > s2[j]; j--) {
        const int ts = s2[j];
        s2[j] = s2[j - 1];
        s2[j - 1] = ts;
        const float tw = w2[j];
        w2[j] = w2[j - 1];
        w2[j - 1] = tw;
      }
    for (int k = 0; k < a.topk; k++) {
      PICK[r * a.topk + k] = uint(s2[k]);
      if (a.trace_layer >= 0) TRACE[(a.trace_layer * 6 + r) * a.topk + k] = uint(s2[k]);
      WEIGHT[r * a.topk + k] = w2[k];
    }
  }
}


[[kernel]] void tf_route_merge(const device float* LOGITS [[buffer(0)]], const device float* BIAS [[buffer(1)]],
                         device uint* PICK [[buffer(2)]], device float* WEIGHT [[buffer(3)]],
                         constant RouteArgs& a [[buffer(4)]], const device int* IMG [[buffer(5)]],
                         const device float* BIAS_VL [[buffer(6)]], device uint* TRACE [[buffer(7)]], uint r [[threadgroup_position_in_grid]],
                         uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
  threadgroup uint bad[32];
  threadgroup float cv[32][16];
  threadgroup int ci[32][16];
  threadgroup float cs[32][16];
  const device float* z = LOGITS + size_t(r) * a.experts;
  float score = 0.0f, key = -INFINITY;
  if (int(t) < a.experts) {
    const float x = z[t];
    score = precise::sqrt(x > 20.0f ? x : precise::log(1.0f + precise::exp(x)));
    key = score + (IMG[r] ? BIAS_VL[t] : BIAS[t]);  // an image span's rows steer by their own bias
  }
  const int myid = int(t) < a.experts ? int(t) : 0x7fffffff;
  const bool ordinary = simd_all(isfinite(key));
  if (lane == 0) bad[sg] = !ordinary;
  for (int k = 0; k < a.topk; k++) {
    float v = key;
    int id = myid;
    if (ordinary) {
      id = tf_route_rank(v, id);
      v = simd_shuffle(v, ushort(id & 31));
    } else {
    for (ushort o = 16; o > 0; o >>= 1) {
      const float ov = simd_shuffle_down(v, o);
      const int oi = simd_shuffle_down(id, o);
      if (route_better(ov, oi, v, id)) {
        v = ov;
        id = oi;
      }
    }
    }
    v = simd_broadcast(v, 0);
    id = simd_broadcast(id, 0);
    const float s = simd_broadcast(id == myid ? score : 0.0f, ushort(id & 31));
    if (lane == 0) {
      cv[sg][k] = v;
      ci[sg][k] = id;
      cs[sg][k] = s;
    }
    if (id == myid) key = -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  const bool all_ordinary = simd_all(lane >= nsg || !bad[lane]);
  int s2[16];
  float w2[16];
  if (all_ordinary) {
    int at = 0;
    const bool live = lane < nsg;
    for (int k = 0; k < a.topk; k++) {
      const bool used = live && at < a.topk;
      const float v = used ? cv[lane][at] : -INFINITY;
      const int candidate = used ? ci[lane][at] : 0x7fffffff;
      const float score = used ? cs[lane][at] : 0.0f;
      const int id = tf_route_rank(v, candidate);
      const uint from = simd_min(candidate == id ? lane : 32u);
      s2[k] = id;
      w2[k] = simd_broadcast(score, ushort(from));
      at += (lane == from);
    }
  } else {
  // simdgroup 0: lane l holds candidates l, l + 32 and l + 64
  const int total = int(nsg) * a.topk;
  float lv[3], ls[3];
  int li[3];
  for (int j = 0; j < 3; j++) {
    const int c = int(lane) + 32 * j;
    const bool live = c < total;
    lv[j] = live ? cv[c / a.topk][c % a.topk] : -INFINITY;
    li[j] = live ? ci[c / a.topk][c % a.topk] : 0x7fffffff;
    ls[j] = live ? cs[c / a.topk][c % a.topk] : 0.0f;
  }
  for (int k = 0; k < a.topk; k++) {
    float v = lv[0];
    int id = li[0];
    for (int j = 1; j < 3; j++)
      if (route_better(lv[j], li[j], v, id)) {
        v = lv[j];
        id = li[j];
      }
    if (all_ordinary) {
      id = tf_route_rank(v, id);
    } else {
    for (ushort o = 16; o > 0; o >>= 1) {
      const float ov = simd_shuffle_down(v, o);
      const int oi = simd_shuffle_down(id, o);
      if (route_better(ov, oi, v, id)) {
        v = ov;
        id = oi;
      }
    }
    }
    id = simd_broadcast(id, 0);
    float mine = 0.0f;
    bool has = false;
    for (int j = 0; j < 3; j++)
      if (li[j] == id) {
        mine = ls[j];
        has = true;
        lv[j] = -INFINITY;
        li[j] = 0x7fffffff;
      }
    s2[k] = id;
    w2[k] = simd_broadcast(mine, ushort(simd_min(has ? lane : 32u)));
  }
  }
  if (lane == 0) {
    float sum = 0.0f;
    for (int k = 0; k < a.topk; k++) sum += w2[k];
    for (int k = 0; k < a.topk; k++) w2[k] = w2[k] / (sum + 1e-20f) * a.scale;
    for (int i = 1; i < a.topk; i++)
      for (int j = i; j > 0 && s2[j - 1] > s2[j]; j--) {
        const int ts = s2[j];
        s2[j] = s2[j - 1];
        s2[j - 1] = ts;
        const float tw = w2[j];
        w2[j] = w2[j - 1];
        w2[j - 1] = tw;
      }
    for (int k = 0; k < a.topk; k++) {
      PICK[r * a.topk + k] = uint(s2[k]);
      if (a.trace_layer >= 0) TRACE[(a.trace_layer * 6 + r) * a.topk + k] = uint(s2[k]);
      WEIGHT[r * a.topk + k] = w2[k];
    }
  }
}


template <int ROWS, int COLS>
[[kernel]] void tf_router_project(
    const device bfloat4* X [[buffer(0)]], const device bfloat4* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
  constexpr int K = 5120, N = 384;
  const int j0 = (int(tg) * int(nsg) + int(sg)) * COLS;
  float acc[ROWS][COLS];
  for (int r = 0; r < ROWS; ++r)
    for (int c = 0; c < COLS; ++c)
      acc[r][c] = 0.0f;

  for (int k = int(lane) * 2; k < K / 4; k += 64) {
    float w[COLS][8];
    for (int c = 0; c < COLS; ++c) {
      const device bfloat4* p = W + (j0 + c) * (K / 4) + k;
      const bfloat4 lo = p[0], hi = p[1];
      for (int i = 0; i < 4; ++i) {
        w[c][i] = float(lo[i]);
        w[c][i + 4] = float(hi[i]);
      }
    }
    for (int r = 0; r < ROWS; ++r) {
      const device bfloat4* p = X + r * (K / 4) + k;
      const bfloat4 lo = p[0], hi = p[1];
      float x[8];
      for (int i = 0; i < 4; ++i) {
        x[i] = float(lo[i]);
        x[i + 4] = float(hi[i]);
      }
      for (int c = 0; c < COLS; ++c) {
        float part = 0.0f;
        for (int i = 0; i < 8; ++i) part += x[i] * w[c][i];
        acc[r][c] += part;
      }
    }
  }
  for (int r = 0; r < ROWS; ++r)
    for (int c = 0; c < COLS; ++c) {
      const float value = simd_sum(acc[r][c]);
      if (lane == 0) Y[r * N + j0 + c] = value;
    }
}

#define TF_ROUTER(R, C) template [[host_name("tf_router_project_" #R "_" #C)]] [[kernel]] \
  decltype(tf_router_project<R, C>) tf_router_project<R, C>;
#define TF_ROUTER_COLS(R) TF_ROUTER(R, 1) TF_ROUTER(R, 2) TF_ROUTER(R, 4)
TF_ROUTER_COLS(1) TF_ROUTER_COLS(2) TF_ROUTER_COLS(3)
TF_ROUTER_COLS(4) TF_ROUTER_COLS(5) TF_ROUTER_COLS(6)

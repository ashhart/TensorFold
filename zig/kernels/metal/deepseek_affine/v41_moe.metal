// DeepSeek-V4.1-Flash's MoE, engram and image-tower kernels, compiled after v41_attend.metal.

// ---------------------------------------------------------------- MoE

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
[[kernel]] void ds_route(const device float* LOGITS [[buffer(0)]], const device float* BIAS [[buffer(1)]],
                         device uint* PICK [[buffer(2)]], device float* WEIGHT [[buffer(3)]],
                         constant RouteArgs& a [[buffer(4)]], const device int* IMG [[buffer(5)]],
                         const device float* BIAS_VL [[buffer(6)]], device uint* TRACE [[buffer(7)]], uint r [[threadgroup_position_in_grid]],
                         uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
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
  for (int k = 0; k < a.topk; k++) {
    float v = key;
    int id = myid;
    for (ushort o = 16; o > 0; o >>= 1) {
      const float ov = simd_shuffle_down(v, o);
      const int oi = simd_shuffle_down(id, o);
      if (route_better(ov, oi, v, id)) {
        v = ov;
        id = oi;
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
    for (ushort o = 16; o > 0; o >>= 1) {
      const float ov = simd_shuffle_down(v, o);
      const int oi = simd_shuffle_down(id, o);
      if (route_better(ov, oi, v, id)) {
        v = ov;
        id = oi;
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

struct GluArgs {
  int width;    // the intermediate width
  int weighted; // multiply by WEIGHT[slot] (routed experts) or not (the shared expert)
  float limit;
};

// silu(min(gate, limit)) * clamp(up, +-limit) [* weight], bf16 out.
[[kernel]] void ds_swiglu(const device bfloat* G [[buffer(0)]], const device bfloat* U [[buffer(1)]],
                          const device float* WEIGHT [[buffer(2)]], device bfloat* OUT [[buffer(3)]],
                          constant GluArgs& a [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  const size_t i = size_t(g.y) * a.width + g.x;
  const float gate = min(float(G[i]), a.limit);
  const float up = clamp(float(U[i]), -a.limit, a.limit);
  float v = gate / (1.0f + precise::exp(-gate)) * up;
  if (a.weighted) v = WEIGHT[g.y] * v;
  OUT[i] = bfloat(v);
}

struct SortArgs {
  int slots;
  int experts;
};

// A chunk's routed slots sorted by expert (stable) for core/affine_mm's gathers, SORT_BLOCK slots a threadgroup.
constant int SORT_BLOCK = 256;

// Each block's count of each expert's slots.

[[kernel]] void ds_sort_count(const device uint* PICK [[buffer(0)]], device int* COUNTS [[buffer(1)]],
                              constant SortArgs& a [[buffer(2)]], uint t [[thread_position_in_threadgroup]],
                              uint b [[threadgroup_position_in_grid]]) {
  threadgroup atomic_int n[1024];
  for (int e = int(t); e < a.experts; e += SORT_BLOCK) atomic_store_explicit(&n[e], 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int i = int(b) * SORT_BLOCK + int(t);
  if (i < a.slots) atomic_fetch_add_explicit(&n[PICK[i]], 1, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = int(t); e < a.experts; e += SORT_BLOCK)
    COUNTS[size_t(b) * a.experts + e] = atomic_load_explicit(&n[e], memory_order_relaxed);
}

// One threadgroup, a thread an expert: COUNTS becomes each block's first place for it, OFFS each expert's first.
[[kernel]] void ds_sort_scan(device int* COUNTS [[buffer(0)]], device int* OFFS [[buffer(1)]],
                             constant SortArgs& a [[buffer(2)]], uint t [[thread_position_in_threadgroup]]) {
  threadgroup int base[1024];
  const int blocks = (a.slots + SORT_BLOCK - 1) / SORT_BLOCK;
  const bool mine = int(t) < a.experts;
  int total = 0;
  if (mine)
    for (int b = 0; b < blocks; b++) total += COUNTS[size_t(b) * a.experts + t];
  base[t] = total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    int at = 0;
    for (int e = 0; e < a.experts; e++) {
      const int n = base[e];
      base[e] = at;
      at += n;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!mine) return;
  int at = base[t];
  OFFS[t] = at;
  for (int b = 0; b < blocks; b++) {
    const int n = COUNTS[size_t(b) * a.experts + t];
    COUNTS[size_t(b) * a.experts + t] = at;
    at += n;
  }
}

// Each slot's place: its block's first for its expert plus its rank among the block's earlier slots of that expert.
[[kernel]] void ds_sort_place(const device uint* PICK [[buffer(0)]], const device float* WEIGHT [[buffer(1)]],
                              const device int* COUNTS [[buffer(2)]], device int* ORDER [[buffer(3)]],
                              device float* WS [[buffer(4)]], constant SortArgs& a [[buffer(5)]],
                              uint t [[thread_position_in_threadgroup]], uint b [[threadgroup_position_in_grid]]) {
  threadgroup uint picks[SORT_BLOCK];
  const int i = int(b) * SORT_BLOCK + int(t);
  const uint e = i < a.slots ? PICK[i] : 0xffffffffu;
  picks[t] = e;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (i >= a.slots) return;
  int rank = 0;
  for (uint k = 0; k < t; k++) rank += picks[k] == e;
  const int at = COUNTS[size_t(b) * a.experts + e] + rank;
  ORDER[at] = i;
  WS[at] = WEIGHT[i];
}

// The mean of a row's four streams (bf16) into Y at column `col` of `ld`-wide rows: a DSpark target layer's input.
[[kernel]] void ds_stream_mean(const device bfloat* X [[buffer(0)]], device bfloat* Y [[buffer(1)]],
                               constant int2& col_ld [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  const device bfloat* x = X + size_t(g.y) * F;
  const float v = (float(x[g.x]) + float(x[D + g.x]) + float(x[2 * D + g.x]) + float(x[3 * D + g.x])) / 4.0f;
  Y[size_t(g.y) * col_ld.y + col_ld.x + g.x] = bfloat(v);
}

struct MarkovArgs {
  int vocab;
  int rank;
  int step;   // which draft row's logits, and IDS[step] the token whose Markov embedding biases them
  uint exclude;
};

// A draft step's logits plus the Markov bias (embed[tok] . head[v]); each threadgroup's best (value, id) to PART.
[[kernel]] void ds_markov(const device bfloat* LOGITS [[buffer(0)]], const device bfloat* EMB [[buffer(1)]],
                          const device bfloat* HEAD [[buffer(2)]], const device uint* IDS [[buffer(3)]],
                          device float2* PART [[buffer(4)]], constant MarkovArgs& a [[buffer(5)]],
                          uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float e[256];
  threadgroup float bv[8];
  threadgroup int bi[8];
  const uint tok = IDS[a.step];
  for (int j = int(t); j < a.rank; j += 256) e[j] = float(EMB[size_t(tok) * a.rank + j]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int v = int(tg) * 256 + int(t);
  float val = -INFINITY;
  int id = 0x7fffffff;
  if (v < a.vocab && uint(v) != a.exclude) {
    const device bfloat* h = HEAD + size_t(v) * a.rank;
    float bias = 0.0f;
    for (int j = 0; j < a.rank; j++) bias += e[j] * float(h[j]);
    val = float(LOGITS[size_t(a.step) * a.vocab + v]) + bias;
    id = v;
  }
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ov = simd_shuffle_down(val, o);
    const int oi = simd_shuffle_down(id, o);
    if (ov > val || (ov == val && oi < id)) {
      val = ov;
      id = oi;
    }
  }
  if (lane == 0) {
    bv[sg] = val;
    bi[sg] = id;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float m = bv[0];
    int mi = bi[0];
    for (int k = 1; k < 8; k++)
      if (bv[k] > m || (bv[k] == m && bi[k] < mi)) {
        m = bv[k];
        mi = bi[k];
      }
    PART[tg] = float2(m, as_type<float>(mi));
  }
}

// As ds_markov, with the BF16 vocabulary head packed in 32-token strips.
[[kernel]] void ds_markov_packed(const device bfloat* LOGITS [[buffer(0)]], const device bfloat* EMB [[buffer(1)]],
                          const device bfloat* HEAD [[buffer(2)]], const device uint* IDS [[buffer(3)]],
                          device float2* PART [[buffer(4)]], constant MarkovArgs& a [[buffer(5)]],
                          uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float e[256];
  threadgroup float bv[8];
  threadgroup int bi[8];
  const uint tok = IDS[a.step];
  for (int j = int(t); j < a.rank; j += 256) e[j] = float(EMB[size_t(tok) * a.rank + j]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int v = int(tg) * 256 + int(t);
  float val = -INFINITY;
  int id = 0x7fffffff;
  if (v < a.vocab && uint(v) != a.exclude) {
    const device bfloat* h = HEAD + (size_t(v / 32) * a.rank * 32) + lane;
    float bias = 0.0f;
    for (int j = 0; j < a.rank; j++) bias += e[j] * float(h[j * 32]);
    val = float(LOGITS[size_t(a.step) * a.vocab + v]) + bias;
    id = v;
  }
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ov = simd_shuffle_down(val, o);
    const int oi = simd_shuffle_down(id, o);
    if (ov > val || (ov == val && oi < id)) {
      val = ov;
      id = oi;
    }
  }
  if (lane == 0) {
    bv[sg] = val;
    bi[sg] = id;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float m = bv[0];
    int mi = bi[0];
    for (int k = 1; k < 8; k++)
      if (bv[k] > m || (bv[k] == m && bi[k] < mi)) {
        m = bv[k];
        mi = bi[k];
      }
    PART[tg] = float2(m, as_type<float>(mi));
  }
}

[[kernel]] void ds_markov_audit(const device uint2* reference [[buffer(0)]],
                               const device uint2* packed_parts [[buffer(1)]],
                               device uint* mismatch [[buffer(2)]],
                               constant uint& groups [[buffer(3)]]) {
  for (uint group = 0; group < groups; ++group) {
    if (any(reference[group] != packed_parts[group])) {
      mismatch[0] = 1;
      return;
    }
  }
}

// Phase two: the step's token, the best of the threadgroups' bests, into IDS[step + 1].
[[kernel]] void ds_markov_pick(const device float2* PART [[buffer(0)]], device uint* IDS [[buffer(1)]],
                               constant int2& n_step [[buffer(2)]], uint t [[thread_position_in_grid]]) {
  if (t != 0) return;
  float m = -INFINITY;
  int mi = 0x7fffffff;
  for (int k = 0; k < n_step.x; k++) {
    const float v = PART[k].x;
    const int i = as_type<int>(PART[k].y);
    if (v > m || (v == m && i < mi)) {
      m = v;
      mi = i;
    }
  }
  IDS[n_step.y + 1] = uint(mi);
}

// ---------------------------------------------------------------- engram

struct EngramArgs {
  float eps;
};

// A threadgroup (256) a (row, copy): gate = sigmoid(signed sqrt(normalized <h, w * key>)), h += gate * value.
[[kernel]] void ds_engram(device bfloat* X [[buffer(0)]], const device bfloat* KV [[buffer(1)]],
                          const device bfloat* QW [[buffer(2)]], const device bfloat* KW [[buffer(3)]],
                          constant EngramArgs& a [[buffer(4)]], const device int* IMG [[buffer(5)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint2 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint t = tpos.x;
  const int c = int(tg.x), r = int(tg.y);
  if (IMG[r]) return;  // an image span's rows take no lookup (the reference shuts their gate)
  threadgroup float red[3][8];
  threadgroup float gate_s;
  device bfloat* x = X + (size_t(r) * S + c) * D;
  const device bfloat* key = KV + size_t(r) * (S + 1) * D + size_t(c) * D;
  const device bfloat* val = KV + size_t(r) * (S + 1) * D + size_t(S) * D;
  float hh = 0.0f, kk = 0.0f, hk = 0.0f;
  for (int d = int(t); d < D; d += 256) {
    const float h = float(x[d]), k = float(key[d]);
    const float w = float(QW[c * D + d]) * float(KW[c * D + d]);
    hh += h * h;
    kk += k * k;
    hk += h * w * k;
  }
  hh = simd_sum(hh);
  kk = simd_sum(kk);
  hk = simd_sum(hk);
  if (lane == 0) {
    red[0][sg] = hh;
    red[1][sg] = kk;
    red[2][sg] = hk;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float s0 = 0, s1 = 0, s2 = 0;
    for (int i = 0; i < 8; i++) {
      s0 += red[0][i];
      s1 += red[1][i];
      s2 += red[2][i];
    }
    const float rstd = precise::rsqrt(s0 / float(D) + a.eps) * precise::rsqrt(s1 / float(D) + a.eps);
    const float dot = s2 * rstd * precise::rsqrt(float(D));
    const float sq = precise::sqrt(max(fabs(dot), 1e-6f));
    gate_s = 1.0f / (1.0f + precise::exp(-copysign(sq, dot)));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int d = int(t); d < D; d += 256) x[d] = bfloat(float(x[d]) + gate_s * float(val[d]));
}

// ---------------------------------------------------------------- the vision tower (vision.py) and aligner

constant constexpr int VD = 1024;  // the tower's width
constant constexpr int VH = 16;    // its heads
constant constexpr int VHD = 64;   // a head's width

struct LinArgs {
  int rows;
  int n;
  int k;
};

// y[r, j] = bf16(sum_k x[r, k] W[j, k] + b[j]), any k (the 588-wide patch embedding): a column a simdgroup.
[[kernel]] void ds_linear_bias(const device bfloat* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
                               const device bfloat* B [[buffer(2)]], device bfloat* Y [[buffer(3)]],
                               constant LinArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const int j = int(tg.x) * 8 + int(sg);
  if (j >= a.n) return;
  const int r0 = int(tg.y) * 8, nr = min(8, a.rows - r0);
  const device bfloat* w = W + size_t(j) * a.k;
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int kk = int(lane); kk < a.k; kk += 32) {
    const float wv = float(w[kk]);
    for (int i = 0; i < nr; i++) acc[i] += float(X[size_t(r0 + i) * a.k + kk]) * wv;
  }
  for (int i = 0; i < nr; i++) {
    const float s = simd_sum(acc[i]);
    if (lane == 0) Y[size_t(r0 + i) * a.n + j] = bfloat(s + float(B[j]));
  }
}

// erf (Metal has none): Numerical Recipes' erfc, fractional error under 1.2e-7, far inside a bf16 step.
inline float ds_erf(float x) {
  const float z = fabs(x);
  const float t = 1.0f / (1.0f + 0.5f * z);
  const float r = t * precise::exp(-z * z - 1.26551223f +
                                   t * (1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f +
                                   t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f)))))))));
  return x >= 0.0f ? 1.0f - r : r - 1.0f;
}

// y = bf16(y + b) by column, and with `gelu` then GELU (erf form) on the sum: a Linear's bias, the aligner's w1.
[[kernel]] void ds_bias(device bfloat* Y [[buffer(0)]], const device bfloat* B [[buffer(1)]],
                        constant int2& n_gelu [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  const size_t i = size_t(g.y) * n_gelu.x + g.x;
  float v = float(bfloat(float(Y[i]) + float(B[g.x])));
  if (n_gelu.y) v = 0.5f * v * (1.0f + ds_erf(v * 0.70710678118654752f));
  Y[i] = bfloat(v);
}

// x = bf16(x + y): a residual branch.
[[kernel]] void ds_add(device bfloat* X [[buffer(0)]], const device bfloat* Y [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  X[i] = bfloat(float(X[i]) + float(Y[i]));
}

// The tower's 2D RoPE on qkv's q and k: a patch's row then column frequencies, half a head against the other.
[[kernel]] void ds_vit_rope(device bfloat* QKV [[buffer(0)]], const device float* FREQ [[buffer(1)]],
                            constant int& n_w [[buffer(2)]], uint3 g [[thread_position_in_grid]]) {
  const int i = int(g.x), h = int(g.y), r = int(g.z);  // i < 32, h < 32 (q's 16 heads, then k's)
  const int f = i < 16 ? (r / n_w) : (r % n_w);
  const float ang = float(f) * FREQ[i % 16];
  const float cs = precise::cos(ang), sn = precise::sin(ang);
  device bfloat* x = QKV + size_t(r) * 3 * VD + size_t(h) * VHD;
  const float x1 = float(x[i]), x2 = float(x[i + 32]);
  x[i] = bfloat(x1 * cs - x2 * sn);
  x[i + 32] = bfloat(x2 * cs + x1 * sn);
}

// The tower's bidirectional attention: a simdgroup a (patch, head), 32 keys a step, online fp32 softmax.
[[kernel]] void ds_vit_attend(const device bfloat* QKV [[buffer(0)]], device bfloat* O [[buffer(1)]],
                              constant int& n [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float qs[8][VHD];
  const int h = int(tg.x), r = int(tg.y) * 8 + int(sg);
  const bool live = r < n;
  const int rr = live ? r : n - 1;
  const device bfloat* q = QKV + size_t(rr) * 3 * VD + h * VHD;
  qs[sg][lane] = float(q[lane]);
  qs[sg][lane + 32] = float(q[lane + 32]);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const float scale = 0.125f;  // 1 / sqrt(64)
  float m = -INFINITY, l = 0.0f, o0 = 0.0f, o1 = 0.0f;
  for (int k0 = 0; k0 < n; k0 += 32) {
    const int key = k0 + int(lane);
    float s = -INFINITY;
    if (key < n) {
      const device bfloat* kr = QKV + size_t(key) * 3 * VD + VD + h * VHD;
      float d = 0.0f;
      for (int i = 0; i < VHD; i++) d += qs[sg][i] * float(kr[i]);
      s = d * scale;
    }
    const float bm = simd_max(s);
    const float mn = max(m, bm);
    const float corr = precise::exp(m - mn);
    const float p = key < n ? precise::exp(s - mn) : 0.0f;
    l = l * corr + simd_sum(p);
    o0 *= corr;
    o1 *= corr;
    const int kn = min(32, n - k0);
    for (int j = 0; j < kn; j++) {
      const float pj = simd_shuffle(p, ushort(j));
      const device bfloat* v = QKV + size_t(k0 + j) * 3 * VD + 2 * VD + h * VHD;
      o0 += pj * float(v[lane]);
      o1 += pj * float(v[lane + 32]);
    }
    m = mn;
  }
  if (live) {
    O[size_t(r) * VD + h * VHD + lane] = bfloat(o0 / l);
    O[size_t(r) * VD + h * VHD + lane + 32] = bfloat(o1 / l);
  }
}

// The tower's MLP gate: [rows, 2 x 2816] -> bf16(bf16(silu(gate)) * up) [rows, 2816].
[[kernel]] void ds_vit_glu(const device bfloat* H [[buffer(0)]], device bfloat* Y [[buffer(1)]],
                           constant int& inter [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  const device bfloat* h = H + size_t(g.y) * 2 * inter;
  const float gt = float(h[g.x]);
  const float sl = float(bfloat(gt / (1.0f + precise::exp(-gt))));
  Y[size_t(g.y) * inter + g.x] = bfloat(sl * float(h[inter + g.x]));
}

struct UnfoldArgs {
  int n_h;
  int n_w;
  int out_w;  // ceil(n_w / 3)
};

// The aligner's zero-padded 3x3 unfold: token (oy, ox) row c x 9 + ky x 3 + kx = tower[(3oy+ky) n_w + 3ox+kx][c].
[[kernel]] void ds_unfold(const device bfloat* T [[buffer(0)]], device bfloat* Y [[buffer(1)]],
                          constant UnfoldArgs& a [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  const int f = int(g.x), t = int(g.y);
  const int c = f / 9, ky = (f % 9) / 3, kx = f % 3;
  const int y = 3 * (t / a.out_w) + ky, x = 3 * (t % a.out_w) + kx;
  Y[size_t(t) * 9 * VD + f] = (y < a.n_h && x < a.n_w) ? T[size_t(y * a.n_w + x) * VD + c] : bfloat(0);
}

// Image spans into every stream: CODE >= 0 an aligner row, -2 / -3 / -4 start, end and line break, -1 text.
[[kernel]] void ds_put_image(device bfloat* X [[buffer(0)]], const device int* CODE [[buffer(1)]],
                             const device bfloat* ROWS [[buffer(2)]], const device bfloat* START [[buffer(3)]],
                             const device bfloat* END [[buffer(4)]], const device bfloat* NL [[buffer(5)]],
                             uint2 g [[thread_position_in_grid]]) {
  const int code = CODE[g.y];
  if (code == -1) return;
  const bfloat v = code >= 0 ? ROWS[size_t(code) * D + g.x] : (code == -2 ? START[g.x] : (code == -3 ? END[g.x] : NL[g.x]));
  for (int s = 0; s < S; s++) X[(size_t(g.y) * S + s) * D + g.x] = v;
}

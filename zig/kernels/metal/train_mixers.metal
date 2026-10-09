// Gradients through Nemotron's Mamba-2 and attention mixers for one sequence from position 0, in f32.
#include <metal_stdlib>
using namespace metal;

// A Mamba layer's shape: rows, heads, head dim, groups, state, inner width, conv width; ck the steps a checkpoint.
struct Shape {
  uint rows, heads, dh, groups, n, inner, conv, proj;
};

constant uint ck = 16;

inline float silu(float v) { return v / (1.0f + exp(-v)); }

inline float silu_grad(float v) {
  const float s = 1.0f / (1.0f + exp(-v));
  return s * (1.0f + v * (1.0f - s));
}

inline float softplus(float v) { return v > 20.0f ? v : log(1.0f + exp(v)); }

// Sum over the 16 lanes holding one state row (lanes 0..15 or 16..31 of a simdgroup).
inline float row_sum(float v) {
  v += simd_shuffle_xor(v, 8);
  v += simd_shuffle_xor(v, 4);
  v += simd_shuffle_xor(v, 2);
  v += simd_shuffle_xor(v, 1);
  return v;
}

// Sum over all 64 state rows for each of N state columns: lanes l, l+16 first, then the 32 simdgroups in turn.
inline void column_sums(thread const float (&v)[8], threadgroup float* scratch, uint lane, uint sg, uint t,
                        uint n, device float* out) {
  float w[8];
  for (uint i = 0; i < 8; i++) w[i] = v[i] + simd_shuffle_down(v[i], 16);
  if (lane < 16)
    for (uint i = 0; i < 8; i++) scratch[sg * n + lane * 8 + i] = w[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < n) {
    float s = 0;
    for (uint g = 0; g < 32; g++) s += scratch[g * n + t];
    out[t] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline float all_sum(float v, threadgroup float* scratch, uint lane, uint sg, uint groups) {
  v = simd_sum(v);
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  v = simd_sum(lane < groups ? scratch[lane] : 0.0f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return v;
}

// dt[t, h] = softplus(raw + bias) from the in-projection's last columns; grid (heads, rows).
kernel void tf_train_dt(const device bfloat* proj [[buffer(0)]],
                        const device bfloat* bias [[buffer(1)]],
                        device float* dt [[buffer(2)]],
                        constant Shape& s [[buffer(3)]],
                        uint2 pos [[thread_position_in_grid]]) {
  const uint h = pos.x, t = pos.y;
  if (h >= s.heads || t >= s.rows) return;
  dt[t * s.heads + h] = softplus(float(proj[size_t(t) * s.proj + s.inner + s.conv + h]) + float(bias[h]));
}

// The scan for one head, a threadgroup of 1024 (64 rows x 16 lanes of 8 columns): y with the D skip, checkpoints.
kernel void tf_train_ssm_fwd(const device bfloat* act [[buffer(0)]],
                             const device float* dt [[buffer(1)]],
                             const device float* a_neg [[buffer(2)]],
                             const device bfloat* d_skip [[buffer(3)]],
                             device float* y [[buffer(4)]],
                             device float* ckpt [[buffer(5)]],
                             constant Shape& s [[buffer(6)]],
                             uint h [[threadgroup_position_in_grid]],
                             uint t [[thread_position_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  const uint p = t / 16, n0 = (t % 16) * 8, g = h / (s.heads / s.groups), size = s.dh * s.n;
  const uint b0 = s.inner + g * s.n, c0 = s.inner + s.groups * s.n + g * s.n;
  float st[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  device float* mine = ckpt + size_t(h) * ((s.rows + ck - 1) / ck + 1) * size;
  for (uint i = 0; i < 8; i++) mine[p * s.n + n0 + i] = 0;
  for (uint r = 0; r < s.rows; r++) {
    const device bfloat* row = act + size_t(r) * s.conv;
    const float d = dt[r * s.heads + h], a = exp(d * a_neg[h]), x = float(row[h * s.dh + p]);
    float acc = 0;
    for (uint i = 0; i < 8; i++) {
      st[i] = a * st[i] + d * x * float(row[b0 + n0 + i]);
      acc += st[i] * float(row[c0 + n0 + i]);
    }
    acc = row_sum(acc);
    if (lane % 16 == 0) y[size_t(r) * s.inner + h * s.dh + p] = acc + float(d_skip[h]) * x;
    if ((r + 1) % ck == 0)
      for (uint i = 0; i < 8; i++) mine[((r + 1) / ck) * size + p * s.n + n0 + i] = st[i];
  }
}

// The scan backward for one head: dx (with the D skip) into dact, per-head dB and dC partials, and ddt.
kernel void tf_train_ssm_back(const device bfloat* act [[buffer(0)]],
                              const device float* dt [[buffer(1)]],
                              const device float* a_neg [[buffer(2)]],
                              const device bfloat* d_skip [[buffer(3)]],
                              const device float* dy [[buffer(4)]],
                              const device float* ckpt [[buffer(5)]],
                              device float* states [[buffer(6)]],
                              device float* dact [[buffer(7)]],
                              device float* db_part [[buffer(8)]],
                              device float* dc_part [[buffer(9)]],
                              device float* ddt [[buffer(10)]],
                              constant Shape& s [[buffer(11)]],
                              uint h [[threadgroup_position_in_grid]],
                              uint t [[thread_position_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[32 * 128];
  const uint p = t / 16, n0 = (t % 16) * 8, g = h / (s.heads / s.groups), size = s.dh * s.n;
  const uint b0 = s.inner + g * s.n, c0 = s.inner + s.groups * s.n + g * s.n;
  const device float* mine = ckpt + size_t(h) * ((s.rows + ck - 1) / ck + 1) * size;
  device float* chunk = states + size_t(h) * ck * size;
  float ds[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int c = int((s.rows - 1) / ck); c >= 0; c--) {
    const uint first = uint(c) * ck, last = min(first + ck, s.rows);
    float st[8];
    for (uint i = 0; i < 8; i++) st[i] = mine[uint(c) * size + p * s.n + n0 + i];
    for (uint r = first; r < last; r++) {
      const device bfloat* row = act + size_t(r) * s.conv;
      const float d = dt[r * s.heads + h], a = exp(d * a_neg[h]), x = float(row[h * s.dh + p]);
      for (uint i = 0; i < 8; i++) {
        st[i] = a * st[i] + d * x * float(row[b0 + n0 + i]);
        chunk[(r - first) * size + p * s.n + n0 + i] = st[i];
      }
    }
    for (int r = int(last) - 1; r >= int(first); r--) {
      const device bfloat* row = act + size_t(r) * s.conv;
      const float d = dt[r * s.heads + h], a = exp(d * a_neg[h]), x = float(row[h * s.dh + p]);
      const float gy = dy[size_t(r) * s.inner + h * s.dh + p];
      float cur[8], prev[8], v[8];
      for (uint i = 0; i < 8; i++) {
        cur[i] = chunk[(uint(r) - first) * size + p * s.n + n0 + i];
        prev[i] = r > int(first) ? chunk[(uint(r) - first - 1) * size + p * s.n + n0 + i]
                                 : mine[uint(c) * size + p * s.n + n0 + i];
        ds[i] += gy * float(row[c0 + n0 + i]);
        v[i] = gy * cur[i];
      }
      column_sums(v, scratch, lane, sg, t, s.n, dc_part + (size_t(r) * s.heads + h) * s.n);
      float dx = 0, dd = 0;
      for (uint i = 0; i < 8; i++) {
        const float bn = float(row[b0 + n0 + i]);
        dx += ds[i] * bn;
        dd += ds[i] * (x * bn + a_neg[h] * a * prev[i]);
        v[i] = d * ds[i] * x;
      }
      column_sums(v, scratch, lane, sg, t, s.n, db_part + (size_t(r) * s.heads + h) * s.n);
      dx = row_sum(dx);
      if (lane % 16 == 0) dact[size_t(r) * s.conv + h * s.dh + p] = d * dx + float(d_skip[h]) * gy;
      dd = all_sum(dd, scratch, lane, sg, 32);
      if (t == 0) ddt[r * s.heads + h] = dd;
      for (uint i = 0; i < 8; i++) ds[i] *= a;
    }
  }
}

// dact's B and C columns: each group's per-head partials summed in head order; grid (n, groups, rows).
kernel void tf_train_ssm_bc(const device float* db_part [[buffer(0)]],
                            const device float* dc_part [[buffer(1)]],
                            device float* dact [[buffer(2)]],
                            constant Shape& s [[buffer(3)]],
                            uint3 pos [[thread_position_in_grid]]) {
  const uint n = pos.x, g = pos.y, r = pos.z, per = s.heads / s.groups;
  if (n >= s.n || g >= s.groups || r >= s.rows) return;
  float b = 0, c = 0;
  for (uint h = g * per; h < (g + 1) * per; h++) {
    b += db_part[(size_t(r) * s.heads + h) * s.n + n];
    c += dc_part[(size_t(r) * s.heads + h) * s.n + n];
  }
  dact[size_t(r) * s.conv + s.inner + g * s.n + n] = b;
  dact[size_t(r) * s.conv + s.inner + s.groups * s.n + g * s.n + n] = c;
}

// The gate and grouped RMS norm undone: dy (f32) into the SSM's output, dz into the projection's z columns.
kernel void tf_train_gate_back(const device float* y [[buffer(0)]],
                               const device bfloat* proj [[buffer(1)]],
                               const device bfloat* w [[buffer(2)]],
                               const device float* dn [[buffer(3)]],
                               device float* dy [[buffer(4)]],
                               device float* dproj [[buffer(5)]],
                               constant Shape& s [[buffer(6)]],
                               constant float& eps [[buffer(7)]],
                               uint2 tg [[threadgroup_position_in_grid]],
                               uint2 tid [[thread_position_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  const uint t = tid.x, g = tg.x, r = tg.y, width = s.inner / s.groups, j0 = g * width;
  float sq = 0, dot = 0;
  for (uint j = j0 + t; j < j0 + width; j += 256) {
    const float gated = y[size_t(r) * s.inner + j] * silu(float(proj[size_t(r) * s.proj + j]));
    sq += gated * gated;
    dot += float(w[j]) * dn[size_t(r) * s.inner + j] * gated;
  }
  sq = all_sum(sq, scratch, lane, sg, 8);
  dot = all_sum(dot, scratch, lane, sg, 8);
  const float inv = rsqrt(sq / float(width) + eps), k = inv * inv * inv * dot / float(width);
  for (uint j = j0 + t; j < j0 + width; j += 256) {
    const float z = float(proj[size_t(r) * s.proj + j]), yv = y[size_t(r) * s.inner + j];
    const float gated = yv * silu(z), dg = inv * float(w[j]) * dn[size_t(r) * s.inner + j] - k * gated;
    dy[size_t(r) * s.inner + j] = dg * silu(z);
    dproj[size_t(r) * s.proj + j] = dg * yv * silu_grad(z);
  }
}

// The depthwise causal conv and its SiLU undone: dproj's xBC columns; conv_out is the conv before its bias.
kernel void tf_train_conv_back(const device float* dact [[buffer(0)]],
                               const device bfloat* conv_out [[buffer(1)]],
                               const device bfloat* weight [[buffer(2)]],
                               const device bfloat* bias [[buffer(3)]],
                               device float* dproj [[buffer(4)]],
                               constant Shape& s [[buffer(5)]],
                               constant uint& taps [[buffer(6)]],
                               uint2 pos [[thread_position_in_grid]]) {
  const uint c = pos.x, r = pos.y;
  if (c >= s.conv || r >= s.rows) return;
  float acc = 0;
  for (uint j = 0; j < taps; j++) {
    const uint at = r + taps - 1 - j;
    if (at >= s.rows) continue;
    const float pre = float(conv_out[size_t(at) * s.conv + c]) + float(bias[c]);
    acc += float(weight[c * taps + j]) * dact[size_t(at) * s.conv + c] * silu_grad(pre);
  }
  dproj[size_t(r) * s.proj + s.inner + c] = acc;
}

// d(raw dt) = ddt sigmoid(raw + bias) into dproj's last columns; grid (heads, rows).
kernel void tf_train_dt_back(const device bfloat* proj [[buffer(0)]],
                             const device bfloat* bias [[buffer(1)]],
                             const device float* ddt [[buffer(2)]],
                             device float* dproj [[buffer(3)]],
                             constant Shape& s [[buffer(4)]],
                             uint2 pos [[thread_position_in_grid]]) {
  const uint h = pos.x, r = pos.y;
  if (h >= s.heads || r >= s.rows) return;
  const size_t at = size_t(r) * s.proj + s.inner + s.conv + h;
  const float v = float(proj[at]) + float(bias[h]);
  dproj[at] = ddt[r * s.heads + h] / (1.0f + exp(-v));
}

// Attention's shape: rows, heads, kv heads, head dim; scale 1/sqrt(head dim).
struct Heads {
  uint rows, heads, kv_heads, dim;
  float scale;
};

// One query row of one head: its probabilities and their gradient kept, and dq; 256 threads (one a key row).
kernel void tf_train_attn_q(const device bfloat* q [[buffer(0)]],
                            const device bfloat* k [[buffer(1)]],
                            const device bfloat* v [[buffer(2)]],
                            const device float* dout [[buffer(3)]],
                            device float* probs [[buffer(4)]],
                            device float* dscores [[buffer(5)]],
                            device float* dq [[buffer(6)]],
                            constant Heads& a [[buffer(7)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint2 tid [[thread_position_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float qi[256], gi[256], ds[256], scratch[32];
  const uint t = tid.x, h = tg.x, i = tg.y, kv = h / (a.heads / a.kv_heads), qw = a.heads * a.dim, kw = a.kv_heads * a.dim;
  for (uint d = t; d < a.dim; d += 256) {
    qi[d] = float(q[size_t(i) * qw + h * a.dim + d]);
    gi[d] = dout[size_t(i) * qw + h * a.dim + d];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool live = t <= i && t < a.rows;
  float score = -INFINITY, dp = 0;
  if (live) {
    float sc = 0, dv = 0;
    for (uint d = 0; d < a.dim; d++) {
      sc += qi[d] * float(k[size_t(t) * kw + kv * a.dim + d]);
      dv += gi[d] * float(v[size_t(t) * kw + kv * a.dim + d]);
    }
    score = sc * a.scale;
    dp = dv;
  }
  float top = simd_max(score);
  if (lane == 0) scratch[sg] = top;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  top = simd_max(lane < 8 ? scratch[lane] : -INFINITY);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float e = live ? exp(score - top) : 0.0f;
  const float total = all_sum(e, scratch, lane, sg, 8);
  const float pr = e / total;
  const float mix = all_sum(pr * dp, scratch, lane, sg, 8);
  const float g = pr * (dp - mix);
  ds[t] = g;
  if (t < a.rows) {
    probs[(size_t(h) * a.rows + i) * a.rows + t] = pr;
    dscores[(size_t(h) * a.rows + i) * a.rows + t] = g;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = t; d < a.dim; d += 256) {
    float acc = 0;
    for (uint j = 0; j <= i; j++) acc += ds[j] * float(k[size_t(j) * kw + kv * a.dim + d]);
    dq[size_t(i) * qw + h * a.dim + d] = acc * a.scale;
  }
}

// One key row of one kv head: dk and dv summed over the heads that read it and the query rows after it.
kernel void tf_train_attn_kv(const device bfloat* q [[buffer(0)]],
                             const device float* dout [[buffer(1)]],
                             const device float* probs [[buffer(2)]],
                             const device float* dscores [[buffer(3)]],
                             device float* dk [[buffer(4)]],
                             device float* dv [[buffer(5)]],
                             constant Heads& a [[buffer(6)]],
                             uint2 tg [[threadgroup_position_in_grid]],
                             uint2 tid [[thread_position_in_threadgroup]]) {
  const uint d = tid.x, kv = tg.x, j = tg.y, per = a.heads / a.kv_heads, qw = a.heads * a.dim, kw = a.kv_heads * a.dim;
  if (d >= a.dim) return;
  float gk = 0, gv = 0;
  for (uint h = kv * per; h < (kv + 1) * per; h++)
    for (uint i = j; i < a.rows; i++) {
      const size_t at = (size_t(h) * a.rows + i) * a.rows + j;
      gk += dscores[at] * float(q[size_t(i) * qw + h * a.dim + d]);
      gv += probs[at] * dout[size_t(i) * qw + h * a.dim + d];
    }
  dk[size_t(j) * kw + kv * a.dim + d] = gk * a.scale;
  dv[size_t(j) * kw + kv * a.dim + d] = gv;
}

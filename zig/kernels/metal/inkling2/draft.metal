// DSpark draft head (mtp.safetensors) kernels: each op the reference rounds to bf16 is rounded; sums in fp32.

struct Ik2dRows {
  uint rows;
  uint width;
  float eps;
  uint pitch;  // elements between input rows (ik2d_mix_rms: taps * width)
  uint taps;
};

// Block start: row 0 = embed_norm(embed[id[0]]), the other rows the raw mask embedding.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik2d_embed_block(
    const device uint* id [[buffer(0)]], const device bfloat* embed [[buffer(1)]], const device bfloat* embed_w [[buffer(2)]],
    const device bfloat* mask [[buffer(3)]], device bfloat* res [[buffer(4)]], constant Ik2dRows& p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint n [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  device bfloat* r = res + ulong(row) * p.width;
  if (row > 0) {
    for (uint i = t; i < p.width; i += n) r[i] = mask[i];
    return;
  }
  threadgroup float part[33];
  const device bfloat* e = embed + ulong(id[0]) * p.width;
  float s = 0.0f;
  for (uint i = t; i < p.width; i += n) {
    const float x = float(e[i]);
    s += x * x;
  }
  const float inv = metal::precise::rsqrt(tf_vis_sum(s, part, n, lane, sg) / float(p.width) + p.eps);
  for (uint i = t; i < p.width; i += n) r[i] = bfloat(float(embed_w[i]) * (float(e[i]) * inv));
}

// y = bf16(x * rsqrt(mean(x^2) + eps) * w), one threadgroup a row.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik2d_rms(
    const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], device bfloat* y [[buffer(2)]],
    constant Ik2dRows& p [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const device bfloat* xr = x + ulong(row) * p.width;
  float s = 0.0f;
  for (uint i = t; i < p.width; i += n) {
    const float v = float(xr[i]);
    s += v * v;
  }
  const float inv = metal::precise::rsqrt(tf_vis_sum(s, part, n, lane, sg) / float(p.width) + p.eps);
  device bfloat* yr = y + ulong(row) * p.width;
  for (uint i = t; i < p.width; i += n) yr[i] = bfloat(float(xr[i]) * inv * float(w[i]));
}

// The fused residual add: res = bf16(res + delta), and y = bf16(norm of the unrounded fp32 sum * w).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik2d_add_rms(
    device bfloat* res [[buffer(0)]], const device bfloat* delta [[buffer(1)]], const device bfloat* w [[buffer(2)]],
    device bfloat* y [[buffer(3)]], constant Ik2dRows& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  device bfloat* rr = res + ulong(row) * p.width;
  const device bfloat* dr = delta + ulong(row) * p.width;
  float s = 0.0f;
  for (uint i = t; i < p.width; i += n) {
    const float v = float(rr[i]) + float(dr[i]);
    s += v * v;
  }
  const float inv = metal::precise::rsqrt(tf_vis_sum(s, part, n, lane, sg) / float(p.width) + p.eps);
  device bfloat* yr = y + ulong(row) * p.width;
  for (uint i = t; i < p.width; i += n) {
    const float v = float(rr[i]) + float(dr[i]);
    rr[i] = bfloat(v);
    yr[i] = bfloat(v * inv * float(w[i]));
  }
}

// mix_rmsnorm: m = sum over taps of feats[row][t] * alpha[t] in fp32, then bf16(m * rsqrt(mean(m^2) + eps) * w).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik2d_mix_rms(
    const device bfloat* feats [[buffer(0)]], const device bfloat* alpha [[buffer(1)]], const device bfloat* w [[buffer(2)]],
    device bfloat* y [[buffer(3)]], constant Ik2dRows& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const device bfloat* fr = feats + ulong(row) * p.pitch;
  float s = 0.0f;
  for (uint i = t; i < p.width; i += n) {
    float m = 0.0f;
    for (uint k = 0; k < p.taps; k++) m += float(fr[ulong(k) * p.width + i]) * float(alpha[k]);
    s += m * m;
  }
  const float inv = metal::precise::rsqrt(tf_vis_sum(s, part, n, lane, sg) / float(p.width) + p.eps);
  device bfloat* yr = y + ulong(row) * p.width;
  for (uint i = t; i < p.width; i += n) {
    float m = 0.0f;
    for (uint k = 0; k < p.taps; k++) m += float(fr[ulong(k) * p.width + i]) * float(alpha[k]);
    yr[i] = bfloat(m * inv * float(w[i]));
  }
}

// One 128-wide head vector: rms norm (bf16), then the neox rotation (pairs d, d + 64) in fp32, rounded again.
inline float ik2d_norm_rope(float x, float w, float eps, const device float* inv_freq, float pos,
                            threadgroup float* nv, threadgroup float* part, uint d, uint lane, uint sg) {
  const float inv = metal::precise::rsqrt(tf_vis_sum(x * x, part, 128, lane, sg) / 128.0f + eps);
  const float v = tf_vis_bf(x * inv * w);
  nv[d] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float ang = pos * inv_freq[d & 63u];
  const float c = metal::precise::cos(ang), s = metal::precise::sin(ang);
  const float partner = nv[d ^ 64u];
  const float o = d < 64 ? v * c - partner * s : v * c + partner * s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return tf_vis_bf(o);
}

struct Ik2dPrep {
  uint rows;
  uint pos0;
  uint heads_q;
  uint heads_kv;
  float eps;
};

// The block's q and k heads (raw projections [rows, heads * 128]) normed, rotated at positions pos0 + row, as bf16.
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void ik2d_qk_prep(
    const device bfloat* qraw [[buffer(0)]], const device bfloat* kraw [[buffer(1)]], const device bfloat* qw [[buffer(2)]],
    const device bfloat* kw [[buffer(3)]], const device float* inv_freq [[buffer(4)]], device bfloat* qout [[buffer(5)]],
    device bfloat* kout [[buffer(6)]], constant Ik2dPrep& p [[buffer(7)]], uint2 g [[threadgroup_position_in_grid]],
    uint d [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  threadgroup float nv[128];
  const bool is_q = g.x < p.heads_q;
  const uint h = is_q ? g.x : g.x - p.heads_q;
  const ulong width = ulong(is_q ? p.heads_q : p.heads_kv) * 128;
  const float x = float((is_q ? qraw : kraw)[ulong(g.y) * width + h * 128 + d]);
  const float w = float((is_q ? qw : kw)[d]);
  const float o = ik2d_norm_rope(x, w, p.eps, inv_freq, float(p.pos0 + g.y), nv, part, d, lane, sg);
  (is_q ? qout : kout)[ulong(g.y) * width + h * 128 + d] = bfloat(o);
}

struct Ik2dCtx {
  uint rows;
  uint pos0;  // absolute position of row 0
  uint ring;  // slots of the layer's cache: position p lives in slot p % ring
  uint groups;
  float eps;
};

// Context rows (committed positions) into one layer's cache: k normed (k_norm_ctx) and rotated, v as projected.
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void ik2d_ctx_store(
    const device bfloat* kraw [[buffer(0)]], const device bfloat* vraw [[buffer(1)]], const device bfloat* kw [[buffer(2)]],
    const device float* inv_freq [[buffer(3)]], device bfloat* kc [[buffer(4)]], device bfloat* vc [[buffer(5)]],
    constant Ik2dCtx& p [[buffer(6)]], uint2 g [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  threadgroup float nv[128];
  const ulong width = ulong(p.groups) * 128;
  const ulong src = ulong(g.y) * width + g.x * 128 + d;
  const ulong pos = ulong(p.pos0) + g.y;
  const ulong dst = (pos % p.ring) * width + g.x * 128 + d;
  const float o = ik2d_norm_rope(float(kraw[src]), float(kw[d]), p.eps, inv_freq, float(pos), nv, part, d, lane, sg);
  kc[dst] = bfloat(o);
  vc[dst] = vraw[src];
}

struct Ik2dAttn {
  uint rows;    // block rows (queries), at most 8
  uint heads;   // query heads
  uint groups;  // key/value heads
  uint lo;      // first context position the block reads
  uint nctx;    // context positions read: lo .. lo + nctx
  uint ring;    // slots of the context cache
  uint splits;
  uint per;     // keys a split reads
  float scale;
};

// Not-causal attention of the block over the context cache, then its own keys: a partial softmax per (kv head, split).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik2d_attn_split(
    const device bfloat* q [[buffer(0)]], const device bfloat* kc [[buffer(1)]], const device bfloat* vc [[buffer(2)]],
    const device bfloat* kb [[buffer(3)]], const device bfloat* vb [[buffer(4)]], device float* pm [[buffer(5)]],
    device float* pl [[buffer(6)]], device float* po [[buffer(7)]], constant Ik2dAttn& p [[buffer(8)]],
    uint2 g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint per_group = p.heads / p.groups;
  const uint h = g.x * per_group + sg;
  const ulong qw = ulong(p.heads) * 128, kvw = ulong(p.groups) * 128;
  float qv[8][4];
  float m[8], l[8], acc[8][4];
  for (uint r = 0; r < 8; r++) {
    for (uint j = 0; j < 4; j++) {
      qv[r][j] = r < p.rows ? float(q[ulong(r) * qw + h * 128 + lane + 32 * j]) : 0.0f;
      acc[r][j] = 0.0f;
    }
    m[r] = -INFINITY;
    l[r] = 0.0f;
  }
  const uint total = p.nctx + p.rows;
  const uint js = min(total, g.y * p.per), je = min(total, js + p.per);
  for (uint kk = js; kk < je; kk++) {
    const device bfloat* kp;
    const device bfloat* vp;
    if (kk < p.nctx) {
      const ulong slot = ulong(p.lo + kk) % p.ring;
      kp = kc + slot * kvw + g.x * 128;
      vp = vc + slot * kvw + g.x * 128;
    } else {
      const ulong rb = kk - p.nctx;
      kp = kb + rb * kvw + g.x * 128;
      vp = vb + rb * kvw + g.x * 128;
    }
    float kf[4], vf[4];
    for (uint j = 0; j < 4; j++) {
      kf[j] = float(kp[lane + 32 * j]);
      vf[j] = float(vp[lane + 32 * j]);
    }
    for (uint r = 0; r < 8; r++) {
      if (r >= p.rows) break;
      const float s = simd_sum(qv[r][0] * kf[0] + qv[r][1] * kf[1] + qv[r][2] * kf[2] + qv[r][3] * kf[3]) * p.scale;
      const float mn = max(m[r], s);
      const float a = metal::precise::exp(m[r] - mn), e = metal::precise::exp(s - mn);
      l[r] = l[r] * a + e;
      for (uint j = 0; j < 4; j++) acc[r][j] = acc[r][j] * a + e * vf[j];
      m[r] = mn;
    }
  }
  for (uint r = 0; r < 8; r++) {
    if (r >= p.rows) break;
    const ulong at = (ulong(g.y) * p.heads + h) * p.rows + r;
    if (lane == 0) {
      pm[at] = m[r];
      pl[at] = l[r];
    }
    for (uint j = 0; j < 4; j++) po[at * 128 + lane + 32 * j] = acc[r][j];
  }
}

// The splits' partial softmaxes merged: out [rows, heads * 128] = bf16(sum_s e^(m_s - M) o_s / sum_s e^(m_s - M) l_s).
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void ik2d_attn_merge(
    const device float* pm [[buffer(0)]], const device float* pl [[buffer(1)]], const device float* po [[buffer(2)]],
    device bfloat* out [[buffer(3)]], constant Ik2dAttn& p [[buffer(4)]], uint2 g [[threadgroup_position_in_grid]],
    uint d [[thread_index_in_threadgroup]]) {
  float top = -INFINITY;
  for (uint s = 0; s < p.splits; s++) top = max(top, pm[(ulong(s) * p.heads + g.x) * p.rows + g.y]);
  float z = 0.0f, o = 0.0f;
  for (uint s = 0; s < p.splits; s++) {
    const ulong at = (ulong(s) * p.heads + g.x) * p.rows + g.y;
    const float m = pm[at];
    if (m == -INFINITY) continue;
    const float w = metal::precise::exp(m - top);
    z += pl[at] * w;
    o += po[at * 128 + d] * w;
  }
  out[ulong(g.y) * p.heads * 128 + ulong(g.x) * 128 + d] = bfloat(o / z);
}

// act = bf16(silu(gate) * up) over [rows, width] bf16 projections (fp32 silu).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik2d_swiglu(
    const device bfloat* gate [[buffer(0)]], const device bfloat* up [[buffer(1)]], device bfloat* act [[buffer(2)]],
    constant uint& width [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * width + at.x;
  const float g = float(gate[i]);
  act[i] = bfloat(g / (1.0f + metal::precise::exp(-g)) * float(up[i]));
}

struct Ik2dRnn {
  uint width;
  uint rank;
  uint step;
};

// raw = bf16(rnn_proj [state | markov_w1[prev] | h | f_rnn] + bias), a threadgroup an output; seq[step] = prev token.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik2d_rnn(
    const device bfloat* state [[buffer(0)]], const device bfloat* w1 [[buffer(1)]], const device uint* seq [[buffer(2)]],
    const device bfloat* h [[buffer(3)]], const device bfloat* f [[buffer(4)]], const device bfloat* W [[buffer(5)]],
    const device bfloat* bias [[buffer(6)]], device bfloat* raw [[buffer(7)]], constant Ik2dRnn& p [[buffer(8)]],
    uint o [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint n [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const uint K = 2 * p.rank + 2 * p.width;
  const device bfloat* wr = W + ulong(o) * K;
  const ulong prev = seq[p.step];
  float s = 0.0f;
  for (uint k = t; k < K; k += n) {
    float z;
    if (k < p.rank) z = p.step == 0 ? 0.0f : float(state[k]);
    else if (k < 2 * p.rank) z = float(w1[prev * p.rank + (k - p.rank)]);
    else if (k < 2 * p.rank + p.width) z = float(h[k - 2 * p.rank]);
    else z = float(f[k - 2 * p.rank - p.width]);
    s += z * float(wr[k]);
  }
  s = tf_vis_sum(s, part, n, lane, sg);
  if (t == 0) raw[o] = bfloat(s + float(bias[o]));
}

// The gate cell on raw = [gate | cand | out]: new state in place, latent = tanh(out), every op rounded to bf16.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik2d_cell(
    const device bfloat* raw [[buffer(0)]], device bfloat* state [[buffer(1)]], device bfloat* latent [[buffer(2)]],
    constant Ik2dRnn& p [[buffer(3)]], uint j [[thread_position_in_grid]]) {
  if (j >= p.rank) return;
  const float gr = float(raw[j]), cr = float(raw[p.rank + j]), orw = float(raw[2 * p.rank + j]);
  const float gate = tf_vis_bf(1.0f / (1.0f + metal::precise::exp(-gr)));
  const float old = p.step == 0 ? 0.0f : float(state[j]);
  const float keep = tf_vis_bf(gate * old);
  const float cand = tf_vis_bf(metal::precise::tanh(cr));
  const float add = tf_vis_bf(tf_vis_bf(1.0f - gate) * cand);
  state[j] = bfloat(keep + add);
  latent[j] = bfloat(metal::precise::tanh(orw));
}

struct Ik2dStep {
  uint vocab;
  uint unpadded;
  uint rank;
  float cap;
  uint lanes;        // latents (at most 8), each scored against every fold_w row from one read of it
  uint base_stride;  // elements between a lane's base logits and the next lane's
  uint groups;       // vocabulary groups of 8: entries between a lane's bests and the next lane's
};

// Per lane: y = bf16(cap * tanh(bf16(base + latent . fold_w[v] + fold_b[v]))); each group of 8's winner lands in best.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik2d_step_logits(
    const device bfloat* base [[buffer(0)]], const device bfloat* latent [[buffer(1)]], const device bfloat* fold_w [[buffer(2)]],
    const device bfloat* fold_b [[buffer(3)]], device float* best [[buffer(4)]], device uint* best_id [[buffer(5)]],
    constant Ik2dStep& p [[buffer(6)]], uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float tv[8][8];
  threadgroup uint ti[8][8];
  const uint v = tg * 8 + sg;
  float s[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  if (v < p.vocab) {
    for (uint k = lane; k < p.rank; k += 32) {
      const float w = float(fold_w[ulong(v) * p.rank + k]);
      for (uint j = 0; j < 8; j++) {
        if (j < p.lanes) s[j] += float(latent[ulong(j) * p.rank + k]) * w;
      }
    }
  }
  for (uint j = 0; j < 8; j++) {
    if (j >= p.lanes) break;
    float y = -INFINITY;
    const float sum = simd_sum(s[j]);
    if (v < p.vocab) {
      const float bias = tf_vis_bf(float(fold_b[v]) + sum);
      const float z = tf_vis_bf(float(base[ulong(j) * p.base_stride + v]) + bias);
      y = v < p.unpadded ? tf_vis_bf(p.cap * metal::precise::tanh(z / p.cap)) : -INFINITY;
    }
    if (lane == 0) {
      tv[j][sg] = y;
      ti[j][sg] = v;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0 && lane < p.lanes) {
    float bv = tv[lane][0];
    uint bi = ti[lane][0];
    for (uint i = 1; i < 8; i++) {
      if (tv[lane][i] > bv) {
        bv = tv[lane][i];
        bi = ti[lane][i];
      }
    }
    best[ulong(lane) * p.groups + tg] = bv;
    best_id[ulong(lane) * p.groups + tg] = bi;
  }
}

// seq[step + 1] = the best token over the threadgroups' bests (the lowest id on a tie), one threadgroup of 1024.
[[max_total_threads_per_threadgroup(1024)]] [[kernel]] void ik2d_pick(
    const device float* best [[buffer(0)]], const device uint* best_id [[buffer(1)]], device uint* seq [[buffer(2)]],
    constant uint2& p [[buffer(3)]], uint t [[thread_position_in_threadgroup]]) {
  threadgroup float tv[1024];
  threadgroup uint ti[1024];
  float bv = -INFINITY;
  uint bi = 0xFFFFFFFFu;
  for (uint i = t; i < p.x; i += 1024) {
    const float v = best[i];
    const uint id = best_id[i];
    if (v > bv || (v == bv && id < bi)) {
      bv = v;
      bi = id;
    }
  }
  tv[t] = bv;
  ti[t] = bi;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = 512; s > 0; s >>= 1) {
    if (t < s) {
      const float v = tv[t + s];
      const uint id = ti[t + s];
      if (v > tv[t] || (v == tv[t] && id < ti[t])) {
        tv[t] = v;
        ti[t] = id;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (t == 0) seq[p.y + 1] = ti[0];
}

struct Ik2dFold {
  uint vocab;
  uint width;
};

// fold_b[v] = bf16(sum_k unembed[v][k] * hidden_out.bias[k]) in fp32, a simdgroup a row.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ik2d_fold_b(
    const device bfloat* unembed [[buffer(0)]], const device bfloat* b [[buffer(1)]], device bfloat* out [[buffer(2)]],
    constant Ik2dFold& p [[buffer(3)]], uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint v = tg * 8 + sg;
  if (v >= p.vocab) return;
  float s = 0.0f;
  for (uint k = lane; k < p.width; k += 32) s += float(unembed[ulong(v) * p.width + k]) * float(b[k]);
  s = simd_sum(s);
  if (lane == 0) out[v] = bfloat(s);
}

// alpha = bf16(softmax(fuse_weight row)), one thread a row of `taps` logits.
[[max_total_threads_per_threadgroup(64)]] [[kernel]] void ik2d_alpha(
    const device bfloat* fw [[buffer(0)]], device bfloat* alpha [[buffer(1)]], constant uint2& p [[buffer(2)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= p.x) return;
  const device bfloat* r = fw + ulong(row) * p.y;
  float top = -INFINITY;
  for (uint k = 0; k < p.y; k++) top = max(top, float(r[k]));
  float z = 0.0f;
  for (uint k = 0; k < p.y; k++) z += metal::precise::exp(float(r[k]) - top);
  for (uint k = 0; k < p.y; k++) alpha[ulong(row) * p.y + k] = bfloat(metal::precise::exp(float(r[k]) - top) / z);
}

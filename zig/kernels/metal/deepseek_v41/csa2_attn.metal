// DeepSeek V4.1 CSA2 attention over one shared latent row (key and value), as a decode launch or a prompt launch.

// Key order, whatever the launch: the window by absolute position ascending, then selected rows in index order.

// Five chunks: the window, then the selected rows in runs of 128 (a negative index is skipped).

// A chunk is an fp32 online softmax from zero state, one key at a time; chunks with keys join in order, then the sink.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

#ifndef TF_CTL
#define TF_CTL 0
#endif

#define TF_ATTN_D 512u
#define TF_ATTN_WIN 128u
#define TF_ATTN_SEL 512u
#define TF_ATTN_RUN 128u
#define TF_ATTN_NCH 5u
#define TF_ATTN_HS 1u
#define TF_ATTN_KU 2u
#define TF_ATTN_RR 2u
#define TF_ATTN_HG 4u
#define TF_ATTN_TILE 16u
#define TF_ATTN_NEG (-1e30f)

struct TfAttnArgs {
  uint rows, heads;
  float scale;
  uint pad;
};

// A row's record (ints): position, window row base, window base position (-1: ring of 128), cache row base, count.

// `rope` is added to the position for the RoPE row alone (a draft row's window holds one key at position 0).
struct TfAttnMeta {
  int pos, wofs, wbase, cofs, ncomp, rope, p1, p2;
};

// One lane's 16 dims of a key: 32 bytes of bf16.
struct TfKey {
  uint4 a, b;
};

struct TfQ {
  float v[TF_ATTN_HS][16];
};

// A chunk's softmax state: running max, sum, weighted sum, and how many keys it has met.
struct TfAcc {
  float m[TF_ATTN_HS];
  float l[TF_ATTN_HS];
  float acc[TF_ATTN_HS][16];
  uint n;
};

inline float tf_attn_lo(uint w) { return as_type<float>(w << 16); }
inline float tf_attn_hi(uint w) { return as_type<float>(w & 0xFFFF0000u); }

inline void tf_attn_x16(TfKey k, thread float* x) {
  for (uint j = 0; j < 4; ++j) {
    x[2 * j] = tf_attn_lo(k.a[j]);
    x[2 * j + 1] = tf_attn_hi(k.a[j]);
    x[8 + 2 * j] = tf_attn_lo(k.b[j]);
    x[8 + 2 * j + 1] = tf_attn_hi(k.b[j]);
  }
}

inline void tf_attn_zero(thread TfAcc& s) {
  for (uint hh = 0; hh < TF_ATTN_HS; ++hh) {
    s.m[hh] = TF_ATTN_NEG;
    s.l[hh] = 0.0f;
    for (uint i = 0; i < 16; ++i) s.acc[hh][i] = 0.0f;
  }
  s.n = 0;
}

// The lane's 16 q dims of each head.
inline void tf_attn_loadq(thread TfQ& q, const device uint4* Q, uint r, uint h0, uint heads, uint lane) {
  for (uint hh = 0; hh < TF_ATTN_HS; ++hh) {
    const ulong at = (ulong(r) * heads + h0 + hh) * (TF_ATTN_D / 8) + 2 * lane;
    tf_attn_x16(TfKey{Q[at], Q[at + 1]}, q.v[hh]);
  }
}

inline float tf_attn_step(float acc, float q, float k) {
  const float r = fma(q, k, acc);
  return TF_CTL == 3 ? float(half(r)) : r;
}

// Online softmax, one key: the max moves only upward, and then l and acc are rescaled by exp(old - new).
inline void tf_attn_update(thread TfAcc& s, uint hh, float sc, thread const float* x) {
  if (TF_CTL == 1 && sc != sc) return;
  float m = s.m[hh];
  if (sc > m) {
    const float a = precise::exp(m - sc);
    s.l[hh] = s.l[hh] * a;
    for (uint i = 0; i < 16; ++i) s.acc[hh][i] = s.acc[hh][i] * a;
    s.m[hh] = sc;
    m = sc;
  }
  const float p = precise::exp(sc - m);
  s.l[hh] = s.l[hh] + p;
  for (uint i = 0; i < 16; ++i) s.acc[hh][i] = fma(p, x[i], s.acc[hh][i]);
}

// One lane's partial dot of a key with q: its 16 FMAs in dim order.
inline float tf_attn_dot(thread const float* q, TfKey k) {
  float x[16];
  tf_attn_x16(k, x);
  float v = 0.0f;
  for (uint i = 0; i < 16; ++i) v = tf_attn_step(v, q[i], x[i]);
  return v;
}

// The two lane sums of two keys added across the simdgroup in the adjacent-pair order; lane l ends holding key l & 1.
inline float tf_attn_reduce2(float v0, float v1, uint lane) {
  const bool b0 = (lane & 1u) != 0;
  float w = (b0 ? v1 : v0) + simd_shuffle_xor(b0 ? v0 : v1, 1);
  for (uint m = 2; m < 32; m *= 2) w += simd_shuffle_xor(w, m);
  return w;
}

// Two keys: the scores in the fixed lane order, then softmax key by key; a block that never raises the max shares exps.
inline void tf_attn_batch(thread TfAcc& s, thread const TfQ& q, thread const TfKey* k, uint valid, float scale,
                          uint lane) {
  const float w = tf_attn_reduce2(tf_attn_dot(q.v[0], k[0]), tf_attn_dot(q.v[0], k[1]), lane) * scale;
  const bool mine = ((valid >> (lane & 1u)) & 1u) != 0;
  if (!simd_any(mine && w > s.m[0])) {
    const float p = precise::exp(w - s.m[0]);
    for (uint u = 0; u < TF_ATTN_KU; ++u)
      if ((valid >> u) & 1u) {
        float x[16];
        tf_attn_x16(k[u], x);
        const float pu = simd_broadcast(p, u);
        s.l[0] = s.l[0] + pu;
        for (uint i = 0; i < 16; ++i) s.acc[0][i] = fma(pu, x[i], s.acc[0][i]);
        s.n += 1;
      }
    return;
  }
  for (uint u = 0; u < TF_ATTN_KU; ++u)
    if ((valid >> u) & 1u) {
      float x[16];
      tf_attn_x16(k[u], x);
      tf_attn_update(s, 0, simd_broadcast(w, u), x);
      s.n += 1;
    }
}

inline uint tf_attn_window_row(thread const TfAttnMeta& m, int t) {
  return uint(m.wofs + (m.wbase < 0 ? (t & int(TF_ATTN_WIN - 1)) : t - m.wbase));
}

// The window keys at positions t .. t + KU - 1 (those past the row's own position are absent).
inline uint tf_attn_load_window(thread TfKey* k, const device uint4* W, thread const TfAttnMeta& m, int lo, int t,
                                uint lane) {
  uint valid = 0;
  for (uint u = 0; u < TF_ATTN_KU; ++u) {
    const bool ok = t + int(u) <= m.pos;
    const ulong at = ulong(tf_attn_window_row(m, ok ? t + int(u) : lo)) * (TF_ATTN_D / 8) + 2 * lane;
    k[u] = TfKey{W[at], W[at + 1]};
    valid |= (ok ? 1u : 0u) << u;
  }
  return valid;
}

// The selected rows i .. i + KU - 1 of a run ending at `end`: their key registers, and a bit for each that exists.
template <typename IP>
inline uint tf_attn_load_selected(thread TfKey* k, const device uint4* C, IP I,
                                  thread const TfAttnMeta& m, int i, int end, uint lane) {
  uint valid = 0;
  for (uint u = 0; u < TF_ATTN_KU; ++u) {
    const int j = i + int(u);
    const int idx = j < end ? I[j] : -1;
    const bool ok = idx >= 0 && !(TF_CTL == 2 && j == m.ncomp - 1);
    const ulong at = ulong(ok ? m.cofs + idx : m.cofs) * (TF_ATTN_D / 8) + 2 * lane;
    k[u] = TfKey{C[at], C[at + 1]};
    valid |= (ok ? 1u : 0u) << u;
  }
  return valid;
}

// Selected run `run` (0..3) into a zeroed state; PF loads the next step's keys before this step's math.

// `I` is the row's index list, in device memory (prompt) or staged in threadgroup memory (decode).
template <bool PF = false, typename IP = const device int*>
inline void tf_attn_selected(thread TfAcc& s, thread const TfQ& q, const device uint4* C, IP I,
                             thread const TfAttnMeta& m, uint run, float scale, uint lane) {
  const int end = min(m.ncomp, int((run + 1) * TF_ATTN_RUN));
  if (PF) {
    int i = int(run * TF_ATTN_RUN);
    if (i >= end) return;
    TfKey k[TF_ATTN_KU];
    uint valid = tf_attn_load_selected(k, C, I, m, i, end, lane);
    for (; i < end; i += int(TF_ATTN_KU)) {
      TfKey kn[TF_ATTN_KU];
      const uint vn = i + int(TF_ATTN_KU) < end ? tf_attn_load_selected(kn, C, I, m, i + int(TF_ATTN_KU), end, lane) : 0u;
      tf_attn_batch(s, q, k, valid, scale, lane);
      for (uint u = 0; u < TF_ATTN_KU; ++u) k[u] = kn[u];
      valid = vn;
    }
    return;
  }
  for (int i = int(run * TF_ATTN_RUN); i < end; i += int(TF_ATTN_KU)) {
    TfKey k[TF_ATTN_KU];
    const uint valid = tf_attn_load_selected(k, C, I, m, i, end, lane);
    tf_attn_batch(s, q, k, valid, scale, lane);
  }
}

// The window chunk of a decode row; PF loads the next step's keys before this step's math.
template <bool PF = false>
inline void tf_attn_window(thread TfAcc& s, thread const TfQ& q, const device uint4* W, thread const TfAttnMeta& m,
                           float scale, uint lane) {
  const int lo = max(m.pos - int(TF_ATTN_WIN) + 1, 0);
  if (PF) {
    TfKey k[TF_ATTN_KU];
    uint valid = tf_attn_load_window(k, W, m, lo, lo, lane);
    for (int t = lo; t <= m.pos; t += int(TF_ATTN_KU)) {
      TfKey kn[TF_ATTN_KU];
      const int tn = t + int(TF_ATTN_KU);
      const uint vn = tn <= m.pos ? tf_attn_load_window(kn, W, m, lo, tn, lane) : 0u;
      tf_attn_batch(s, q, k, valid, scale, lane);
      for (uint u = 0; u < TF_ATTN_KU; ++u) k[u] = kn[u];
      valid = vn;
    }
    return;
  }
  for (int t = lo; t <= m.pos; t += int(TF_ATTN_KU)) {
    TfKey k[TF_ATTN_KU];
    const uint valid = tf_attn_load_window(k, W, m, lo, t, lane);
    tf_attn_batch(s, q, k, valid, scale, lane);
  }
}

// A chunk's (max, sum, weighted sum) joined to the running result of head hh: both rescaled to the larger max.
inline void tf_attn_join(thread TfAcc& t, uint hh, float mc, float lc, thread const float* ac) {
  const float nm = mc > t.m[hh] ? mc : t.m[hh];
  const float a = precise::exp(t.m[hh] - nm);
  const float b = precise::exp(mc - nm);
  const float l1 = t.l[hh] * a;
  const float l2 = lc * b;
  t.l[hh] = l1 + l2;
  for (uint i = 0; i < 16; ++i) {
    const float x1 = t.acc[hh][i] * a;
    const float x2 = ac[i] * b;
    t.acc[hh][i] = x1 + x2;
  }
  t.m[hh] = nm;
}

// Sink, normalisation, the inverse RoPE of the last 64 dims and the store of a head's row.
template <typename T>
inline void tf_attn_store(thread TfAcc& s, uint hh, float sink, const device float* cs, device T* out, uint lane) {
  const float m = s.m[hh];
  const float top = sink > m ? sink : m;
  const float af = precise::exp(m - top);
  const float e = precise::exp(sink - top);
  const float den = fma(s.l[hh], af, e);
  float x[16];
  for (uint i = 0; i < 16; ++i) x[i] = (s.acc[hh][i] * af) / den;
  if (lane >= 28) {
    for (uint i = 0; i < 16; i += 2) {
      const uint pr = 8 * (lane - 28) + i / 2;
      const float c = cs[pr], sn = cs[32 + pr];
      const float ne = fma(x[i], c, x[i + 1] * sn);
      const float no = fma(x[i + 1], c, -(x[i] * sn));
      x[i] = ne;
      x[i + 1] = no;
    }
  }
  for (uint i = 0; i < 16; ++i) out[16 * lane + i] = T(x[i]);
}

#define TF_ATTN_ARGS                                                                                              \
  const device uint4* Q [[buffer(0)]], const device uint4* W [[buffer(1)]], const device uint4* C [[buffer(2)]], \
      const device int* I [[buffer(3)]], const device TfAttnMeta* M [[buffer(4)]],                               \
      const device float* SK [[buffer(5)]], const device float* CS [[buffer(6)]], device T* O [[buffer(7)]],     \
      constant TfAttnArgs& a [[buffer(8)]]

// Decode: a threadgroup is one row, HS heads and a simdgroup for each of the five chunks; simdgroup 0 joins them.

// A selected run's simdgroup first copies its 128 indices to threadgroup memory, so no key load waits on an index load.
template <typename T, bool PF>
[[kernel, max_total_threads_per_threadgroup(32 * TF_ATTN_NCH)]] void tf_dsv41_attn_dec(
    TF_ATTN_ARGS, uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tacc[TF_ATTN_NCH][TF_ATTN_HS][TF_ATTN_D];
  threadgroup float tml[TF_ATTN_NCH][TF_ATTN_HS][2];
  threadgroup uint tn[TF_ATTN_NCH];
  threadgroup int tidx[TF_ATTN_SEL];
  const uint r = tg.x, h0 = tg.y * TF_ATTN_HS;
  if (r >= a.rows || h0 >= a.heads) return;
  if (sg != 0)
    for (uint u = 0; u < TF_ATTN_RUN / 32; ++u) {
      const uint j = (sg - 1) * TF_ATTN_RUN + 32 * u + lane;
      tidx[j] = I[ulong(r) * TF_ATTN_SEL + j];
    }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const TfAttnMeta m = M[r];
  TfQ q;
  tf_attn_loadq(q, Q, r, h0, a.heads, lane);
  TfAcc s;
  tf_attn_zero(s);
  if (sg == 0) tf_attn_window<PF>(s, q, W, m, a.scale, lane);
  else tf_attn_selected<PF>(s, q, C, (const threadgroup int*)tidx, m, sg - 1, a.scale, lane);
  for (uint hh = 0; hh < TF_ATTN_HS; ++hh)
    for (uint i = 0; i < 16; ++i) tacc[sg][hh][16 * lane + i] = s.acc[hh][i];
  if (lane < 2 * TF_ATTN_HS) tml[sg][lane / 2][lane % 2] = (lane % 2) ? s.l[lane / 2] : s.m[lane / 2];
  if (lane == 0) tn[sg] = s.n;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  for (uint c = 1; c < TF_ATTN_NCH; ++c) {
    if (tn[c] == 0) continue;
    for (uint hh = 0; hh < TF_ATTN_HS; ++hh) {
      float ac[16];
      for (uint i = 0; i < 16; ++i) ac[i] = tacc[c][hh][16 * lane + i];
      tf_attn_join(s, hh, tml[c][hh][0], tml[c][hh][1], ac);
    }
  }
  for (uint hh = 0; hh < TF_ATTN_HS; ++hh)
    tf_attn_store(s, hh, SK[h0 + hh], CS + ulong(m.pos + m.rope) * 64, O + (ulong(r) * a.heads + h0 + hh) * TF_ATTN_D,
                  lane);
  if (TF_CTL == 4 && r == 0 && h0 == 0 && lane == 0) O[ulong(a.rows) * a.heads * TF_ATTN_D] = T(1.0f);
}

// Prompt: a threadgroup is RR consecutive rows of one linear window buffer by HG head groups.

// The window is staged in tiles of 16 keys at absolute positions, one barrier pair a tile.

// Each row reads its own span of a tile, so its keys and their order do not depend on its chunk or the rows beside it.
template <typename T>
[[kernel, max_total_threads_per_threadgroup(32 * TF_ATTN_RR * TF_ATTN_HG)]] void tf_dsv41_attn_pre(
    TF_ATTN_ARGS, uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint4 tile[TF_ATTN_TILE * (TF_ATTN_D / 8)];
  const uint ra = sg / TF_ATTN_HG, r0 = tg.x * TF_ATTN_RR, r = r0 + ra;
  const uint h0 = (tg.y * TF_ATTN_HG + sg % TF_ATTN_HG) * TF_ATTN_HS;
  const bool on = r < a.rows && h0 < a.heads;
  const TfAttnMeta m0 = M[r0];
  const int last = M[min(r0 + TF_ATTN_RR, a.rows) - 1].pos;
  const int lo = max(m0.pos - int(TF_ATTN_WIN) + 1, 0);
  const TfAttnMeta m = on ? M[r] : m0;
  const float scale = (TF_CTL == 5 && ra == 1) ? a.scale * 1.0000001f : a.scale;
  TfQ q;
  TfAcc s;
  tf_attn_zero(s);
  if (on) tf_attn_loadq(q, Q, r, h0, a.heads, lane);
  for (int ti = lo / int(TF_ATTN_TILE); ti <= last / int(TF_ATTN_TILE); ++ti) {
    const int base = ti * int(TF_ATTN_TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint u = 0; u < TF_ATTN_TILE * (TF_ATTN_D / 8) / (32 * TF_ATTN_RR * TF_ATTN_HG); ++u) {
      const uint g = tid + 32 * TF_ATTN_RR * TF_ATTN_HG * u;
      const int t = base + int(g / (TF_ATTN_D / 8));
      if (t >= lo && t <= last) tile[g] = W[ulong(m0.wofs + t - m0.wbase) * (TF_ATTN_D / 8) + g % (TF_ATTN_D / 8)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!on) continue;
    const int ja = max(base, max(m.pos - int(TF_ATTN_WIN) + 1, 0)), jb = min(base + int(TF_ATTN_TILE) - 1, m.pos);
    for (int j = ja; j <= jb; j += int(TF_ATTN_KU)) {
      TfKey k[TF_ATTN_KU];
      uint valid = 0;
      for (uint u = 0; u < TF_ATTN_KU; ++u) {
        const bool ok = j + int(u) <= jb;
        const uint at = uint((ok ? j + int(u) : ja) - base) * (TF_ATTN_D / 8) + 2 * lane;
        k[u] = TfKey{tile[at], tile[at + 1]};
        valid |= (ok ? 1u : 0u) << u;
      }
      tf_attn_batch(s, q, k, valid, scale, lane);
    }
  }
  if (!on) return;
  for (uint run = 0; run < TF_ATTN_NCH - 1; ++run) {
    if (int(run * TF_ATTN_RUN) >= m.ncomp) break;
    TfAcc c;
    tf_attn_zero(c);
    tf_attn_selected(c, q, C, I + ulong(r) * TF_ATTN_SEL, m, run, scale, lane);
    if (c.n == 0) continue;
    for (uint hh = 0; hh < TF_ATTN_HS; ++hh) tf_attn_join(s, hh, c.m[hh], c.l[hh], c.acc[hh]);
  }
  for (uint hh = 0; hh < TF_ATTN_HS; ++hh)
    tf_attn_store(s, hh, SK[h0 + hh], CS + ulong(m.pos) * 64, O + (ulong(r) * a.heads + h0 + hh) * TF_ATTN_D, lane);
}

template [[host_name("tf_dsv41_attn_dec_bf16")]] [[kernel]] decltype(tf_dsv41_attn_dec<bfloat, false>) tf_dsv41_attn_dec<bfloat, false>;
template [[host_name("tf_dsv41_attn_dec_f32")]] [[kernel]] decltype(tf_dsv41_attn_dec<float, false>) tf_dsv41_attn_dec<float, false>;
template [[host_name("tf_dsv41_attn_dec_pf_bf16")]] [[kernel]] decltype(tf_dsv41_attn_dec<bfloat, true>) tf_dsv41_attn_dec<bfloat, true>;
template [[host_name("tf_dsv41_attn_dec_pf_f32")]] [[kernel]] decltype(tf_dsv41_attn_dec<float, true>) tf_dsv41_attn_dec<float, true>;
template [[host_name("tf_dsv41_attn_pre_bf16")]] [[kernel]] decltype(tf_dsv41_attn_pre<bfloat>) tf_dsv41_attn_pre<bfloat>;
template [[host_name("tf_dsv41_attn_pre_f32")]] [[kernel]] decltype(tf_dsv41_attn_pre<float>) tf_dsv41_attn_pre<float>;

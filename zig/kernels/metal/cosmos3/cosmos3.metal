// Cosmos3-Edge's reasoner beside the core row projections: the text model's glue (M-RoPE, relu^2, attention) and the
// SigLIP2 image tower's (layer norms, biases, GELU, bidirectional attention over one image's patches).
#include <metal_stdlib>
using namespace metal;

// The text model's pinned geometry.
constant constexpr uint D = 2048;
constant constexpr uint QH = 16;
constant constexpr uint KVH = 8;
constant constexpr uint HD = 128;
// Rotary pairs on each M-RoPE axis (t, h, w), interleaved as Qwen3-VL's: h at j % 3 == 1, w at j % 3 == 2.
constant constexpr uint SEC_H = 20;
constant constexpr uint SEC_W = 20;

// Table row ids[r] as bf16 scale * q + bias (MLX's dequantize), 8-bit codes, one byte a thread.
[[kernel]] void cos3_embed_b8(const device uint* ids [[buffer(0)]], const device uint8_t* table [[buffer(1)]],
                              const device bfloat* scales [[buffer(2)]], const device bfloat* biases [[buffer(3)]],
                              device bfloat* out [[buffer(4)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= D) return;
  const long id = long(ids[at.y]);
  const bfloat scale = scales[id * (D / 64) + at.x / 64];
  const bfloat bias = biases[id * (D / 64) + at.x / 64];
  out[size_t(at.y) * D + at.x] = scale * table[id * D + at.x] + bias;
}

// Image rows over their placeholders: row r takes rows[map[r]] when map[r] >= 0.
[[kernel]] void cos3_place(device bfloat* h [[buffer(0)]], const device bfloat* rows [[buffer(1)]], const device int* map [[buffer(2)]],
                           uint2 at [[thread_position_in_grid]]) {
  if (at.x >= D) return;
  const int src = map[at.y];
  if (src >= 0) h[size_t(at.y) * D + at.x] = rows[size_t(src) * D + at.x];
}

// h += r (bf16) when `add`, then x = w * bf16(h * rsqrt(mean(h^2) + eps)), 4 values a thread (Llama's RMSNorm order).
[[kernel]] void cos3_norm(device bfloat* h [[buffer(0)]], const device bfloat* r [[buffer(1)]], const device bfloat* w [[buffer(2)]],
                          constant float& eps [[buffer(3)]], constant uint& add [[buffer(4)]], device bfloat* x [[buffer(5)]],
                          uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const size_t base = size_t(row) * D + 4 * t;
  float v[4];
  float sq = 0.0f;
  for (int j = 0; j < 4; ++j) {
    bfloat hv = h[base + j];
    if (add != 0) {
      hv = hv + r[base + j];
      h[base + j] = hv;
    }
    v[j] = hv;
    sq += v[j] * v[j];
  }
  threadgroup float sums[32];
  threadgroup float inv[1];
  sq = simd_sum(sq);
  if (sg == 0) sums[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) sums[sg] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float all = simd_sum(sums[lane]);
    if (lane == 0) inv[0] = metal::precise::rsqrt(all / float(D) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int j = 0; j < 4; ++j) x[base + j] = w[4 * t + j] * static_cast<bfloat>(v[j] * inv[0]);
}

// The axis of rotary pair j (0 t, 1 h, 2 w).
inline uint cos3_axis(uint j) {
  if (j % 3 == 1 && j < 3 * SEC_H) return 1;
  if (j % 3 == 2 && j < 3 * SEC_W) return 2;
  return 0;
}

// One head of one row rotated (rotate_half pairs j and j + HD/2) at its three positions; lane l: pairs l and l + 32.
inline void cos3_rotate(const device bfloat* src, const thread int* pos, float log2_theta, uint lane, device bfloat* dst) {
  for (uint i = 0; i < 2; ++i) {
    const uint j = lane + 32 * i;
    const float inv_freq = metal::exp2(-(float(2 * j) / float(HD)) * log2_theta);
    const float angle = float(pos[cos3_axis(j)]) * inv_freq;
    const float c = metal::precise::cos(angle), s = metal::precise::sin(angle);
    const float a = src[j], b = src[j + HD / 2];
    dst[j] = static_cast<bfloat>(a * c - b * s);
    dst[j + HD / 2] = static_cast<bfloat>(b * c + a * s);
  }
}

// A simdgroup a (head, row): queries rotated into `queries`; keys rotated and values copied into the [KVH, cap, HD]
// caches at slot slots[r] (the row's place in its stream), positions pos[r] = (t, h, w).
struct cos3_qk_args { uint cap; float log2_theta; };
[[kernel]] void cos3_qk(const device bfloat* q [[buffer(0)]], const device bfloat* k [[buffer(1)]], const device bfloat* v [[buffer(2)]],
                        const device int* pos [[buffer(3)]], const device uint* slots [[buffer(4)]], constant cos3_qk_args& p [[buffer(5)]],
                        device bfloat* queries [[buffer(6)]], device bfloat* keys [[buffer(7)]], device bfloat* values [[buffer(8)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint head = tg.x, r = tg.y;
  const int at[3] = {pos[3 * r], pos[3 * r + 1], pos[3 * r + 2]};
  if (head < QH) {
    cos3_rotate(q + (size_t(r) * QH + head) * HD, at, p.log2_theta, lane, queries + (size_t(r) * QH + head) * HD);
    return;
  }
  const uint kh = head - QH;
  const size_t dst = (size_t(kh) * p.cap + slots[r]) * HD;
  cos3_rotate(k + (size_t(r) * KVH + kh) * HD, at, p.log2_theta, lane, keys + dst);
  for (uint d = lane; d < HD; d += 32) values[dst + d] = v[(size_t(r) * KVH + kh) * HD + d];
}

// x = bf16(relu(x)^2) in place.
[[kernel]] void cos3_relu2(device bfloat* x [[buffer(0)]], constant uint& count [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  if (i >= count) return;
  const float v = max(float(x[i]), 0.0f);
  x[i] = static_cast<bfloat>(v * v);
}

// Attention, one simdgroup a (head, row): queries [rows, heads, HDT] (rows q_row apart), keys and values at
// key j of kv head kh: base + kh * k_head + j * k_row; a causal row r sees keys [0, first + r], else all `keys`.
// Each 32-key block: a lane's key scored, the block's softmax folded in, then every lane's dims take the block's values.
struct cos3_attn_args { uint first, keys, causal, group, q_row, k_row, k_head, o_row; float scale; };
template <int HDT>
[[kernel]] void cos3_attn(const device bfloat* q [[buffer(0)]], const device bfloat* k [[buffer(1)]], const device bfloat* v [[buffer(2)]],
                          constant cos3_attn_args& p [[buffer(3)]], device bfloat* o [[buffer(4)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NS = (HDT + 31) / 32;
  const uint head = tg.x, r = tg.y, kh = head / p.group;
  threadgroup float qs[HDT];
  for (uint d = lane; d < uint(HDT); d += 32) qs[d] = float(q[size_t(r) * p.q_row + head * HDT + d]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n = p.causal != 0 ? p.first + r + 1 : p.keys;
  const device bfloat* kb = k + size_t(kh) * p.k_head;
  const device bfloat* vb = v + size_t(kh) * p.k_head;
  float m = -INFINITY, l = 0.0f;
  float acc[NS];
  for (int i = 0; i < NS; ++i) acc[i] = 0.0f;
  for (uint j0 = 0; j0 < n; j0 += 32) {
    const uint j = j0 + lane;
    float s = -INFINITY;
    if (j < n) {
      const device bfloat* kr = kb + size_t(j) * p.k_row;
      float dot = 0.0f;
      for (int d = 0; d < HDT; d += 4) {
        const float4 kv4 = float4(*(const device vec<bfloat, 4>*)(kr + d));
        dot += qs[d] * kv4.x + qs[d + 1] * kv4.y + qs[d + 2] * kv4.z + qs[d + 3] * kv4.w;
      }
      s = dot * p.scale;
    }
    const float mn = max(m, simd_max(s));
    const float pj = j < n ? metal::exp(s - mn) : 0.0f;
    const float alpha = m == -INFINITY ? 0.0f : metal::exp(m - mn);
    l = l * alpha + simd_sum(pj);
    for (int i = 0; i < NS; ++i) acc[i] *= alpha;
    const uint count = min(32u, n - j0);
    for (uint t = 0; t < count; ++t) {
      const float pt = simd_shuffle(pj, ushort(t));
      const device bfloat* vr = vb + size_t(j0 + t) * p.k_row;
      for (int i = 0; i < NS; ++i) {
        const uint d = lane + 32 * i;
        if (d < uint(HDT)) acc[i] += pt * float(vr[d]);
      }
    }
    m = mn;
  }
  for (int i = 0; i < NS; ++i) {
    const uint d = lane + 32 * i;
    if (d < uint(HDT)) o[size_t(r) * p.o_row + head * HDT + d] = static_cast<bfloat>(acc[i] / l);
  }
}
template [[host_name("cos3_attn_128")]] [[kernel]] decltype(cos3_attn<128>) cos3_attn<128>;
template [[host_name("cos3_attn_72")]] [[kernel]] decltype(cos3_attn<72>) cos3_attn<72>;

// cos3_attn over 8 query rows a threadgroup (one simdgroup each, one head): each 32-key block of keys and values is
// staged once in threadgroup memory (keys padded to an odd stride) for all 8 rows. Causal rows mask past their own.
constant constexpr uint TR = 8;
template <int HDT>
[[kernel]] void cos3_attn_tile(const device bfloat* q [[buffer(0)]], const device bfloat* k [[buffer(1)]], const device bfloat* v [[buffer(2)]],
                               constant cos3_attn_args& p [[buffer(3)]], device bfloat* o [[buffer(4)]], constant uint& rows [[buffer(5)]],
                               uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NS = (HDT + 31) / 32;
  constexpr int KS = HDT + 1;  // odd: lane j's key row starts in its own bank
  threadgroup bfloat ks[32 * KS];  // bf16 values as the caches hold them: staged exactly
  threadgroup bfloat vs[32 * HDT];
  threadgroup float qs[TR][HDT];
  const uint head = tg.x, r0 = tg.y * TR, kh = head / p.group;
  const uint r = r0 + sg;
  const bool live = r < rows;
  const uint rr = live ? r : rows - 1;
  for (uint d = lane; d < uint(HDT); d += 32) qs[sg][d] = float(q[size_t(rr) * p.q_row + head * HDT + d]);
  const uint n = p.causal != 0 ? p.first + rr + 1 : p.keys;
  const uint top = p.causal != 0 ? p.first + min(r0 + TR, rows) : p.keys;  // the most keys a row here sees
  const device bfloat* kb = k + size_t(kh) * p.k_head;
  const device bfloat* vb = v + size_t(kh) * p.k_head;
  float m = -INFINITY, l = 0.0f;
  float acc[NS];
  for (int i = 0; i < NS; ++i) acc[i] = 0.0f;
  for (uint j0 = 0; j0 < top; j0 += 32) {
    const uint count = min(32u, top - j0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 32 * uint(HDT); e += TR * 32) {
      const uint kj = e / HDT, d = e % HDT;
      const bool in = kj < count;
      ks[kj * KS + d] = in ? kb[size_t(j0 + kj) * p.k_row + d] : bfloat(0.0f);
      vs[kj * HDT + d] = in ? vb[size_t(j0 + kj) * p.k_row + d] : bfloat(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint j = j0 + lane;
    float s = -INFINITY;
    if (j < n) {
      float dot = 0.0f;
      for (int d = 0; d < HDT; ++d) dot += qs[sg][d] * float(ks[lane * KS + d]);
      s = dot * p.scale;
    }
    const float mb = simd_max(s);
    if (mb == -INFINITY) continue;  // every key of this block is past this row
    const float mn = max(m, mb);
    const float pj = j < n ? metal::exp(s - mn) : 0.0f;
    const float alpha = m == -INFINITY ? 0.0f : metal::exp(m - mn);
    l = l * alpha + simd_sum(pj);
    for (int i = 0; i < NS; ++i) acc[i] *= alpha;
    const uint upto = min(count, n > j0 ? n - j0 : 0u);
    for (uint t = 0; t < upto; ++t) {
      const float pt = simd_shuffle(pj, ushort(t));
      for (int i = 0; i < NS; ++i) {
        const uint d = lane + 32 * i;
        if (d < uint(HDT)) acc[i] += pt * float(vs[t * HDT + d]);
      }
    }
    m = mn;
  }
  if (!live) return;
  for (int i = 0; i < NS; ++i) {
    const uint d = lane + 32 * i;
    if (d < uint(HDT)) o[size_t(r) * p.o_row + head * HDT + d] = static_cast<bfloat>(acc[i] / l);
  }
}
template [[host_name("cos3_attn_tile_128")]] [[kernel]] decltype(cos3_attn_tile<128>) cos3_attn_tile<128>;
template [[host_name("cos3_attn_tile_72")]] [[kernel]] decltype(cos3_attn_tile<72>) cos3_attn_tile<72>;

// Flash attention on simdgroup matrices: a threadgroup a (head, 32 query rows), a simdgroup 8 of them. Each 32-key block
// of keys and values is staged in threadgroup memory; S = Q K^T on 8 x 8 bf16 tiles (fp32 sums), the block's online
// softmax by rows (4 lanes a row), P rounded to bf16 as the eager attention's weights are, then O += P V.
constant constexpr uint FQ = 32;  // query rows a threadgroup
constant constexpr uint FK = 32;  // keys a block
template <int HDT>
[[kernel]] void cos3_flash(const device bfloat* q [[buffer(0)]], const device bfloat* k [[buffer(1)]], const device bfloat* v [[buffer(2)]],
                           constant cos3_attn_args& p [[buffer(3)]], device bfloat* o [[buffer(4)]], constant uint& rows [[buffer(5)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int DT = HDT / 8;
  threadgroup bfloat ks[FK * HDT];
  threadgroup bfloat vs[FK * HDT];
  threadgroup float ss[4][8 * FK];
  threadgroup bfloat ps[4][8 * FK];
  threadgroup float rowv[4][8];
  const uint head = tg.x, kh = head / p.group;
  const uint r0 = tg.y * FQ, s0 = r0 + 8 * sg;  // this simdgroup's first row
  const device bfloat* kb = k + size_t(kh) * p.k_head;
  const device bfloat* vb = v + size_t(kh) * p.k_head;
  // the lane's two elements of every 8 x 8 tile: row fm, columns fn and fn + 1
  const uint qid = lane / 4, fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
  simdgroup_matrix<bfloat, 8, 8> Q[DT];
  for (int d = 0; d < DT; ++d) simdgroup_load(Q[d], q + size_t(s0) * p.q_row + head * HDT, p.q_row, ulong2(8 * d, 0));
  simdgroup_matrix<float, 8, 8> O[DT];
  for (int d = 0; d < DT; ++d) O[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  // softmax state of row (lane / 4), the same in its four lanes
  const uint srow = lane / 4, scol = (lane % 4) * 8;
  const uint my_row = s0 + srow;
  const uint my_n = p.causal != 0 ? p.first + my_row + 1 : p.keys;
  float m = -INFINITY, l = 0.0f;
  const uint top = p.causal != 0 ? p.first + min(r0 + FQ, rows) : p.keys;
  for (uint j0 = 0; j0 < top; j0 += FK) {
    const uint count = min(FK, top - j0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < FK * uint(HDT) / 4; e += 128) {
      const uint kj = (4 * e) / HDT, d = (4 * e) % HDT;
      vec<bfloat, 4> kv = vec<bfloat, 4>(bfloat(0.0f)), vv = vec<bfloat, 4>(bfloat(0.0f));
      if (kj < count) {
        kv = *(const device vec<bfloat, 4>*)(kb + size_t(j0 + kj) * p.k_row + d);
        vv = *(const device vec<bfloat, 4>*)(vb + size_t(j0 + kj) * p.k_row + d);
      }
      *(threadgroup vec<bfloat, 4>*)(ks + kj * HDT + d) = kv;
      *(threadgroup vec<bfloat, 4>*)(vs + kj * HDT + d) = vv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // S [8, 32] = Q K^T
    for (uint kt = 0; kt < FK / 8; ++kt) {
      simdgroup_matrix<float, 8, 8> S = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      for (int d = 0; d < DT; ++d) {
        simdgroup_matrix<bfloat, 8, 8> KT;
        simdgroup_load(KT, ks, HDT, ulong2(8 * d, 8 * kt), true);
        simdgroup_multiply_accumulate(S, Q[d], KT, S);
      }
      simdgroup_store(S, ss[sg], FK, ulong2(8 * kt, 0));
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float sv[8];
    float bm = -INFINITY;
    for (uint c = 0; c < 8; ++c) {
      const uint j = j0 + scol + c;
      sv[c] = (j < my_n && scol + c < count) ? ss[sg][srow * FK + scol + c] * p.scale : -INFINITY;
      bm = max(bm, sv[c]);
    }
    bm = max(bm, simd_shuffle_xor(bm, ushort(1)));
    bm = max(bm, simd_shuffle_xor(bm, ushort(2)));
    const float mn = max(m, bm);
    float alpha = 1.0f, sum = 0.0f;
    if (mn == -INFINITY) {
      for (uint c = 0; c < 8; ++c) ps[sg][srow * FK + scol + c] = bfloat(0.0f);
    } else {
      alpha = m == -INFINITY ? 0.0f : metal::exp(m - mn);
      for (uint c = 0; c < 8; ++c) {
        const float e = sv[c] == -INFINITY ? 0.0f : metal::exp(sv[c] - mn);
        sum += e;
        ps[sg][srow * FK + scol + c] = static_cast<bfloat>(e);
      }
      m = mn;
    }
    sum += simd_shuffle_xor(sum, ushort(1));
    sum += simd_shuffle_xor(sum, ushort(2));
    l = l * alpha + sum;
    if (lane % 4 == 0) rowv[sg][srow] = alpha;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const float a = rowv[sg][fm];
    for (int d = 0; d < DT; ++d) reinterpret_cast<thread float2&>(O[d].thread_elements()) *= a;
    for (uint kt = 0; kt < FK / 8; ++kt) {
      simdgroup_matrix<bfloat, 8, 8> P;
      simdgroup_load(P, ps[sg], FK, ulong2(8 * kt, 0));
      for (int d = 0; d < DT; ++d) {
        simdgroup_matrix<bfloat, 8, 8> V;
        simdgroup_load(V, vs, HDT, ulong2(8 * d, 8 * kt));
        simdgroup_multiply_accumulate(O[d], P, V, O[d]);
      }
    }
  }
  if (lane % 4 == 0) rowv[sg][srow] = l;
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const uint r = s0 + fm;
  if (r >= rows) return;
  const float inv = 1.0f / rowv[sg][fm];
  for (int d = 0; d < DT; ++d) {
    const float2 el = reinterpret_cast<thread float2&>(O[d].thread_elements());
    device bfloat* out = o + size_t(r) * p.o_row + head * HDT + 8 * d + fn;
    out[0] = static_cast<bfloat>(el.x * inv);
    out[1] = static_cast<bfloat>(el.y * inv);
  }
}
template [[host_name("cos3_flash_128")]] [[kernel]] decltype(cos3_flash<128>) cos3_flash<128>;
template [[host_name("cos3_flash_72")]] [[kernel]] decltype(cos3_flash<72>) cos3_flash<72>;

// h += r (bf16) when `add`, then x = LayerNorm(h) * w + b over `width` values (fp32 statistics), a threadgroup a row.
struct cos3_ln_args { uint width; float eps; uint add; };
[[kernel]] void cos3_layer_norm(device bfloat* h [[buffer(0)]], const device bfloat* r [[buffer(1)]], const device bfloat* w [[buffer(2)]],
                                const device bfloat* b [[buffer(3)]], constant cos3_ln_args& p [[buffer(4)]], device bfloat* x [[buffer(5)]],
                                uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                uint threads [[threads_per_threadgroup]]) {
  device bfloat* hr = h + size_t(row) * p.width;
  float s = 0.0f;
  for (uint i = t; i < p.width; i += threads) {
    bfloat hv = hr[i];
    if (p.add != 0) {
      hv = hv + r[size_t(row) * p.width + i];
      hr[i] = hv;
    }
    s += float(hv);
  }
  threadgroup float part[32];
  threadgroup float stat[2];
  s = simd_sum(s);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float all = simd_sum(lane < (threads + 31) / 32 ? part[lane] : 0.0f);
    if (lane == 0) stat[0] = all / float(p.width);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float mean = stat[0];
  float q = 0.0f;
  for (uint i = t; i < p.width; i += threads) {
    const float d = float(hr[i]) - mean;
    q += d * d;
  }
  q = simd_sum(q);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) part[sg] = q;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float all = simd_sum(lane < (threads + 31) / 32 ? part[lane] : 0.0f);
    if (lane == 0) stat[1] = metal::precise::rsqrt(all / float(p.width) + p.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = stat[1];
  for (uint i = t; i < p.width; i += threads)
    x[size_t(row) * p.width + i] = static_cast<bfloat>((float(hr[i]) - mean) * inv * float(w[i]) + float(b[i]));
}

// erf to ~1.5e-7 (Abramowitz and Stegun 7.1.26): Metal has none.
inline float cos3_erf(float x) {
  const float s = x < 0.0f ? -1.0f : 1.0f;
  const float a = fabs(x);
  const float t = 1.0f / (1.0f + 0.3275911f * a);
  const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * metal::exp(-a * a);
  return s * y;
}

// y[r, c] = act(bf16(y + bias[c])) in place over [rows, width]: act 0 none, 1 GELU (tanh), 2 GELU (erf).
struct cos3_bias_args { uint width, count, act; };
[[kernel]] void cos3_bias(device bfloat* y [[buffer(0)]], const device bfloat* bias [[buffer(1)]], constant cos3_bias_args& p [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
  if (i >= p.count) return;
  const float v = float(static_cast<bfloat>(float(y[i]) + float(bias[i % p.width])));
  float out = v;
  if (p.act == 1) {
    const float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    out = 0.5f * v * (1.0f + metal::precise::tanh(inner));
  } else if (p.act == 2) {
    out = 0.5f * v * (1.0f + cos3_erf(v * 0.7071067811865476f));
  }
  y[i] = static_cast<bfloat>(out);
}

// a += b (bf16) over `count` values.
[[kernel]] void cos3_add(device bfloat* a [[buffer(0)]], const device bfloat* b [[buffer(1)]], constant uint& count [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
  if (i < count) a[i] = a[i] + b[i];
}

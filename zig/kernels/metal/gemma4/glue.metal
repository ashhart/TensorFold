// Gemma 4's decode glue around the core lane projections: gilby's RowDecode kernels (#360) and MLX's ops it calls.
#include <metal_stdlib>
using namespace metal;

// Each pipeline's constants: D, T, DH, NQ, NK, VK, W, HK, G, CK, S, BLK, SCALE, GELU_C0, GELU_C1 (kernels.zig).

// A row's group sums for the reg lane kernel in tf_lane_sums' order: XS[g * stride + r], none when stride is 0.
inline void gemma_row_sums(const device bfloat* row, int len, device float* XS, int stride, int r, uint t, uint nt) {
  if (stride == 0) return;
  for (int g = int(t); g < len / 64; g += int(nt)) {
    float acc = 0.0f;
    for (int i = 0; i < 64; i++) acc += float(row[g * 64 + i]);
    XS[g * stride + r] = acc;
  }
}

// Embedding rows of the 4-bit table times the bf16 embedding scale: MLX's dequantize, then a bf16 multiply.
kernel void gemma_embed(const device uint* ids [[buffer(0)]], const device uchar* weight [[buffer(1)]],
                        const device bfloat* scales [[buffer(2)]], const device bfloat* biases [[buffer(3)]],
                        device bfloat* out [[buffer(4)]], constant float& scale [[buffer(5)]],
                        uint2 pos [[thread_position_in_grid]]) {
  const uint b = pos.x, r = pos.y;
  const size_t id = ids[r];
  const uchar v = weight[id * (D / 2) + b];
  const size_t g = id * (D / 64) + (2 * b) / 64;
  const float s = float(scales[g]), z = float(biases[g]);
  out[size_t(r) * D + 2 * b] = bfloat(float(bfloat(s * float(v & 0xf) + z)) * scale);
  out[size_t(r) * D + 2 * b + 1] = bfloat(float(bfloat(s * float(v >> 4) + z)) * scale);
}

// MLX's rms_looped (rows past 4096): 1024 threads, 4 reads a thread a pass, out = bf16(w * bf16(x * inv)).
kernel void gemma_rms_looped(const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                             device bfloat* out [[buffer(2)]], constant float& eps [[buffer(3)]],
                             device float* XS [[buffer(4)]], constant int& xst [[buffer(5)]],
                             uint gid [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
                             uint lsize [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float inv[1];
  threadgroup float sums[32];
  const device bfloat* xr = x + gid * size_t(D) + lid * 4;
  const device bfloat* wr = w + lid * 4;
  float acc = 0;
  for (uint r = 0; r < D; r += lsize * 4) {
    for (int i = 0; i < 4; i++) {
      if (r + lid * 4 + i < D) {
        const float xi = xr[i + r];
        acc += xi * xi;
      }
    }
  }
  acc = simd_sum(acc);
  if (sg == 0) sums[lane] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) sums[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    acc = simd_sum(sums[lane]);
    if (lane == 0) inv[0] = metal::precise::rsqrt(acc / D + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device bfloat* o = out + gid * size_t(D) + lid * 4;
  for (uint r = 0; r < D; r += lsize * 4)
    for (int i = 0; i < 4; i++)
      if (r + lid * 4 + i < D) o[r + i] = bfloat(float(wr[i + r]) * float(bfloat(float(xr[r + i]) * inv[0])));
  threadgroup_barrier(mem_flags::mem_device);
  gemma_row_sums(out + gid * size_t(D), D, XS, xst, int(gid), lid, lsize);
}

// gilby's _EMIT's arithmetic: a head's raw values xv (lane + 32 i) normed (v unweighted), q and k roped at pos, in y.
inline void gemma_norm_rope(int kind, uint lane, thread const float* xv, const device bfloat* QW, const device bfloat* KW,
                            const device float* INVF, float pos, float eps, thread float* y) {
  constexpr int PER = DH / 32;
  float ss = 0.0f;
  for (int i = 0; i < PER; i++) ss = fma(xv[i], xv[i], ss);
  ss = simd_sum(ss);
  const float inv = metal::precise::rsqrt(ss / float(DH) + eps);
  if (kind == 2) {
    for (int i = 0; i < PER; i++) y[i] = float(bfloat(xv[i] * inv));
    return;
  }
  const device bfloat* w = kind == 0 ? QW : KW;
  for (int i = 0; i < PER; i++) y[i] = float(bfloat(float(w[int(lane) + 32 * i]) * float(bfloat(xv[i] * inv))));
  for (int i = 0; i < PER / 2; i++) {
    const float theta = pos * INVF[int(lane) + 32 * i];
    const float c = metal::fast::cos(theta), s = metal::fast::sin(theta);
    const float x1 = y[i], x2 = y[i + PER / 2];
    y[i] = x1 * c - x2 * s;
    y[i + PER / 2] = x1 * s + x2 * c;
  }
}

// The stacked q|k|v row's head `slot` as raw values in a simdgroup's lanes; VK: v is k's raw values.
inline void gemma_head_raw(const device bfloat* QKV, int r, int slot, uint lane, thread float* xv) {
  constexpr int PER = DH / 32;
  const int kind = slot < NQ ? 0 : (slot < NQ + NK ? 1 : 2);
  const int head = kind == 0 ? slot : (kind == 1 ? slot - NQ : slot - NQ - NK);
  const int src = kind == 2 ? (VK ? NQ * DH : (NQ + NK) * DH) + head * DH : slot * DH;
  for (int i = 0; i < PER; i++) xv[i] = float(QKV[r * W + src + int(lane) + 32 * i]);
}

// gilby's _EMIT: the head's values to Q, K or V, and with cache slots (cm: ring or 0, slots) k and v to the row's slot.
inline void gemma_emit(int kind, int head, int r, int R, uint lane, thread float* xv, const device bfloat* QW,
                       const device bfloat* KW, const device float* INVF, const device int* POS, float eps,
                       device bfloat* Q, device bfloat* K, device bfloat* V, device bfloat* KC, device bfloat* VC,
                       int3 cm) {
  const size_t at = cm.y ? (size_t(head) * cm.y + (cm.x ? POS[r] % cm.x : POS[r])) * DH : 0;
  constexpr int PER = DH / 32;
  float y[PER];
  gemma_norm_rope(kind, lane, xv, QW, KW, INVF, float(POS[r]), eps, y);
  if (kind == 0) {
    for (int i = 0; i < PER; i++) Q[(r * NQ + head) * DH + int(lane) + 32 * i] = bfloat(y[i]);
    return;
  }
  device bfloat* out = (kind == 1 ? K : V) + (head * R + r) * DH;
  device bfloat* cache = kind == 1 ? KC : VC;
  for (int i = 0; i < PER; i++) {
    out[int(lane) + 32 * i] = bfloat(y[i]);
    if (cm.y) cache[at + int(lane) + 32 * i] = bfloat(y[i]);
  }
}

// gilby's qkv_prep: a simdgroup per (head slot, row) of the stacked q|k|v rows; VK: v is k's raw values.
kernel void gemma_qkv_prep(const device bfloat* QKV [[buffer(0)]], const device bfloat* QW [[buffer(1)]],
                           const device bfloat* KW [[buffer(2)]], const device float* INVF [[buffer(3)]],
                           const device int* POS [[buffer(4)]], constant float& eps [[buffer(5)]],
                           device bfloat* Q [[buffer(6)]], device bfloat* K [[buffer(7)]], device bfloat* V [[buffer(8)]],
                           device bfloat* KC [[buffer(9)]], device bfloat* VC [[buffer(10)]], constant int3& cm [[buffer(11)]],
                           uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]],
                           uint3 tgs [[threadgroups_per_grid]]) {
  const int slot = int(tg.x), r = int(tg.y), R = int(tgs.y);
  constexpr int PER = DH / 32;
  const int kind = slot < NQ ? 0 : (slot < NQ + NK ? 1 : 2);
  const int head = kind == 0 ? slot : (kind == 1 ? slot - NQ : slot - NQ - NK);
  float xv[PER];
  gemma_head_raw(QKV, r, slot, lane, xv);
  gemma_emit(kind, head, r, R, lane, xv, QW, KW, INVF, POS, eps, Q, K, V, KC, VC, cm);
}

// gilby's attend partials, sums fixed by key position; META: first chunk, chunks, ring, rows, first new key.
kernel void gemma_attention_partial(const device bfloat* Q [[buffer(0)]], const device bfloat* K [[buffer(1)]],
                                    const device bfloat* V [[buffer(2)]], const device bfloat* KN [[buffer(3)]],
                                    const device bfloat* VN [[buffer(4)]], const device int* POS [[buffer(5)]],
                                    const device int* LO [[buffer(6)]], const device int* META [[buffer(7)]],
                                    constant int& CAP [[buffer(8)]], device float* PM [[buffer(9)]],
                                    device float* PL [[buffer(10)]], device float* PO [[buffer(11)]],
                                    const device bfloat* QKV [[buffer(12)]], const device bfloat* QW [[buffer(13)]],
                                    const device bfloat* KW [[buffer(14)]], const device float* INVF [[buffer(15)]],
                                    constant float& eps [[buffer(16)]], constant int2& FUSE [[buffer(17)]],
                                    device bfloat* OUT [[buffer(18)]], device float* XS [[buffer(19)]],
                                    constant int& xst [[buffer(20)]], device bfloat* KC [[buffer(21)]],
                                    device bfloat* VC [[buffer(22)]],
                                    uint lane [[thread_index_in_simdgroup]], uint sgu [[simdgroup_index_in_threadgroup]],
                                    uint3 tg [[threadgroup_position_in_grid]]) {
  const int sgi = int(sgu);
  const int g = sgi / S, s = sgi % S;
  const int C0 = META[0], NCH = META[1], RING = META[2], R = META[3];
  const int c = C0 + int(tg.x);
  const int h = int(tg.y);
  const int r = int(tg.z);
  const int qh = h * G + g;
  const int P0 = META[4];
  constexpr int DPL = DH / 32;
  const int k0 = metal::max(c * CK, LO[r]), k1 = metal::min((c + 1) * CK, POS[r] + 1);
  float q[DPL], o[DPL];
  threadgroup bfloat qs[G][DH];
  // FUSE.x (one row, every key cached): gemma_qkv_prep's work here; the last chunk's group writes k and v to the cache
  if (FUSE.x) { // simdgroups 0..G-1 prep the query heads; G and G + 1 the key and value, into the cache at the row's slot
    float xv[DPL], y[DPL];
    if (sgi < G) {
      gemma_head_raw(QKV, r, h * G + sgi, lane, xv);
      gemma_norm_rope(0, lane, xv, QW, KW, INVF, float(POS[r]), eps, y);
      for (int i = 0; i < DPL; i++) qs[sgi][int(lane) + 32 * i] = bfloat(y[i]);
    } else if (sgi < G + 2 && int(tg.x) + 1 == NCH) {
      const int kind = sgi - G + 1;
      gemma_head_raw(QKV, r, NQ + (kind - 1) * NK + h, lane, xv);
      gemma_norm_rope(kind, lane, xv, QW, KW, INVF, float(POS[r]), eps, y);
      device bfloat* cache = (kind == 1 ? KC : VC) + (size_t(h) * CAP + (RING ? POS[r] % RING : POS[r])) * DH;
      for (int i = 0; i < DPL; i++) cache[int(lane) + 32 * i] = bfloat(y[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (int i = 0; i < DPL; i++) { q[i] = float(qs[g][lane * DPL + i]) * SCALE; o[i] = 0.0f; }
  } else {
    const device bfloat* qp = Q + (size_t(r) * (HK * G) + qh) * DH + lane * DPL;
    for (int i = 0; i < DPL; i++) { q[i] = float(qp[i]) * SCALE; o[i] = 0.0f; }
  }
  float m = -INFINITY, l = 0.0f;
  const device bfloat* kb = K + size_t(h) * CAP * DH + lane * DPL;
  const device bfloat* vb = V + size_t(h) * CAP * DH + lane * DPL;
  const device bfloat* kn = KN + size_t(h) * R * DH + lane * DPL;
  const device bfloat* vn = VN + size_t(h) * R * DH + lane * DPL;
  for (int base = k0 + s; base < k1; base += S * BLK) {
    float sc[BLK];
    size_t at[BLK];
    bool fresh[BLK], live[BLK];
    float bm = -INFINITY;
    for (int j = 0; j < BLK; j++) {
      const int p = base + j * S;
      live[j] = p < k1;
      fresh[j] = p >= P0;
      at[j] = size_t(fresh[j] ? p - P0 : (RING ? p % RING : p)) * DH;
      float d = 0.0f;
      if (live[j]) {
        const device bfloat* kr = (fresh[j] ? kn : kb) + at[j];
        for (int i = 0; i < DPL; i++) d = fma(q[i], float(kr[i]), d);
      }
      sc[j] = simd_sum(d);
      if (live[j]) bm = metal::max(bm, sc[j]);
    }
    const float mn = metal::max(m, bm);
    const float a = metal::exp(m - mn);
    l *= a;
    for (int i = 0; i < DPL; i++) o[i] *= a;
    for (int j = 0; j < BLK; j++) {
      if (!live[j]) continue;
      const float b = metal::exp(sc[j] - mn);
      l += b;
      const device bfloat* vr = (fresh[j] ? vn : vb) + at[j];
      for (int i = 0; i < DPL; i++) o[i] = fma(b, float(vr[i]), o[i]);
    }
    m = mn;
  }
  const size_t slot = (size_t(qh) * R + r) * NCH + (c - C0);
  if (S == 1) {
    if (FUSE.y) {
      for (int i = 0; i < DPL; i++) OUT[(size_t(r) * NQ + qh) * DH + lane * DPL + i] = bfloat(o[i] / l);
      simdgroup_barrier(mem_flags::mem_device);
      gemma_row_sums(OUT + (size_t(r) * NQ + qh) * DH, DH, XS + qh * (DH / 64) * xst, xst, r, lane, 32);
      return;
    }
    if (lane == 0) { PM[slot] = m; PL[slot] = l; }
    for (int i = 0; i < DPL; i++) PO[slot * DH + lane * DPL + i] = o[i];
    return;
  }
  threadgroup float sm[G * S], sl[G * S];
  threadgroup float so[S > 1 ? G * S : 1][S > 1 ? DH : 1];
  if (lane == 0) { sm[sgi] = m; sl[sgi] = l; }
  for (int i = 0; i < DPL; i++) so[sgi][lane * DPL + i] = o[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (s != 0) return;
  float top = -INFINITY;
  for (int t = 0; t < S; t++) top = metal::max(top, sm[g * S + t]);
  float lsum = 0.0f, acc[DPL];
  for (int i = 0; i < DPL; i++) acc[i] = 0.0f;
  for (int t = 0; t < S; t++) {
    const float e = sl[g * S + t] > 0.0f ? metal::exp(sm[g * S + t] - top) : 0.0f;
    lsum = fma(sl[g * S + t], e, lsum);
    for (int i = 0; i < DPL; i++) acc[i] = fma(so[g * S + t][lane * DPL + i], e, acc[i]);
  }
  if (FUSE.y) { // FUSE.y (one chunk): gemma_attention_merge's output and sums, exp(top - top) being 1, to OUT and XS
    for (int i = 0; i < DPL; i++) OUT[(size_t(r) * NQ + qh) * DH + lane * DPL + i] = bfloat(acc[i] / lsum);
    simdgroup_barrier(mem_flags::mem_device);
    gemma_row_sums(OUT + (size_t(r) * NQ + qh) * DH, DH, XS + qh * (DH / 64) * xst, xst, r, lane, 32);
    return;
  }
  if (lane == 0) { PM[slot] = top; PL[slot] = lsum; }
  for (int i = 0; i < DPL; i++) PO[slot * DH + lane * DPL + i] = acc[i];
}

// gilby's attend, merge: a simdgroup per (query head, row) adds the chunks' partials in chunk order.
kernel void gemma_attention_merge(const device float* PM [[buffer(0)]], const device float* PL [[buffer(1)]],
                                  const device float* PO [[buffer(2)]], const device int* META [[buffer(3)]],
                                  device bfloat* OUT [[buffer(4)]], device float* XS [[buffer(5)]],
                                  constant int& xst [[buffer(6)]], uint lane [[thread_index_in_simdgroup]],
                                  uint3 tg [[threadgroup_position_in_grid]]) {
  const int qh = int(tg.y), r = int(tg.z);
  const int NCH = META[1], R = META[3];
  constexpr int DPL = DH / 32;
  const size_t base = (size_t(qh) * R + r) * NCH;
  float top = -INFINITY;
  for (int c = 0; c < NCH; c++) top = metal::max(top, PM[base + c]);
  float lsum = 0.0f, acc[DPL];
  for (int i = 0; i < DPL; i++) acc[i] = 0.0f;
  for (int c = 0; c < NCH; c++) {
    const float e = PL[base + c] > 0.0f ? metal::exp(PM[base + c] - top) : 0.0f;
    lsum = fma(PL[base + c], e, lsum);
    for (int i = 0; i < DPL; i++) acc[i] = fma(PO[(base + c) * DH + lane * DPL + i], e, acc[i]);
  }
  for (int i = 0; i < DPL; i++) OUT[(size_t(r) * NQ + qh) * DH + lane * DPL + i] = bfloat(acc[i] / lsum);
  if (xst == 0) return;
  simdgroup_barrier(mem_flags::mem_device);
  gemma_row_sums(OUT + (size_t(r) * NQ + qh) * DH, DH, XS + qh * (DH / 64) * xst, xst, r, lane, 32);
}

// Sum of squares over T threads into `out` (p: a threadgroup array of T / 32 floats).
inline float gemma_reduce(float acc, threadgroup float* p, uint lane, uint sg) {
  const float s = simd_sum(acc);
  if (lane == 0) p[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float out = 0.0f;
  for (int q = 0; q < T / 32; q++) out += p[q];
  return out;
}

// gilby's attn_tail: hn = h + RMSNorm(o) w_attn, then RMSNorm(hn) times the pre-MLP norm (a dense layer's one input).
kernel void gemma_attn_tail(const device bfloat* H [[buffer(0)]], const device bfloat* O [[buffer(1)]],
                            const device bfloat* WA [[buffer(2)]], const device bfloat* W1 [[buffer(3)]],
                            constant float& eps [[buffer(4)]], device bfloat* HN [[buffer(5)]], device bfloat* N1 [[buffer(6)]],
                            device float* XS [[buffer(7)]], constant int& xst [[buffer(8)]],
                            uint t [[thread_position_in_threadgroup]], uint r [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr int PER = D / T;
  threadgroup float p1[T / 32], p2[T / 32];
  float ov[PER], hin[PER], wa[PER], w1[PER];
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    ov[i] = float(O[int(r) * D + c]); hin[i] = float(H[int(r) * D + c]);
    wa[i] = float(WA[c]); w1[i] = float(W1[c]);
  }
  float ss = 0.0f;
  for (int i = 0; i < PER; i++) ss = fma(ov[i], ov[i], ss);
  const float inv1 = metal::precise::rsqrt(gemma_reduce(ss, p1, lane, sg) / float(D) + eps);
  float hv[PER];
  float ss2 = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    const float a = float(bfloat(wa[i] * float(bfloat(ov[i] * inv1))));
    const bfloat hn = bfloat(hin[i] + a);
    HN[int(r) * D + c] = hn;
    hv[i] = float(hn);
    ss2 = fma(hv[i], hv[i], ss2);
  }
  const float inv2 = metal::precise::rsqrt(gemma_reduce(ss2, p2, lane, sg) / float(D) + eps);
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    N1[int(r) * D + c] = bfloat(w1[i] * float(bfloat(hv[i] * inv2)));
  }
  threadgroup_barrier(mem_flags::mem_device);
  gemma_row_sums(N1 + int(r) * D, D, XS, xst, int(r), t, T);
}

// gilby's dense_tail: hn = (h + RMSNorm(y) w_post) * scalar, and RMSNorm(hn) w_next for the next layer.
kernel void gemma_dense_tail(const device bfloat* H [[buffer(0)]], const device bfloat* Y [[buffer(1)]],
                             const device bfloat* WP [[buffer(2)]], const device bfloat* SC [[buffer(3)]],
                             const device bfloat* WN [[buffer(4)]], constant float& eps [[buffer(5)]],
                             device bfloat* HN [[buffer(6)]], device bfloat* NEXT [[buffer(7)]],
                             device float* XS [[buffer(8)]], constant int& xst [[buffer(9)]],
                             device bfloat* TAP [[buffer(10)]], constant int& tst [[buffer(11)]],
                             uint t [[thread_position_in_threadgroup]], uint r [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr int PER = D / T;
  threadgroup float p1[T / 32], p2[T / 32];
  float yv[PER], hin[PER], wp[PER], wn[PER];
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    yv[i] = float(Y[int(r) * D + c]); hin[i] = float(H[int(r) * D + c]);
    wp[i] = float(WP[c]); wn[i] = float(WN[c]);
  }
  const float sc = float(SC[0]);
  float ss1 = 0.0f;
  for (int i = 0; i < PER; i++) ss1 = fma(yv[i], yv[i], ss1);
  const float inv1 = metal::precise::rsqrt(gemma_reduce(ss1, p1, lane, sg) / float(D) + eps);
  float hv[PER];
  float ss2 = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    const float b = float(bfloat(wp[i] * float(bfloat(yv[i] * inv1))));
    const bfloat hs = bfloat(hin[i] + b);
    const bfloat hn = bfloat(float(hs) * sc);
    HN[int(r) * D + c] = hn;
    if (tst) TAP[size_t(r) * tst + c] = hn; // a drafter's tapped layer: its row at the tap's columns
    hv[i] = float(hn);
    ss2 = fma(hv[i], hv[i], ss2);
  }
  const float inv2 = metal::precise::rsqrt(gemma_reduce(ss2, p2, lane, sg) / float(D) + eps);
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    NEXT[int(r) * D + c] = bfloat(wn[i] * float(bfloat(hv[i] * inv2)));
  }
  threadgroup_barrier(mem_flags::mem_device);
  gemma_row_sums(NEXT + int(r) * D, D, XS, xst, int(r), t, T);
}

// mlx_lm's geglu as MLX's bf16 ops; IL: gate and up columns in interleaved 16-wide blocks, else two halves.
kernel void gemma_geglu(const device bfloat* GU [[buffer(0)]], device bfloat* ACT [[buffer(1)]],
                        constant uint& n [[buffer(2)]], device float* XS [[buffer(3)]], constant int& xst [[buffer(4)]],
                        constant int& IL [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]], uint2 tp [[thread_position_in_threadgroup]]) {
  const uint g = tg.x, r = tg.y, i = tp.x, c = g * 64 + i;
  const uint gi = IL ? 32 * (c / 16) + c % 16 : c, ui = IL ? gi + 16 : n + c;
  const bfloat x = GU[size_t(r) * 2 * n + gi], up = GU[size_t(r) * 2 * n + ui];
  const bfloat cube = bfloat(metal::pow(float(x), 3.0f));
  const bfloat inner = bfloat(float(x) + float(bfloat(float(bfloat(GELU_C1)) * float(cube))));
  const bfloat th = bfloat(metal::precise::tanh(float(bfloat(float(bfloat(GELU_C0)) * float(inner)))));
  const bfloat half_x = bfloat(0.5f * float(x));
  const bfloat gl = bfloat(float(half_x) * float(bfloat(1.0f + float(th))));
  const bfloat out = bfloat(float(gl) * float(up));
  ACT[size_t(r) * n + c] = out;
  if (xst == 0) return;
  threadgroup float vals[64];
  vals[i] = float(out);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (i != 0) return;
  float sum = 0.0f;
  for (int j = 0; j < 64; j++) sum += vals[j];
  XS[g * xst + r] = sum;
}

// mlx_lm's logit_softcap: tanh(x / cap) * cap as bf16 ops.
kernel void gemma_softcap(device bfloat* X [[buffer(0)]], constant float& cap [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
  X[i] = bfloat(float(bfloat(metal::precise::tanh(float(bfloat(float(X[i]) / cap))))) * cap);
}

// mlx_lm's ProportionalRoPE frequencies: base ** (arange(0, rotated, 2) / dims) in float32, as MLX's pow computes it.
kernel void gemma_rope_freqs(device float* F [[buffer(0)]], constant float& base [[buffer(1)]],
                             constant uint& dims [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  F[i] = metal::precise::pow(base, float(2 * i) / float(dims));
}

// A window's new keys and values [HK, R, DH] into a layer's cache [HK, cap, DH] at slot p % ring (0: p).
kernel void gemma_kv_write(const device bfloat* KN [[buffer(0)]], const device bfloat* VN [[buffer(1)]],
                           device bfloat* K [[buffer(2)]], device bfloat* V [[buffer(3)]],
                           const device int* POS [[buffer(4)]], constant int3& m [[buffer(5)]],
                           uint3 pos [[thread_position_in_grid]]) {
  const int d = int(pos.x), r = int(pos.y), h = int(pos.z), R = m.x, ring = m.y, cap = m.z;
  const int p = POS[r];
  const size_t dst = (size_t(h) * cap + (ring ? p % ring : p)) * DH + d;
  const size_t src = (size_t(h) * R + r) * DH + d;
  K[dst] = KN[src];
  V[dst] = VN[src];
}

// DFlash drafter rows (a: x stride, width, first position, ring or 0) into cache slots, position % ring or position.
kernel void gemma_draft_store(const device bfloat* X [[buffer(0)]], device bfloat* C [[buffer(1)]],
                              constant uint4& a [[buffer(2)]], uint2 pos [[thread_position_in_grid]]) {
  if (pos.x >= a.y) return;
  const uint p = a.z + pos.y, slot = a.w ? p % a.w : p;
  C[size_t(slot) * a.y + pos.x] = X[size_t(pos.y) * a.x + pos.x];
}

// tf_draft_swiglu's bf16 steps on GU rows of gate then up (pitch 2n) into ACT (pitch n), 64 threads a group of sums.
kernel void gemma_draft_swiglu(const device bfloat* GU [[buffer(0)]], device bfloat* ACT [[buffer(1)]],
                               constant uint& n [[buffer(2)]], device float* XS [[buffer(3)]], constant int& xst [[buffer(4)]],
                               uint2 tg [[threadgroup_position_in_grid]], uint2 tp [[thread_position_in_threadgroup]]) {
  const uint g = tg.x, r = tg.y, i = tp.x, c = g * 64 + i;
  const float gate = float(GU[size_t(r) * 2 * n + c]), up = float(GU[size_t(r) * 2 * n + n + c]);
  const bfloat exponential = bfloat(metal::exp(metal::abs(gate)));
  const bfloat denominator = bfloat(1.0f + float(exponential));
  const bfloat inverse = bfloat(1.0f / float(denominator));
  const bfloat sigmoid = gate < 0.0f ? inverse : bfloat(1.0f - float(inverse));
  const bfloat activation = bfloat(float(bfloat(gate)) * float(sigmoid));
  const bfloat out = bfloat(float(activation) * float(bfloat(up)));
  ACT[size_t(r) * n + c] = out;
  threadgroup float vals[64];
  vals[i] = float(out);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (i != 0) return;
  float sum = 0.0f;
  for (int j = 0; j < 64; j++) sum += vals[j];
  XS[g * xst + r] = sum;
}

// The drafter's residual: H += Y, each sum rounded to bf16.
kernel void gemma_draft_add(device bfloat* H [[buffer(0)]], const device bfloat* Y [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
  H[i] = bfloat(float(H[i]) + float(Y[i]));
}

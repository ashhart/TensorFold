// Flash Next prompt chunks: hyper-connection pieces in prefill_hc's arithmetic, router rows, top-k, the expert sort
// (count, offsets, slots), row gathers and scatters, and the activation. Compiled after the gate/up header (simd_topk_all).
inline float pf_bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
inline float pf_bsilu(float x) { return float(bfloat(x / (1.0f + metal::exp(-x)))); }
inline float pf_rinv(const device float* ssp, int r, int s, int nt, int streams, int dims, float eps) {
  float total = 0.0f;
  for (int j = 0; j < nt; j++) total += ssp[(r * nt + j) * streams + s];
  return metal::rsqrt(total / float(dims) + eps);
}

// Thread (e, r): bf16((h * rinv(stream)) * scale), the block's normed input.
[[kernel]] void pf_hc_normed(const device bfloat* HN [[buffer(0)]], const device float* SSP [[buffer(1)]],
    const device float* NW [[buffer(2)]], const device float* eps [[buffer(3)]], device bfloat* NORMED [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]) {
  constexpr int S = 4, D = 2560, W = S * D;
  const int e = int(pos.x), r = int(pos.y);
  const float rv = pf_rinv(SSP, r, e / D, D / 256, S, D, eps[0]);
  NORMED[size_t(r) * W + e] = bfloat((float(HN[size_t(r) * W + e]) * rv) * NW[e]);
}

// Thread (c, r): output c of the down + inject rows / S, then SiLU (c < LOW) or the gate 2 sigmoid.
[[kernel]] void pf_hc_act(const device bfloat* DN [[buffer(0)]], const constant int& nd [[buffer(1)]],
    device bfloat* ACT [[buffer(2)]], device bfloat* INJ [[buffer(3)]], uint2 pos [[thread_position_in_grid]]) {
  constexpr int S = 4, LOW = 320;
  const int c = int(pos.x), r = int(pos.y);
  if (c >= nd) return;
  const float v4 = float(bfloat(float(DN[size_t(r) * nd + c]) / float(S)));
  if (c < LOW) ACT[size_t(r) * LOW + c] = bfloat(pf_bsilu(v4));
  else INJ[size_t(r) * S + (c - LOW)] = bfloat(2.0f * pf_bsig(v4));
}

// Thread (d, r): the mean over streams of bf16(sigmoid(up) * normed).
[[kernel]] void pf_hc_mix(const device bfloat* UP [[buffer(0)]], const device bfloat* NORMED [[buffer(1)]],
    device bfloat* MIXED [[buffer(2)]], uint2 pos [[thread_position_in_grid]]) {
  constexpr int S = 4, D = 2560, W = S * D;
  const int d = int(pos.x), r = int(pos.y);
  float total = 0.0f;
  for (int s = 0; s < S; s++) {
    const size_t e = size_t(r) * W + s * D + d;
    total += float(bfloat(pf_bsig(float(UP[e])) * float(NORMED[e])));
  }
  MIXED[size_t(r) * D + d] = bfloat(total / float(S));
}

// Router logits in fp32: simdgroup s of threadgroup (j-block, r) sums output 8 j-block + s over K, lanes strided.
[[kernel]] void pf_router(const device bfloat* X [[buffer(0)]], const device bfloat* RW [[buffer(1)]],
    device float* LG [[buffer(2)]], uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int K = 2560, NL = 513;
  const int j = int(tg.x) * 8 + int(sgi), r = int(tg.y);
  if (j >= NL) return;
  float acc = 0.0f;
  for (int k = int(lane); k < K; k += 32) acc += float(X[size_t(r) * K + k]) * float(RW[size_t(j) * K + k]);
  acc = simd_sum(acc);
  if (lane == 0) LG[size_t(r) * NL + j] = acc;
}

// Row r's ten experts (the decode's top-k rounds) and their softmax weights; counts each expert's pairs.
[[kernel]] void pf_route(const device float* LG [[buffer(0)]], device uint* PICK [[buffer(1)]],
    device float* WTS [[buffer(2)]], device atomic_uint* CNT [[buffer(3)]], uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int TOPK = 10, NE = 512, NL = 513;
  const int r = int(tg.y);
  int ids[TOPK];
  float picked[TOPK];
  simd_topk_all<NE, TOPK>(LG + size_t(r) * NL, lane, ids, picked);
  if (lane == 0) {
    float total = 0.0f;
    float ex[TOPK];
    for (int kk = 0; kk < TOPK; kk++) { ex[kk] = metal::exp(picked[kk] - picked[0]); total += ex[kk]; }
    for (int kk = 0; kk < TOPK; kk++) {
      WTS[r * TOPK + kk] = float(bfloat(ex[kk] / total));
      PICK[r * TOPK + kk] = uint(ids[kk]);
      atomic_fetch_add_explicit(CNT + ids[kk], 1u, memory_order_relaxed);
    }
  }
}

// Each expert's first sorted slot (exclusive scan of the counts); the counts go back to zero for the next layer.
[[kernel]] void pf_offsets(device atomic_uint* CNT [[buffer(0)]], device int* OFF [[buffer(1)]],
    device atomic_uint* CUR [[buffer(2)]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup int counts[512];
  counts[t] = int(atomic_load_explicit(CNT + t, memory_order_relaxed));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int before = 0;
  for (uint i = 0; i < t; i++) before += counts[i];
  OFF[t] = before;
  atomic_store_explicit(CUR + t, uint(before), memory_order_relaxed);
  atomic_store_explicit(CNT + t, 0u, memory_order_relaxed);
}

// Pair p (row * 10 + k) into a slot of its expert's range (order within an expert does not change any row's sums).
[[kernel]] void pf_sort(const device uint* PICK [[buffer(0)]], device atomic_uint* CUR [[buffer(1)]],
    device int* ROW_OF [[buffer(2)]], const constant int& pairs [[buffer(3)]], uint p [[thread_position_in_grid]]) {
  if (int(p) >= pairs) return;
  const uint slot = atomic_fetch_add_explicit(CUR + PICK[p], 1u, memory_order_relaxed);
  ROW_OF[slot] = int(p);
}

// Slot s's input row (the pair's token row), 8 values a thread.
[[kernel]] void pf_gather_rows(const device bfloat* X [[buffer(0)]], const device int* ROW_OF [[buffer(1)]],
    device bfloat* XS [[buffer(2)]], uint2 pos [[thread_position_in_grid]]) {
  constexpr int K = 2560, TOPK = 10;
  const int c = int(pos.x) * 8, slot = int(pos.y);
  const int row = ROW_OF[slot] / TOPK;
  for (int i = 0; i < 8; i++) XS[size_t(slot) * K + c + i] = X[size_t(row) * K + c + i];
}

// The expert activation as the decode rounds it: bf16(bsilu(gate) * up).
[[kernel]] void pf_act(const device bfloat* G [[buffer(0)]], const device bfloat* U [[buffer(1)]],
    device bfloat* A [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  A[i] = bfloat(pf_bsilu(float(G[i])) * float(U[i]));
}

// Slot s's down output into its pair's place in the combine's layout [rows, 11, D], 8 values a thread.
[[kernel]] void pf_scatter_y(const device bfloat* DS [[buffer(0)]], const device int* ROW_OF [[buffer(1)]],
    device bfloat* YD [[buffer(2)]], uint2 pos [[thread_position_in_grid]]) {
  constexpr int D = 2560, TOPK = 10, SLOTS = TOPK + 1;
  const int c = int(pos.x) * 8, slot = int(pos.y);
  const int p = ROW_OF[slot], row = p / TOPK, k = p % TOPK;
  for (int i = 0; i < 8; i++) YD[(size_t(row) * SLOTS + k) * D + c + i] = DS[size_t(slot) * D + c + i];
}

[[kernel]] void pf_copy(const device uint* S [[buffer(0)]], device uint* T [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  T[i] = S[i];
}

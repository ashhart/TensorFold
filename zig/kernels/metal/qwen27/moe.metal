// Routed experts: the top-k of a row's router logits and their weights, picks sorted by expert, the weighted sum.
#include <metal_stdlib>
using namespace metal;
// E experts, TOPK picks a row, LS logit columns a row (E, the shared gate, padding), D hidden width.
Q27_SPECIALIZE

constant constexpr uint PER = (E + 31) / 32;
constant constexpr uint WS = TOPK + 1; // weights a row: the picks', then the shared expert's

// A simdgroup a row: top-k (ties to the lower id, NaN and -Inf as -FLT_MAX), bf16 weights, the shared gate's sigmoid.
[[kernel]] void q27_moe_select(const device float* L [[buffer(0)]], device int* PICK [[buffer(1)]], device float* WTS [[buffer(2)]],
                               uint r [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const device float* row = L + size_t(r) * LS;
  float c[PER];
  for (uint j = 0; j < PER; j++) {
    const uint id = j * 32 + lane;
    const float v = id < E ? row[id] : -INFINITY;
    c[j] = isnan(v) || (id < E && v == -INFINITY) ? -FLT_MAX : v; // -Inf marks a taken expert alone
  }
  float picked[TOPK];
  for (uint k = 0; k < TOPK; k++) {
    float best = -INFINITY;
    uint bid = E;
    for (uint j = 0; j < PER; j++) {
      const uint id = j * 32 + lane;
      if (id < E && (c[j] > best || (c[j] == best && id < bid))) { best = c[j]; bid = id; }
    }
    for (ushort off = 16; off > 0; off /= 2) {
      const float ob = simd_shuffle_xor(best, off);
      const uint oi = simd_shuffle_xor(bid, off);
      if (ob > best || (ob == best && oi < bid)) { best = ob; bid = oi; }
    }
    picked[k] = best;
    if (lane == bid % 32) c[bid / 32] = -INFINITY;
    if (lane == 0) PICK[r * TOPK + k] = int(bid);
  }
  if (lane != 0) return;
  float ex[TOPK];
  float total = 0.0f;
  for (uint k = 0; k < TOPK; k++) {
    ex[k] = precise::exp(picked[k] - picked[0]);
    total += ex[k];
  }
  for (uint k = 0; k < TOPK; k++) WTS[r * WS + k] = float(bfloat(precise::divide(ex[k], total)));
  const float g = float(bfloat(row[E]));
  const float y = precise::divide(1.0f, 1.0f + precise::exp(abs(g)));
  WTS[r * WS + TOPK] = float(bfloat(g < 0.0f ? y : 1.0f - y));
}

// Picks by expert in slot order: STARTS [E + 1], POS (slot -> sorted), ROWS (sorted -> row), ONE (shared {0, rows}).
constant constexpr uint SORT_CHUNKS = 4; // thread (e, c) counts expert e's picks in chunk c, then places them
[[kernel]] void q27_moe_sort(const device int* ids [[buffer(0)]], constant uint& slots [[buffer(1)]], device uint* starts [[buffer(2)]],
                             device uint* pos [[buffer(3)]], device uint* rows [[buffer(4)]], device uint* one [[buffer(5)]],
                             uint t [[thread_position_in_threadgroup]]) {
  threadgroup uint counts[E * SORT_CHUNKS];
  threadgroup uint totals[E];
  const uint e = t / SORT_CHUNKS, c = t % SORT_CHUNKS;
  const uint span = (slots + SORT_CHUNKS - 1) / SORT_CHUNKS, lo = min(c * span, slots), hi = min(lo + span, slots);
  uint n = 0;
  for (uint j = lo; j < hi; ++j) n += uint(ids[j]) == e ? 1 : 0;
  counts[t] = n;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (c == 0) {
    uint total = 0;
    for (uint i = 0; i < SORT_CHUNKS; ++i) total += counts[e * SORT_CHUNKS + i];
    totals[e] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint at = 0;
  for (uint i = 0; i < e; ++i) at += totals[i];
  if (c == 0) starts[e] = at;
  if (t == E * SORT_CHUNKS - 1) {
    starts[E] = at + totals[e];
    one[0] = 0;
    one[1] = slots / TOPK;
  }
  for (uint i = 0; i < c; ++i) at += counts[e * SORT_CHUNKS + i];
  for (uint j = lo; j < hi; ++j)
    if (uint(ids[j]) == e) {
      rows[at] = j / TOPK;
      pos[j] = at;
      ++at;
    }
}

// OUT [rows, D] = bf16 of one fp32 fma chain over the picks in order (Y at POS when SORTED), then the shared expert.
[[kernel]] void q27_moe_combine(const device bfloat* Y [[buffer(0)]], const device bfloat* YS [[buffer(1)]], const device float* WTS [[buffer(2)]],
                                const device uint* POS [[buffer(3)]], constant uint& sorted [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                                uint2 at [[thread_position_in_grid]]) {
  if (at.x >= D) return;
  const uint r = at.y;
  float acc = 0.0f;
  for (uint k = 0; k < TOPK; k++) {
    const uint slot = r * TOPK + k;
    const uint y = sorted != 0 ? POS[slot] : slot;
    acc = fma(WTS[r * WS + k], float(Y[size_t(y) * D + at.x]), acc);
  }
  acc = fma(WTS[r * WS + TOPK], float(YS[size_t(r) * D + at.x]), acc);
  OUT[size_t(r) * D + at.x] = bfloat(acc);
}

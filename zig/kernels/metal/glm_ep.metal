// GLM-5.3-Flash speed-up mode's exchange kernels (families/glm/ep.zig): expert outputs, row sums and TP partials.
#include <metal_stdlib>
using namespace metal;
constant constexpr int TOPK = 8, MAXR = 16, WORDS = 4096 / 2;
// One threadgroup of 128: the route's experts held here (local ids, members), the picks each Mac computes, the count
kernel void ep_localize(const device int* PICK [[buffer(0)]], const device int* UIDS [[buffer(1)]],
    const device int* UMEM [[buffer(2)]], const device int* UCOUNT [[buffer(3)]], constant int4& arg [[buffer(4)]],
    device int* LIDS [[buffer(5)]], device int* LMEM [[buffer(6)]], device int* LCOUNT [[buffer(7)]],
    device int* MINE [[buffer(8)]], device int* THEIRS [[buffer(9)]], device int* COUNTS [[buffer(10)]],
    device atomic_uint* CNT [[buffer(11)]], uint t [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint g [[simdgroup_index_in_threadgroup]]) {
  const int rows = arg.x, lo = arg.y, hi = arg.z, i = int(t);
  const int e = i < UCOUNT[0] ? UIDS[i] : -1;
  const int held = e >= lo && e < hi ? 1 : 0;
  const int pe = i < rows * TOPK ? PICK[i] : -1; // pick i's expert
  const int mine = pe >= lo && pe < hi ? 1 : 0;
  const int theirs = pe >= 0 && mine == 0 ? 1 : 0;
  const int b0 = simd_prefix_exclusive_sum(held), b1 = simd_prefix_exclusive_sum(mine), b2 = simd_prefix_exclusive_sum(theirs);
  threadgroup int part[3][4];
  if (lane == 31) { part[0][g] = b0 + held; part[1][g] = b1 + mine; part[2][g] = b2 + theirs; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int o0 = 0, o1 = 0, o2 = 0, n0 = 0, n1 = 0, n2 = 0;
  for (uint q = 0; q < 4; q++) {
    if (q < g) { o0 += part[0][q]; o1 += part[1][q]; o2 += part[2][q]; }
    n0 += part[0][q]; n1 += part[1][q]; n2 += part[2][q];
  }
  if (held) { LIDS[o0 + b0] = e - lo; for (int j = 0; j < MAXR; j++) LMEM[(o0 + b0) * MAXR + j] = UMEM[i * MAXR + j]; }
  if (mine) MINE[o1 + b1] = i;
  if (theirs) THEIRS[o2 + b2] = i;
  if (t == 0) { LCOUNT[0] = n0; COUNTS[0] = n1; COUNTS[1] = n2; atomic_store_explicit(CNT, uint(n1), memory_order_relaxed); }
}
// Exchange x writes its send slot once the peer's x - 1 landed (posted after our x - 2 reached it); give-ups counted
inline bool ep_slot_free(device atomic_uint* flag, uint x, device atomic_uint* gave_up, uint t, threadgroup uint* ok) {
  if (t == 0) {
    uint polls = 0;
    *ok = 1u;
    while (int(atomic_load_explicit(flag, memory_order_relaxed) - (x - 1u)) < 0) {
      if (++polls > 400000000u) { atomic_fetch_add_explicit(gave_up, 1u, memory_order_relaxed); *ok = 0u; break; }
    }
    if (atomic_load_explicit(gave_up, memory_order_relaxed) != 0u) *ok = 0u;
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  return *ok != 0u;
}
// This Mac's picks' outputs (Y [picks, 4096] bf16) packed in MINE order for the peer; a threadgroup a pick
kernel void ep_pack(const device uint* Y [[buffer(0)]], const device int* MINE [[buffer(1)]],
    const device int* COUNTS [[buffer(2)]], device uint* OUT [[buffer(3)]], device atomic_uint* flag [[buffer(4)]],
    constant uint& x [[buffer(5)]], device atomic_uint* gave_up [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
    uint3 tpos [[thread_position_in_threadgroup]]) {
  const int i = int(tg.y);
  const uint t = tpos.x;
  threadgroup uint ok;
  if (!ep_slot_free(flag, x, gave_up, t, &ok)) return;
  if (i >= COUNTS[0]) return;
  const device uint* src = Y + size_t(MINE[i]) * WORDS;
  device uint* dst = OUT + size_t(i) * WORDS;
  for (uint j = t; j < uint(WORDS); j += 256) dst[j] = src[j];
}
// By rows: this Mac's routed sum a row (slot partials weighted in slot order) into the send slot; two entries a row
#pragma clang fp contract(off)
inline float ep_mul_add(float acc, float a, float b) { return a * b + acc; }
#pragma clang fp contract(on)
kernel void ep_rcombine(const device float* YP [[buffer(0)]], const device float* WTS [[buffer(1)]],
    constant int& rows [[buffer(2)]], device float* OUT [[buffer(3)]], device atomic_uint* CNT [[buffer(4)]],
    device atomic_uint* flag [[buffer(5)]], constant uint& x [[buffer(6)]], device atomic_uint* gave_up [[buffer(7)]],
    uint gid [[thread_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup uint ok;
  if (!ep_slot_free(flag, x, gave_up, t, &ok)) return;
  const int r = int(gid) / 4096, d = int(gid) % 4096;
  if (gid == 0) atomic_store_explicit(CNT, uint(rows * 2), memory_order_relaxed);
  if (r >= rows) return;
  const device float* y = YP + size_t(r) * TOPK * 4096 + d;
  float acc = WTS[r * TOPK] * y[0];
  for (int k = 1; k < TOPK; k++) acc = ep_mul_add(acc, WTS[r * TOPK + k], y[size_t(k) * 4096]);
  OUT[size_t(r) * 4096 + d] = acc;
}
// By rows: once the peer's sums land, each branch is both Macs' sums in rank order, rounded, plus the shared expert
kernel void ep_rfinal(device atomic_uint* flag [[buffer(0)]], constant uint& x [[buffer(1)]],
    device atomic_uint* gave_up [[buffer(2)]], const device float* MINE [[buffer(3)]], device atomic_uint* PEER [[buffer(4)]],
    constant uint& rank [[buffer(5)]], const device bfloat* YS [[buffer(6)]], device bfloat* OUT [[buffer(7)]],
    constant int& rows [[buffer(8)]], uint3 tg [[threadgroup_position_in_grid]], uint3 tpos [[thread_position_in_threadgroup]]) {
  const uint t = tpos.x;
  if (t == 0) {
    uint polls = 0;
    while (int(atomic_load_explicit(flag, memory_order_relaxed) - x) < 0) {
      if (++polls > 400000000u) { atomic_fetch_add_explicit(gave_up, 1u, memory_order_relaxed); break; }
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  const int i = int(tg.x) * 256 + int(t);
  if (i >= rows * 4096) return;
  const float mine = MINE[i], peer = as_type<float>(atomic_load_explicit(&PEER[i], memory_order_relaxed));
  const float total = rank == 0u ? mine + peer : peer + mine;
  OUT[i] = bfloat(total) + YS[i];
}
// TP: this Mac's fp32 partial rows into the send slot once it is free, and the entries the host sends (two a row)
kernel void ep_rsend(const device float* PART [[buffer(0)]], constant int& rows [[buffer(1)]],
    device float* OUT [[buffer(2)]], device atomic_uint* CNT [[buffer(3)]], device atomic_uint* flag [[buffer(4)]],
    constant uint& x [[buffer(5)]], device atomic_uint* gave_up [[buffer(6)]],
    uint gid [[thread_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup uint ok;
  if (!ep_slot_free(flag, x, gave_up, t, &ok)) return;
  if (gid == 0) atomic_store_explicit(CNT, uint(rows * 2), memory_order_relaxed);
  if (int(gid) < rows * 4096) OUT[gid] = PART[gid];
}
// TP: wait for the peer's partials of exchange x; each value is both Macs' partials added in rank order, rounded
kernel void ep_rsum(device atomic_uint* flag [[buffer(0)]], constant uint& x [[buffer(1)]],
    device atomic_uint* gave_up [[buffer(2)]], const device float* MINE [[buffer(3)]], device atomic_uint* PEER [[buffer(4)]],
    constant uint& rank [[buffer(5)]], device bfloat* OUT [[buffer(6)]], constant int& rows [[buffer(7)]],
    uint3 tg [[threadgroup_position_in_grid]], uint3 tpos [[thread_position_in_threadgroup]]) {
  const uint t = tpos.x;
  if (t == 0) {
    uint polls = 0;
    while (int(atomic_load_explicit(flag, memory_order_relaxed) - x) < 0) {
      if (++polls > 400000000u) { atomic_fetch_add_explicit(gave_up, 1u, memory_order_relaxed); break; }
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  const int i = int(tg.x) * 256 + int(t);
  if (i >= rows * 4096) return;
  const float mine = MINE[i], peer = as_type<float>(atomic_load_explicit(&PEER[i], memory_order_relaxed));
  OUT[i] = bfloat(rank == 0u ? mine + peer : peer + mine);
}
// Exchange x packed: the host sends it once the GPU gets here
kernel void ep_post(device atomic_uint* posted [[buffer(0)]], constant uint& x [[buffer(1)]],
    device atomic_uint* gave_up [[buffer(2)]]) {
  if (atomic_load_explicit(gave_up, memory_order_relaxed) == 0u) atomic_store_explicit(posted, x, memory_order_relaxed);
}
// Once the peer's exchange x lands (its flag follows its bytes), its entries (atomic reads) into their picks' rows of Y
kernel void ep_unpack(device atomic_uint* flag [[buffer(0)]], constant uint& x [[buffer(1)]],
    device atomic_uint* gave_up [[buffer(2)]], device atomic_uint* IN [[buffer(3)]], const device int* THEIRS [[buffer(4)]],
    const device int* COUNTS [[buffer(5)]], device uint* Y [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
    uint3 tpos [[thread_position_in_threadgroup]]) {
  const uint t = tpos.x;
  if (t == 0) {
    uint polls = 0;
    while (int(atomic_load_explicit(flag, memory_order_relaxed) - x) < 0) {
      if (++polls > 400000000u) { atomic_fetch_add_explicit(gave_up, 1u, memory_order_relaxed); break; }
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  const int i = int(tg.y);
  if (i >= COUNTS[1]) return;
  device atomic_uint* src = IN + size_t(i) * WORDS;
  device uint* dst = Y + size_t(THEIRS[i]) * WORDS;
  for (uint j = t; j < uint(WORDS); j += 256) dst[j] = atomic_load_explicit(&src[j], memory_order_relaxed);
}
// TP: each row's best logit of this Mac's vocabulary rows [lo, lo + n) and its token id to send (n = 0: none)
kernel void ep_amax_send(const device bfloat* LOGITS [[buffer(0)]], const device uint* PICKS [[buffer(1)]],
    constant uint4& arg [[buffer(2)]], device uint* OUT [[buffer(3)]], device atomic_uint* CNT [[buffer(4)]],
    device atomic_uint* flag [[buffer(5)]], constant uint& x [[buffer(6)]], device atomic_uint* gave_up [[buffer(7)]],
    uint t [[thread_index_in_threadgroup]]) {
  threadgroup uint ok;
  if (!ep_slot_free(flag, x, gave_up, t, &ok)) return;
  const uint rows = arg.x, n = arg.y, lo = arg.z;
  if (t == 0) atomic_store_explicit(CNT, 1u, memory_order_relaxed);
  if (t >= rows) return;
  const uint p = n > 0 ? PICKS[t] : 0xffffffffu;
  const bool has = p < n;
  OUT[2 * t] = as_type<uint>(has ? float(LOGITS[size_t(t) * n + p]) : -INFINITY);
  OUT[2 * t + 1] = has ? lo + p : 0xffffffffu;
}
// TP: once the peer's candidates land, each row's pick is the larger logit, the lower id on a tie (one Mac's rule)
kernel void ep_amax_merge(device atomic_uint* flag [[buffer(0)]], constant uint& x [[buffer(1)]],
    device atomic_uint* gave_up [[buffer(2)]], const device uint* MINE [[buffer(3)]], device atomic_uint* PEER [[buffer(4)]],
    constant uint& rows [[buffer(5)]], device uint* PICKS [[buffer(6)]], uint t [[thread_index_in_threadgroup]]) {
  if (t == 0) {
    uint polls = 0;
    while (int(atomic_load_explicit(flag, memory_order_relaxed) - x) < 0) {
      if (++polls > 400000000u) { atomic_fetch_add_explicit(gave_up, 1u, memory_order_relaxed); break; }
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (t >= rows) return;
  const float a = as_type<float>(MINE[2 * t]), b = as_type<float>(atomic_load_explicit(&PEER[2 * t], memory_order_relaxed));
  const uint ia = MINE[2 * t + 1], ib = atomic_load_explicit(&PEER[2 * t + 1], memory_order_relaxed);
  PICKS[t] = (b > a || (b == a && ib < ia)) ? ib : ia;
}

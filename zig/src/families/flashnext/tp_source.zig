//! The existing pair kernels, separate from host ownership and memory accounting.
pub const source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\// this sequence and the window's rows (seq << 5 | rows), after every earlier dispatch of the serial encoder
    \\kernel void tp_post(device atomic_uint* sync [[buffer(0)]], constant uint& seq [[buffer(1)]],
    \\                    const constant int* rows [[buffer(2)]]) {
    \\  atomic_store_explicit(&sync[1024], (seq << 5) | uint(rows[0]), memory_order_relaxed);
    \\}
    \\// post sx.x (every threadgroup stores the same word: the host sends this rank's part), then a threadgroup's first
    \\// thread polls until the peer's message for exchange sx.y has landed (its flag, stored after its bytes)
    \\inline void tp_post_poll(device atomic_uint* sync, device atomic_uint* flag, uint2 sx, int rows, bool poll, uint t) {
    \\  if (t != 0) return;
    \\  atomic_store_explicit(&sync[1024], (sx.x << 5) | uint(rows), memory_order_relaxed);
    \\  if (!poll) return;
    \\  uint polls = 0;
    \\  while (int(atomic_load_explicit(flag, memory_order_relaxed) - sx.y) < 0) {
    \\    if (++polls > 400000000u) { atomic_fetch_add_explicit(&sync[2048], 1u, memory_order_relaxed); return; }
    \\  }
    \\}
    \\// a row's argmax over this rank's vocab columns [lo, lo + n) of the full-width logits: (value, index), the
    \\// larger value and on a tie the lower index, as fz_argmax
    \\kernel void tp_argmax_part(device const bfloat* logits [[buffer(0)]], device uint* out [[buffer(1)]],
    \\    constant uint4& dims [[buffer(2)]], uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    \\    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    \\  device const bfloat* x = logits + row * dims.x;
    \\  float best = -INFINITY; uint at = 0xffffffffu;
    \\  for (uint i = dims.y + t; i < dims.y + dims.z; i += 1024) { const float v = float(x[i]); if (v > best) { best = v; at = i; } }
    \\  for (ushort o = 16; o > 0; o >>= 1) {
    \\    const float ob = simd_shuffle_xor(best, o); const uint oa = simd_shuffle_xor(at, o);
    \\    if (ob > best || (ob == best && oa < at)) { best = ob; at = oa; }
    \\  }
    \\  threadgroup float vb[32]; threadgroup uint va[32];
    \\  if (lane == 0) { vb[sg] = best; va[sg] = at; }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (sg != 0) return;
    \\  best = vb[lane]; at = va[lane];
    \\  for (ushort o = 16; o > 0; o >>= 1) {
    \\    const float ob = simd_shuffle_xor(best, o); const uint oa = simd_shuffle_xor(at, o);
    \\    if (ob > best || (ob == best && oa < at)) { best = ob; at = oa; }
    \\  }
    \\  if (lane == 0) { out[2 * row] = as_type<uint>(best); out[2 * row + 1] = at; }
    \\}
    \\// post, poll until the peer's half lands, then each row's pick from both halves: the larger value, the lower index on a tie
    \\kernel void tp_pick(device atomic_uint* sync [[buffer(0)]], constant uint2& sx [[buffer(1)]],
    \\    const constant int* rows [[buffer(2)]], const device uint* mine [[buffer(3)]],
    \\    device atomic_uint* peer [[buffer(4)]], device uint* picks [[buffer(5)]], device atomic_uint* flag [[buffer(6)]]) {
    \\  tp_post_poll(sync, flag, sx, rows[0], true, 0);
    \\  for (int r = 0; r < rows[0]; r++) {
    \\    const float vm = as_type<float>(mine[2 * r]);
    \\    const uint am = mine[2 * r + 1];
    \\    const float vp = as_type<float>(atomic_load_explicit(&peer[2 * r], memory_order_relaxed));
    \\    const uint ap = atomic_load_explicit(&peer[2 * r + 1], memory_order_relaxed);
    \\    picks[r] = (vp > vm || (vp == vm && ap < am)) ? ap : am;
    \\  }
    \\}
    \\// the MTP head's merged draft pick (an index into its draft list) as a token id
    \\kernel void tp_ids(const device uint* pick [[buffer(0)]], const device uint* ids [[buffer(1)]],
    \\                   device uint* out [[buffer(2)]]) {
    \\  out[0] = ids[pick[0]];
    \\}
    \\// one thread polls until the 32-bit word reaches `value`; a give-up is counted, never silent
    \\kernel void tp_wait(device atomic_uint* word [[buffer(0)]], constant uint& value [[buffer(1)]],
    \\                    device atomic_uint* sync [[buffer(2)]]) {
    \\  uint polls = 0;
    \\  while (int(atomic_load_explicit(word, memory_order_relaxed) - value) < 0) {
    \\    if (++polls > 400000000u) { atomic_fetch_add_explicit(&sync[2048], 1u, memory_order_relaxed); return; }
    \\  }
    \\}
    \\// a stream update of row r at dims j * 256 + t from its branch, and each stream's partial sum of squares over those
    \\// dims: q4_hc_norm_plain's and q4_hc_norm_grouped's arithmetic
    \\inline void tp_update(const device bfloat* H, const device bfloat* INJ, float branch, device bfloat* HN, device float* SSP,
    \\    int j, int r, uint t, uint g, uint lane, threadgroup float (*part)[4]) {
    \\  constexpr int S = 4, D = 2560, W = S * D, NT = D / 256;
    \\  const int d = j * 256 + int(t);
    \\  float ss[S];
    \\  for (int s = 0; s < S; s++) {
    \\    const int e = s * D + d;
    \\    float hv = float(H[r * W + e]);
    \\    hv = float(bfloat(hv + float(bfloat(branch * float(INJ[r * S + s])))));
    \\    HN[r * W + e] = bfloat(hv);
    \\    ss[s] = simd_sum(hv * hv);
    \\  }
    \\  if (lane == 0) for (int s = 0; s < S; s++) part[g][s] = ss[s];
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (t < S) {
    \\    float total = 0.0f;
    \\    for (int k = 0; k < 8; k++) total += part[k][t];
    \\    SSP[(r * NT + j) * S + t] = total;
    \\  }
    \\}
    \\// post, wait for the peer's partial, then q4_hc_norm_plain's update with the branch from the two ranks' fp32
    \\// partials added in rank order and rounded once; the peer's words read as atomics (written by the host mid-buffer)
    \\kernel void tp_plain(device atomic_uint* sync [[buffer(0)]], constant uint2& sx [[buffer(1)]],
    \\    const constant int* rows [[buffer(2)]], const device float* MINE [[buffer(3)]], device atomic_uint* PEER [[buffer(4)]],
    \\    constant uint& rank [[buffer(5)]], const device bfloat* H [[buffer(6)]], const device bfloat* INJ [[buffer(7)]],
    \\    device bfloat* HN [[buffer(8)]], device float* SSP [[buffer(9)]], device atomic_uint* flag [[buffer(10)]],
    \\    uint g [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint3 tpos [[thread_position_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
    \\  const int j = int(tg.x), r = int(tg.y), i = r * 2560 + j * 256 + int(tpos.x);
    \\  tp_post_poll(sync, flag, sx, rows[0], true, tpos.x);
    \\  threadgroup_barrier(mem_flags::mem_device);
    \\  const float mine = MINE[i], peer = as_type<float>(atomic_load_explicit(&PEER[i], memory_order_relaxed));
    \\  threadgroup float part[8][4];
    \\  tp_update(H, INJ, float(bfloat(rank == 0u ? mine + peer : peer + mine)), HN, SSP, j, r, tpos.x, g, lane, part);
    \\}
    \\// this rank's rows [own.x, own.y) of the MoE branch, as q4_hc_norm_grouped makes it on one Mac (the routed slots
    \\// summed in slot order, fp32; the gated shared expert; one bf16 rounding each): what the peer's streams take
    \\inline float tp_bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
    \\kernel void tp_branch(const device bfloat* Y [[buffer(0)]], const device float* WTS [[buffer(1)]],
    \\    const device float* LG [[buffer(2)]], device bfloat* OUT [[buffer(3)]], constant uint2& own [[buffer(4)]],
    \\    uint2 pos [[thread_position_in_grid]]) {
    \\  constexpr int D = 2560, TOPK = 10, NL = 513;
    \\  const int d = int(pos.x), r = int(own.x) + int(pos.y);
    \\  if (r >= int(own.y)) return;
    \\  float routed = 0.0f;
    \\  for (int k = 0; k < TOPK; k++) routed = fma(float(Y[(r * (TOPK + 1) + k) * D + d]), WTS[r * TOPK + k], routed);
    \\  const float shared = float(bfloat(float(Y[(r * (TOPK + 1) + TOPK) * D + d]) * tp_bsig(float(bfloat(LG[r * NL + NL - 1])))));
    \\  OUT[(r - int(own.x)) * D + d] = bfloat(float(bfloat(routed)) + shared);
    \\}
    \\// post, then q4_hc_norm_grouped's stream update with each row's branch from the rank that made it (rows
    \\// [split.x, split.y) this rank's, in MINE, at once; the rest the peer's, from split.z, in THEIRS, once landed)
    \\kernel void tp_merge(device atomic_uint* sync [[buffer(0)]], constant uint2& sx [[buffer(1)]],
    \\    const constant int* rows [[buffer(2)]], const device bfloat* H [[buffer(3)]], const device bfloat* INJ [[buffer(4)]],
    \\    const device bfloat* MINE [[buffer(5)]], device atomic_uint* THEIRS [[buffer(6)]], constant uint4& split [[buffer(7)]],
    \\    device bfloat* HN [[buffer(8)]], device float* SSP [[buffer(9)]], device atomic_uint* flag [[buffer(10)]],
    \\    uint g [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint3 tpos [[thread_position_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
    \\  const int j = int(tg.x), r = int(tg.y), d = j * 256 + int(tpos.x);
    \\  const bool mine = uint(r) >= split.x && uint(r) < split.y;
    \\  tp_post_poll(sync, flag, sx, rows[0], !mine, tpos.x);
    \\  threadgroup_barrier(mem_flags::mem_device);
    \\  float branch;
    \\  if (mine) branch = float(MINE[(r - int(split.x)) * 2560 + d]);
    \\  else {
    \\    const uint at = uint(r - int(split.z)) * 2560u + uint(d);
    \\    branch = float(as_type<bfloat2>(atomic_load_explicit(&THEIRS[at >> 1], memory_order_relaxed))[at & 1]);
    \\  }
    \\  threadgroup float part[8][4];
    \\  tp_update(H, INJ, branch, HN, SSP, j, r, tpos.x, g, lane, part);
    \\}
;

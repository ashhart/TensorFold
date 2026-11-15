//! Pair kernels guard exchanged payloads with host-published epochs.
pub const source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\// Each row's argmax over this rank's vocabulary columns; ties choose the lower index.
    \\kernel void tp_argmax_part(device const bfloat* logits [[buffer(0)]], device uint* out [[buffer(1)]],
    \\    constant uint4& dims [[buffer(2)]], const device ulong* previous [[buffer(3)]],
    \\    constant ulong& expected [[buffer(4)]], uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    \\    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    \\  if (previous[0] != expected) return;
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
    \\// After the host arrival event, each row's pick from both halves: the larger value, the lower index on a tie
    \\kernel void tp_pick(const device ulong* ready [[buffer(0)]], constant ulong& expected [[buffer(1)]],
    \\    const constant int* rows [[buffer(2)]], const device uint* mine [[buffer(3)]],
    \\    const device uint* peer [[buffer(4)]], device uint* picks [[buffer(5)]]) {
    \\  if (ready[0] != expected) return;
    \\  for (int r = 0; r < rows[0]; r++) {
    \\    const float vm = as_type<float>(mine[2 * r]);
    \\    const uint am = mine[2 * r + 1];
    \\    const float vp = as_type<float>(peer[2 * r]);
    \\    const uint ap = peer[2 * r + 1];
    \\    picks[r] = (vp > vm || (vp == vm && ap < am)) ? ap : am;
    \\  }
    \\}
    \\// the MTP head's merged draft pick (an index into its draft list) as a token id
    \\kernel void tp_ids(const device uint* pick [[buffer(0)]], const device uint* ids [[buffer(1)]],
    \\                   device uint* out [[buffer(2)]], const device ulong* ready [[buffer(3)]], constant ulong& expected [[buffer(4)]]) {
    \\  if (ready[0] != expected) return;
    \\  out[0] = ids[pick[0]];
    \\}
    \\// Copy only a successfully published epoch, for prefill receives and outgoing projection partials.
    \\kernel void tp_copy(const device ulong* ready [[buffer(0)]], constant ulong& expected [[buffer(1)]],
    \\    const device uint* src [[buffer(2)]], device uint* dst [[buffer(3)]], constant uint& words [[buffer(4)]],
    \\    uint i [[thread_position_in_grid]]) {
    \\  if (ready[0] != expected || i >= words) return;
    \\  dst[i] = src[i];
    \\}
    \\// Update streams and partial sums of squares with q4_hc_norm_plain/grouped's arithmetic.
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
    \\// After host arrival, add fp32 partials in rank order and round the branch once for the stream update.
    \\kernel void tp_plain(const device ulong* ready [[buffer(0)]], constant ulong& expected [[buffer(1)]],
    \\    const constant int* rows [[buffer(2)]], const device float* MINE [[buffer(3)]], const device float* PEER [[buffer(4)]],
    \\    constant uint& rank [[buffer(5)]], const device bfloat* H [[buffer(6)]], const device bfloat* INJ [[buffer(7)]],
    \\    device bfloat* HN [[buffer(8)]], device float* SSP [[buffer(9)]],
    \\    uint g [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint3 tpos [[thread_position_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
    \\  const int j = int(tg.x), r = int(tg.y), i = r * 2560 + j * 256 + int(tpos.x);
    \\  if (ready[0] != expected) return;
    \\  const float mine = MINE[i], peer = PEER[i];
    \\  threadgroup float part[8][4];
    \\  tp_update(H, INJ, float(bfloat(rank == 0u ? mine + peer : peer + mine)), HN, SSP, j, r, tpos.x, g, lane, part);
    \\}
    \\// Owned MoE branches sum routed slots in order, then the gated shared expert, with bf16 rounding.
    \\inline float tp_bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
    \\kernel void tp_branch(const device bfloat* Y [[buffer(0)]], const device float* WTS [[buffer(1)]],
    \\    const device float* LG [[buffer(2)]], device bfloat* OUT [[buffer(3)]], constant uint2& own [[buffer(4)]],
    \\    const device ulong* previous [[buffer(5)]], constant ulong& expected [[buffer(6)]],
    \\    uint2 pos [[thread_position_in_grid]]) {
    \\  if (previous[0] != expected) return;
    \\  constexpr int D = 2560, TOPK = 10, NL = 513;
    \\  const int d = int(pos.x), r = int(own.x) + int(pos.y);
    \\  if (r >= int(own.y)) return;
    \\  float routed = 0.0f;
    \\  for (int k = 0; k < TOPK; k++) routed = fma(float(Y[(r * (TOPK + 1) + k) * D + d]), WTS[r * TOPK + k], routed);
    \\  const float shared = float(bfloat(float(Y[(r * (TOPK + 1) + TOPK) * D + d]) * tp_bsig(float(bfloat(LG[r * NL + NL - 1])))));
    \\  OUT[(r - int(own.x)) * D + d] = bfloat(float(bfloat(routed)) + shared);
    \\}
    \\// After host arrival, update streams from owned rows in MINE and peer rows starting at split.z in THEIRS.
    \\kernel void tp_merge(const device ulong* ready [[buffer(0)]], constant ulong& expected [[buffer(1)]],
    \\    const constant int* rows [[buffer(2)]], const device bfloat* H [[buffer(3)]], const device bfloat* INJ [[buffer(4)]],
    \\    const device bfloat* MINE [[buffer(5)]], const device bfloat* THEIRS [[buffer(6)]], constant uint4& split [[buffer(7)]],
    \\    device bfloat* HN [[buffer(8)]], device float* SSP [[buffer(9)]],
    \\    uint g [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint3 tpos [[thread_position_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
    \\  const int j = int(tg.x), r = int(tg.y), d = j * 256 + int(tpos.x);
    \\  const bool mine = uint(r) >= split.x && uint(r) < split.y;
    \\  if (ready[0] != expected) return;
    \\  float branch;
    \\  if (mine) branch = float(MINE[(r - int(split.x)) * 2560 + d]);
    \\  else {
    \\    const uint at = uint(r - int(split.z)) * 2560u + uint(d);
    \\    branch = float(THEIRS[at]);
    \\  }
    \\  threadgroup float part[8][4];
    \\  tp_update(H, INJ, branch, HN, SSP, j, r, tpos.x, g, lane, part);
    \\}
;

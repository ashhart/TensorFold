// Two Macs' tensor-parallel exchanges: the host relays parts through shared events, and these kernels move bf16 bits.
#include <metal_stdlib>
using namespace metal;
// A rank compares one row's logits across ranks; no kernel sums across them.

// Sync page words (shared with pair_rows.zig): FAILED = the host gave up on a seam; OBSERVED = this side's refusals.
constant constexpr uint FAILED = 16, OBSERVED = 20;
// An argmax id word's bit 31: the row had a logit that is not finite, so pair_pick sends the row to the id sentinel.
constant constexpr uint PAIR_FLAG = 0x80000000u;
constant constexpr uint PAIR_SENTINEL = 0xFFFFFFFFu;

// Kernels' argument block, byte-identical to pair_rows.zig's PairRows: width, the frame row's stride, this rank, one.
struct PairRows {
  uint width, stride, rank, x;
};

// bf16 bits as the exact fp32 value they name (argmax reads a bf16 logits stream this way).
inline float pair_bf16(uint bits) {
  return as_type<float>(bits << 16);
}

// True once the host gave up: FAILED is set before the wait releases, so a later kernel poisons its output rows.
inline bool pair_link_broken(device atomic_uint* sync, uint t) {
  if (atomic_load_explicit(&sync[FAILED], memory_order_relaxed) == 0u) return false;
  if (t == 0) atomic_fetch_add_explicit(&sync[OBSERVED], 1u, memory_order_relaxed);
  return true;
}

// Row r: this rank's own half of the frame row into the exchange slot; a bf16 copy, so it carries the frame's bits.
kernel void pair_stage(device atomic_uint* sync [[buffer(0)]], constant PairRows& a [[buffer(1)]],
                       const device ushort* src [[buffer(2)]], device ushort* dst [[buffer(3)]],
                       uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  const ulong in = ulong(r) * a.stride;
  const ulong out = ulong(r) * a.width;
  if (pair_link_broken(sync, t)) {
    for (uint i = t; i < a.width; i += 256) dst[out + i] = 0x7FC0u; // bf16 NaN: the link stays down
    return;
  }
  for (uint i = t; i < a.width; i += 256) dst[out + i] = src[in + i];
}

// Row r: this rank's half at its own offset in the frame row and the peer's at the other; a bf16 copy, no arithmetic.
kernel void pair_gather(device atomic_uint* sync [[buffer(0)]], constant PairRows& a [[buffer(1)]],
                        const device ushort* mine [[buffer(2)]], device atomic_uint* peer [[buffer(3)]],
                        device ushort* out [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                        uint t [[thread_index_in_threadgroup]]) {
  const ulong mrow = ulong(r) * a.width;
  const ulong orow = ulong(r) * a.stride;
  const ulong own = a.rank == 0 ? orow : orow + a.width;
  const ulong got = a.rank == 0 ? orow + a.width : orow;
  if (pair_link_broken(sync, t)) {
    for (uint i = t; i < a.width; i += 256) {
      out[own + i] = 0x7FC0u;
      out[got + i] = 0x7FC0u;
    }
    return;
  }
  for (uint i = t; i < a.width; i += 256) {
    // The relay wrote the peer's half: an untracked frame read, so one atomic word (two bf16) at a time.
    const uint word = atomic_load_explicit(&peer[(mrow + i) >> 1], memory_order_relaxed);
    out[own + i] = mine[mrow + i];
    out[got + i] = ushort(word >> ((i & 1) * 16));
  }
}

// Row r's argmax over this rank's logits (ids lo..lo+n): (fp32 value, id), lower id on tie; dims.w 0 = fp32, 1 = bf16.
kernel void pair_argmax(const device float* logits [[buffer(0)]], constant uint4& dims [[buffer(1)]],
                        device uint* out [[buffer(2)]], uint r [[threadgroup_position_in_grid]],
                        uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  const ulong row = ulong(r) * dims.x;
  const device float* xf = logits + row;
  const device ushort* xb = reinterpret_cast<const device ushort*>(logits) + row;
  const bool wide = dims.w != 0u;
  float best = -INFINITY;
  uint at = PAIR_SENTINEL;
  bool bad = false;
  for (uint i = t; i < dims.z; i += 1024) {
    const float v = wide ? pair_bf16(uint(xb[i])) : xf[i];
    if (!isfinite(v)) { bad = true; continue; } // the flagged id is the argmax over finite logits
    if (v > best) best = v, at = dims.y + i;
  }
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ob = simd_shuffle_xor(best, o);
    const uint oa = simd_shuffle_xor(at, o);
    if (ob > best || (ob == best && oa < at)) best = ob, at = oa;
  }
  // The flag is the row's, not the thread's: one lane saw a non-finite logit, so every lane must know.
  uint dirty = bad ? 1u : 0u;
  for (ushort o = 16; o > 0; o >>= 1) dirty |= simd_shuffle_xor(dirty, o);
  threadgroup float vb[32];
  threadgroup uint va[32];
  threadgroup uint vd[32];
  if (lane == 0) vb[sg] = best, va[sg] = at, vd[sg] = dirty;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  best = vb[lane];
  at = va[lane];
  dirty = vd[lane];
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ob = simd_shuffle_xor(best, o);
    const uint oa = simd_shuffle_xor(at, o);
    if (ob > best || (ob == best && oa < at)) best = ob, at = oa;
    dirty |= simd_shuffle_xor(dirty, o);
  }
  if (lane == 0) {
    out[2 * r] = dirty != 0u ? as_type<uint>(NAN) : as_type<uint>(best);
    out[2 * r + 1] = dirty != 0u ? (at | PAIR_FLAG) : at;
  }
}

// Each row's pick over both Macs' argmaxes: a flagged row draws no token; else the larger value, lower id on a tie.
kernel void pair_pick(device atomic_uint* sync [[buffer(0)]], constant uint2& rx [[buffer(1)]],
                      const device uint* mine [[buffer(2)]], device atomic_uint* peer [[buffer(3)]],
                      device uint* picks [[buffer(4)]], uint t [[thread_index_in_threadgroup]]) {
  if (pair_link_broken(sync, t)) {
    if (t == 0) for (uint r = 0; r < rx.x; ++r) picks[r] = 0xFFFFFFFFu;  // no such token: visibly wrong, never a token
    return;
  }
  if (t != 0) return;
  for (uint r = 0; r < rx.x; ++r) {
    const float vm = as_type<float>(mine[2 * r]);
    const uint am = mine[2 * r + 1];
    const float vp = as_type<float>(atomic_load_explicit(&peer[2 * r], memory_order_relaxed));
    const uint ap = atomic_load_explicit(&peer[2 * r + 1], memory_order_relaxed);
    // A non-finite row on either Mac draws no token: the same sentinel a broken link reads, never an id.
    picks[r] = ((am | ap) & PAIR_FLAG) != 0u ? PAIR_SENTINEL : (vp > vm || (vp == vm && ap < am)) ? ap : am;
  }
}

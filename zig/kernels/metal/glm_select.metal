// GLM decode key lists on many threadgroups, glm_index_select's bytes (glm_glue.metal); a sampled floor is speed only.
#include <metal_stdlib>
using namespace metal;

struct GlmSelectArgs {
  uint p0, top, width, s_stride, i_stride;
};

constant constexpr uint SPAN = 4096; // blocks a mark threadgroup covers: 128 words, 16 a simdgroup
constant constexpr uint MT = 256;    // mark threads
constant constexpr uint NS = 2048;   // sampled blocks: 256 runs of 8 at even spacing
constant constexpr uint AIM = 2048;  // the marked blocks the floor aims at
constant constexpr uint T2 = 512;    // merge threads
constant constexpr uint CAP = 3072;  // candidates the merge holds

// glm_order_key: a score's bits as a uint in the scores' order (-0 below +0, as glm_index_select ranks them).
inline uint sel_key(float v) {
  const uint u = as_type<uint>(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Exclusive prefix sums of two counts over T2 threads (`part`: 2 T2 / 32 slots), and their totals.
inline uint2 sel_scan2(uint2 v, threadgroup uint* part, uint t, thread uint2& total) {
  constexpr uint SG = T2 / 32;
  const uint lane = t % 32, sg = t / 32;
  const uint2 inside = uint2(simd_prefix_exclusive_sum(v.x), simd_prefix_exclusive_sum(v.y));
  if (lane == 31) {
    part[sg] = inside.x + v.x;
    part[SG + sg] = inside.y + v.y;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint2 base = 0, all = 0;
  for (uint i = 0; i < SG; i++) {
    const uint2 p = uint2(part[i], part[SG + i]);
    base += i < sg ? p : uint2(0);
    all += p;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  total = all;
  return base + inside;
}

// The keys a select reads: a row's scores in device memory, or candidates in threadgroup memory (in block order).
struct RowKeys {
  const device float* s;
  uint key(uint i) const { return sel_key(s[i]); }
  uint block(uint i) const { return i; }
};
struct CandKeys {
  const threadgroup uint* k;
  const threadgroup ushort* b;
  uint key(uint i) const { return k[i]; }
  uint block(uint i) const { return b[i]; }
};

struct SelShared {
  atomic_uint hist[2048];
  uint prefix, need;
  uint part[2 * T2 / 32];
};

// The digit of `hist`'s 2048 bins holding the need-th largest: out[0] = prefix with it at `shift`, out[1] = need left.
template <uint T>
inline void sel_digit(threadgroup atomic_uint* hist, uint need, uint prefix, uint shift, threadgroup uint* out, threadgroup uint* part, uint t) {
  constexpr uint B = 2048 / T;
  uint h[B], own = 0;
  for (uint j = 0; j < B; j++) own += h[j] = atomic_load_explicit(&hist[2047 - B * t - j], memory_order_relaxed);
  const uint incl = simd_prefix_inclusive_sum(own);
  if (t % 32 == 31) part[t / 32] = incl;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint above = incl - own;
  for (uint i = 0; i < t / 32; i++) above += part[i];
  if (above < need && above + own >= need) {
    uint d = 0;
    for (uint j = 0; j < B; j++) {
      if (above + h[j] >= need) {
        d = 2047 - B * t - j;
        break;
      }
      above += h[j];
    }
    out[0] = prefix | (d << shift);
    out[1] = need - above;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// glm_index_select's choice of n keys (top <= n) as 4-entry runs in order; radix passes of 11, 11 and 10 bits.
template <typename Src>
inline void sel_core(threadgroup SelShared& sh, Src src, uint n, uint top, device int* out, uint t) {
  uint prefix = 0, need = top;
  threadgroup uint* res = &sh.prefix;
  for (int pass = 0; pass < 3; pass++) {
    const uint shift = pass == 0 ? 21u : (pass == 1 ? 10u : 0u);
    const uint mask = pass == 2 ? 1023u : 2047u;
    const uint hi_mask = pass == 0 ? 0u : (0xffffffffu << (shift + (pass == 2 ? 10u : 11u)));
    for (uint i = t; i < 2048; i += T2) atomic_store_explicit(&sh.hist[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint b = t; b < n; b += T2) {
      const uint key = src.key(b);
      if ((key & hi_mask) == (prefix & hi_mask)) atomic_fetch_add_explicit(&sh.hist[(key >> shift) & mask], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sel_digit<T2>(sh.hist, need, prefix, shift, res, sh.part, t);
    prefix = sh.prefix;
    need = sh.need;
  }
  // in order: every key above the threshold, then the first `need` equal to it; each thread a run
  const uint per = (n + T2 - 1) / T2;
  const uint lo = min(t * per, n), hi = min(lo + per, n);
  uint2 c = 0;
  for (uint i = lo; i < hi; i++) {
    const uint key = src.key(i);
    c += uint2(key > prefix ? 1u : 0u, key == prefix ? 1u : 0u);
  }
  uint2 total;
  uint2 before = sel_scan2(c, sh.part, t, total);
  for (uint i = lo; i < hi; i++) {
    const uint key = src.key(i);
    if (key < prefix) continue;
    const bool above = key > prefix;
    if (above || before.y < need) {
      const uint at = before.x + min(before.y, need);
      const int b = int(src.block(i));
      for (uint j = 0; j < 4; j++) out[4 * at + j] = 4 * b + int(j);
    }
    before += uint2(above ? 1u : 0u, above ? 0u : 1u);
  }
}

// The floor of row r's scores [0, np): the lowest key of the 22-bit bin holding the need-th best of NS sampled blocks.
inline uint sel_floor(const device float* s, uint np, uint t, threadgroup atomic_uint* hist, threadgroup uint* sh) {
  constexpr uint PT = NS / MT;
  uint u[PT];
  for (uint j = 0; j < PT; j++) {
    const uint q = t + j * MT;
    u[j] = sel_key(s[uint(ulong(q / 8) * np / (NS / 8)) / 8 * 8 + q % 8]);
  }
  uint prefix = 0, need = max(uint(ulong(AIM) * NS / np), 1u);
  for (int pass = 0; pass < 2; pass++) {
    const uint shift = pass == 0 ? 21u : 10u;
    for (uint i = t; i < 2048; i += MT) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < PT; j++)
      if (pass == 0 || (u[j] >> 21) == (prefix >> 21)) atomic_fetch_add_explicit(&hist[(u[j] >> shift) & 2047u], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sel_digit<MT>(hist, need, prefix, shift, sh, sh + 2, t);
    prefix = sh[0];
    need = sh[1];
  }
  return prefix;
}

// Grid (cdiv(blocks, SPAN), rows) of 256: span tg.x's blocks at or above row tg.y's floor (bit b of word w: 32 w + b).
[[kernel]] void glm_select_mark(const device float* scores [[buffer(0)]], device uint* bits [[buffer(1)]],
                                constant GlmSelectArgs& g [[buffer(2)]], constant uint& words [[buffer(3)]],
                                uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup atomic_uint hist[2048];
  threadgroup uint sh[2 + MT / 32];
  const uint row = tg.y, np = (g.p0 + row + 1) / 4;
  const uint w0 = tg.x * (SPAN / 32);
  if (np <= CAP || w0 * 32 >= np) return;
  const device float* s = scores + size_t(row) * g.s_stride;
  const uint floor_key = sel_floor(s, np, t, hist, sh);
  const uint nw = min(SPAN / 32, (np + 31) / 32 - w0);
  device uint* out = bits + size_t(row) * words;
  for (uint wi = sg; wi < nw; wi += MT / 32) {
    const uint i = (w0 + wi) * 32 + lane;
    const bool ge = i < np && sel_key(s[min(i, np - 1)]) >= floor_key;
    const uint word = uint(ulong(simd_vote::vote_t(simd_ballot(ge))));
    if (lane == 0) out[w0 + wi] = word;
  }
}

// A threadgroup a row: glm_index_select's key list, from the marked blocks when they hold TOP to CAP of them.
[[kernel]] void glm_select_merge(const device float* scores [[buffer(0)]], device int* indices [[buffer(1)]],
                                 constant GlmSelectArgs& g [[buffer(2)]], constant uint& words [[buffer(3)]],
                                 const device uint* bits [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                                 uint t [[thread_position_in_threadgroup]]) {
  threadgroup SelShared sh;
  threadgroup uint ckey[CAP];
  threadgroup ushort cblock[CAP];
  const uint np = (g.p0 + row + 1) / 4, position = g.p0 + row;
  const device float* s = scores + size_t(row) * g.s_stride;
  device int* out = indices + size_t(row) * g.i_stride;
  uint chosen = min(g.top, np);
  if (g.top >= np) { // every block, in order
    for (uint i = t; i < 4 * np; i += T2) out[i] = int(i);
  } else if (np <= CAP) { // the whole row as candidates
    for (uint i = t; i < np; i += T2) {
      ckey[i] = sel_key(s[i]);
      cblock[i] = ushort(i);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sel_core(sh, CandKeys{ckey, cblock}, np, g.top, out, t);
  } else {
    const device uint* ge = bits + size_t(row) * words;
    const uint nw = (np + 31) / 32, per = (nw + T2 - 1) / T2;
    const uint wa = min(nw, t * per), wb = min(nw, wa + per);
    uint count = 0;
    for (uint w = wa; w < wb; w++) count += popcount(ge[w]);
    uint2 all;
    uint at = sel_scan2(uint2(count, 0), sh.part, t, all).x;
    const uint n = all.x;
    if (n >= g.top && n <= CAP && np <= 65536) { // the marked blocks in order, then their keys
      for (uint w = wa; w < wb; w++)
        for (uint m = ge[w]; m != 0; m &= m - 1) cblock[at++] = ushort(32 * w + ctz(m));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint c = t; c < n; c += T2) ckey[c] = sel_key(s[cblock[c]]);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      sel_core(sh, CandKeys{ckey, cblock}, n, g.top, out, t);
    } else {
      sel_core(sh, RowKeys{s}, np, g.top, out, t);
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  const uint tail = (position + 1) % 4;
  for (uint i = 4 * chosen + t; i < g.width; i += T2) {
    const uint j = i - 4 * chosen;
    out[i] = j < tail ? int(position + 1 - tail + j) : -1;
  }
}

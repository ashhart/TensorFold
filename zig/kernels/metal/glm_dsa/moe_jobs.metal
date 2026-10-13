// GLM-5.3's expert jobs written on the GPU from the route: each picked expert over its rows, the shared one last.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint MAXH = 16, MAXR = 16, SHARED = 256;

// The trellis kernels' job (core/exlr8_job.zig's Job).
struct Job {
  uint k, blocks, base, in_scale;
  uint out_scale, out_sign, out_block, x_col;
  uint y_col, raw, members, count;
};

// One expert's tables on this Mac: per local half its width, first tile-run offset, channel and block scales.
struct ExpertDesc {
  uint gate_k[MAXH], gate_base[MAXH], gate_chan[MAXH], gate_block[MAXH];
  uint up_k[MAXH], up_base[MAXH], up_chan[MAXH], up_block[MAXH];
  uint down_k, down_base, down_suh, down_block;
};

struct JobsArgs {
  uint rows, halves, topk, slots;
  uint gu_cap, pad0, pad1, pad2;
};

// Jobs, member lists, per-slot tables and the two dispatches' threadgroup counts for one window.
kernel void glm_moe_jobs(const device int* lids [[buffer(0)]], const device int* lmem [[buffer(1)]],
                         const device int* lcount [[buffer(2)]], const device int* pick [[buffer(3)]],
                         const device float* wts [[buffer(4)]], const device ExpertDesc* ex [[buffer(5)]],
                         constant JobsArgs& a [[buffer(6)]], device Job* jobs [[buffer(7)]],
                         device uint* members [[buffer(8)]], device float* wts9 [[buffer(9)]],
                         device uint* sv9 [[buffer(10)]], device uint* suh9 [[buffer(11)]],
                         device uint* dispatch [[buffer(12)]], uint t [[thread_index_in_threadgroup]]) {
  const uint n = uint(lcount[0]), total = n + 1, loc = 128 * a.halves;
  for (uint p = t; p < a.rows * a.slots; p += 256) {
    const uint r = p / a.slots, k = p % a.slots;
    const uint e = k < a.topk ? uint(pick[r * a.topk + k]) : SHARED;
    wts9[p] = k < a.topk ? wts[r * a.topk + k] : 1.0f;
    sv9[p] = ex[e].down_block;
    suh9[p] = ex[e].down_suh;
  }
  for (uint i = t; i < total; i += 256) {
    const bool shared = i == n;
    const uint e = shared ? SHARED : uint(lids[i]);
    const device ExpertDesc& d = ex[e];
    // gate/up members read row r and write slot r * slots + k; down's read and write that slot
    const uint gm = i * 2 * MAXR, dm = gm + MAXR;
    uint cnt = 0;
    for (uint j = 0; j < (shared ? a.rows : MAXR); ++j) {
      uint r, p9;
      if (shared) {
        r = j;
        p9 = j * a.slots + a.slots - 1;
      } else {
        const int m = lmem[i * MAXR + j];
        if (m < 0) break;
        r = uint(m) / a.topk;
        p9 = r * a.slots + uint(m) % a.topk;
      }
      members[gm + cnt] = (r << 16) | p9;
      members[dm + cnt] = (p9 << 16) | p9;
      ++cnt;
    }
    for (uint h = 0; h < a.halves; ++h) {
      const uint c = 128 * h;
      jobs[i * 2 * a.halves + 2 * h] = Job{d.gate_k[h], 48, d.gate_base[h], d.gate_block[h], d.gate_chan[h], 0xFFFFFFFFu, 0xFFFFFFFFu, 0, c, 0, gm, cnt};
      jobs[i * 2 * a.halves + 2 * h + 1] = Job{d.up_k[h], 48, d.up_base[h], d.up_block[h], d.up_chan[h], 0xFFFFFFFFu, 0xFFFFFFFFu, 0, loc + c, 0, gm, cnt};
    }
    for (uint ob = 0; ob < 48; ++ob)
      jobs[a.gu_cap + i * 48 + ob] = Job{d.down_k, a.halves, d.down_base + ob * a.halves, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0, 128 * ob, 1, dm, cnt};
  }
  if (t == 0) {
    dispatch[0] = total * 2 * a.halves, dispatch[1] = 1, dispatch[2] = 1;
    dispatch[4] = total * 48, dispatch[5] = 1, dispatch[6] = 1;
  }
}

struct PromptJobsArgs {
  uint rows, halves, topk, slots;
  uint gu_cap, experts, pad0, pad1;
};

// A prompt window's expert jobs: each expert's rows in row order, tiles of 32 rows with their jobs and member lists.
kernel void glm_moe_jobs_prompt(const device int* pick [[buffer(0)]], const device float* wts [[buffer(1)]],
                                const device ExpertDesc* ex [[buffer(2)]], constant PromptJobsArgs& a [[buffer(3)]],
                                device Job* jobs [[buffer(4)]], device uint* members [[buffer(5)]],
                                device float* wts9 [[buffer(6)]], device uint* sv9 [[buffer(7)]],
                                device uint* suh9 [[buffer(8)]], device uint* dispatch [[buffer(9)]],
                                uint t [[thread_index_in_threadgroup]]) {
  threadgroup uint first[SHARED + 2];
  const uint loc = 128 * a.halves;
  for (uint p = t; p < a.rows * a.slots; p += 1024) {
    const uint r = p / a.slots, k = p % a.slots;
    const uint e = k < a.topk ? uint(pick[r * a.topk + k]) : SHARED;
    wts9[p] = k < a.topk ? wts[r * a.topk + k] : 1.0f;
    sv9[p] = ex[e].down_block;
    suh9[p] = ex[e].down_suh;
  }
  // each expert's rows, then the tiles before it
  for (uint e = t; e <= SHARED; e += 1024) {
    uint n = 0;
    if (e == SHARED) n = a.rows;
    else
      for (uint i = 0; i < a.rows * a.topk; ++i) n += uint(pick[i]) == e ? 1u : 0u;
    first[e + 1] = (n + 31) / 32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    first[0] = 0;
    for (uint e = 0; e <= SHARED; ++e) first[e + 1] += first[e];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = t; e <= SHARED; e += 1024) {
    const uint tiles = first[e + 1] - first[e];
    if (tiles == 0) continue;
    uint n = 0;
    for (uint r = 0; r < a.rows; ++r) {
      uint k = a.slots - 1;
      if (e != SHARED) {
        k = a.topk;
        for (uint q = 0; q < a.topk; ++q)
          if (uint(pick[r * a.topk + q]) == e) k = q;
        if (k == a.topk) continue;
      }
      const uint p9 = r * a.slots + k, m = (first[e] + n / 32) * 64 + n % 32;
      members[m] = (r << 16) | p9;
      members[m + 32] = (p9 << 16) | p9;
      ++n;
    }
    const device ExpertDesc& d = ex[e];
    for (uint i = 0; i < tiles; ++i) {
      const uint tile = first[e] + i, cnt = min(32u, n - 32 * i), gm = tile * 64, dm = gm + 32;
      for (uint h = 0; h < a.halves; ++h) {
        const uint c = 128 * h;
        jobs[tile * 2 * a.halves + 2 * h] = Job{d.gate_k[h], 48, d.gate_base[h], d.gate_block[h], d.gate_chan[h], 0xFFFFFFFFu, 0xFFFFFFFFu, 0, c, 0, gm, cnt};
        jobs[tile * 2 * a.halves + 2 * h + 1] = Job{d.up_k[h], 48, d.up_base[h], d.up_block[h], d.up_chan[h], 0xFFFFFFFFu, 0xFFFFFFFFu, 0, loc + c, 0, gm, cnt};
      }
      for (uint ob = 0; ob < 48; ++ob)
        jobs[a.gu_cap + tile * 48 + ob] = Job{d.down_k, a.halves, d.down_base + ob * a.halves, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0, 128 * ob, 1, dm, cnt};
    }
  }
  if (t == 0) {
    const uint tiles = first[SHARED + 1];
    dispatch[0] = tiles * 2 * a.halves, dispatch[1] = 1, dispatch[2] = 1;
    dispatch[4] = tiles * 48, dispatch[5] = 1, dispatch[6] = 1;
  }
}

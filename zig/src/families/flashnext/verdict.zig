//! The GPU-side rounds' verdict kernels: kept drafts and tokens (fz_accept), copied drafts, a forced span (think.zig).

pub const source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\// The round's verdict on the GPU: drafts kept, the emitted tokens to the ring, the next window's pending token,
    \\// positions and n-gram history; ar = keep, target length, round, then the meta blocks the next round reads:
    \\// the window's positions [4, 20), key counts [20, 36), cache meta [36, 39); the head's absorb [40, 56), [56, 72),
    \\// [72, 75); chained draft j at 76 + 36 (j - 1): positions, key counts (16 each), cache meta.
    \\// cfg = this window's rows, the next window's rows, cache capacity. The ring keeps 512 rounds of 20 words: keep,
    \\// the drafts' source (1: copied, from ar[801]), the copy's match length (ar[802]), the emitted tokens; H, the
    \\// token history (prompt, then every emitted token; its length in ar[800]), gets this round's.
    \\kernel void fz_accept(device uint* wids [[buffer(0)]], device const uint* picks [[buffer(1)]],
    \\    device int* ar [[buffer(2)]], device uint* ring [[buffer(3)]], device long* pm [[buffer(4)]],
    \\    constant uint4& cfg [[buffer(5)]], device uint* H [[buffer(6)]], const device int* G [[buffer(7)]],
    \\    uint tid [[thread_position_in_grid]]) {
    \\  if (tid != 0) return;
    \\  const int W = min(int(cfg.x), G[0]), Wn = int(cfg.y), cap = int(cfg.z); // G: a GPU-gated window's verified rows
    \\  int keep = 1;
    \\  while (keep < W && wids[keep] == picks[keep - 1]) keep++;
    \\  const int round = ar[2], slot = (round & 511) * 20;
    \\  ring[slot] = uint(keep);
    \\  ring[slot + 1] = uint(ar[801]);
    \\  ring[slot + 2] = uint(ar[802]);
    \\  for (int i = 0; i < keep; i++) ring[slot + 3 + i] = picks[i];
    \\  ring[slot + 19] = uint(W);
    \\  const int hl = ar[800];
    \\  for (int i = 0; i < keep; i++) H[hl + i] = picks[i];
    \\  ar[800] = hl + keep;
    \\  for (int i = 0; i < keep; i++) { pm[0] = pm[1]; pm[1] = long(wids[i]); }
    \\  const int t_old = ar[1], t_new = t_old + keep;
    \\  ar[0] = keep; ar[1] = t_new; ar[2] = round + 1;
    \\  for (int i = 0; i < 16; i++) {
    \\    ar[4 + i] = i < Wn ? t_new + i : 0;
    \\    ar[20 + i] = i < Wn ? t_new + i + 1 : 0;
    \\    ar[40 + i] = i < W ? t_old + i : 0;
    \\    ar[56 + i] = i < W ? t_old + i + 1 : 0;
    \\  }
    \\  ar[36] = t_new; ar[37] = cap; ar[38] = Wn;
    \\  ar[72] = t_old; ar[73] = cap; ar[74] = W;
    \\  for (int j = 1; j < Wn - 1; j++) {
    \\    const int b = 76 + (j - 1) * 36;
    \\    for (int i = 0; i < 16; i++) { ar[b + i] = 0; ar[b + 16 + i] = 0; }
    \\    ar[b] = t_new + j - 1; ar[b + 16] = t_new + j;
    \\    ar[b + 32] = t_new + j - 1; ar[b + 33] = cap; ar[b + 34] = 1;
    \\  }
    \\  wids[0] = picks[keep - 1];
    \\}
    \\// Copy drafts: the latest earlier occurrence of the longest suffix (cfg.y .. 8 tokens) of the history H[0, L);
    \\// when there is one and it is at least cfg.z long or the tokens after it start with the head's first cfg.w drafts,
    \\// they replace the drafts wids[1 .. cfg.x) (as many as follow it); ar[801] = 1 then, else 0; ar[802] the match
    \\// length. One threadgroup of 1024.
    \\kernel void fz_lookup(device const uint* H [[buffer(0)]], device int* ar [[buffer(1)]], device uint* wids [[buffer(2)]],
    \\    constant uint4& cfg [[buffer(3)]], uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint sg [[simdgroup_index_in_threadgroup]]) {
    \\  const int L = ar[800], W = int(cfg.x), MINN = int(cfg.y), LONGN = int(cfg.z), AGREE = int(cfg.w);
    \\  const uint last = H[L - 1];
    \\  uint best = 0;
    \\  for (int p = int(t); p < L - 1; p += 1024) {
    \\    if (H[p] != last) continue;
    \\    int n = 1;
    \\    while (n < 8 && p - n >= 0 && H[p - n] == H[L - 1 - n]) n++;
    \\    best = max(best, (uint(n) << 24) | uint(p + 1));
    \\  }
    \\  best = simd_max(best);
    \\  threadgroup uint part[32];
    \\  if (lane == 0) part[sg] = best;
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (t != 0) return;
    \\  for (int i = 0; i < 32; i++) best = max(best, part[i]);
    \\  const int n = int(best >> 24), p = int(best & 0xffffffu) - 1;
    \\  ar[801] = 0;
    \\  ar[802] = n;
    \\  if (n < MINN || p < 0) return;
    \\  if (n < LONGN)
    \\    for (int i = 0; i < AGREE && i < W - 1; i++) if (p + 1 + i >= L || H[p + 1 + i] != wids[1 + i]) return;
    \\  const int k = min(W - 1, L - 1 - p);
    \\  for (int i = 0; i < k; i++) wids[1 + i] = H[p + 1 + i];
    \\  ar[801] = 1;
    \\}
    \\// Before a window of c.x rows; a forced span at ar + c.y: first position (0: done), cancel token, length, seen, tokens.
    \\kernel void fz_force(device uint* wids [[buffer(0)]], device int* ar [[buffer(1)]], device const uint* H [[buffer(2)]],
    \\    constant uint2& c [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    \\  device int* f = ar + c.y;
    \\  if (tid != 0 || f[0] <= 0) return;
    \\  for (int p = f[3] + 1; p <= ar[1]; p++) { // tokens landed since it last looked (H[p]): its last, or a cancel first
    \\    const int at = p - f[0];
    \\    if (at + 1 >= f[2] || (at < 0 && int(H[p]) == f[1])) { f[0] = 0; return; }
    \\  }
    \\  f[3] = ar[1];
    \\  for (int j = 1; j < int(c.x); j++) { // the window's drafts at the span's positions take its tokens
    \\    const int at = ar[1] + j - f[0];
    \\    if (at >= 0 && at < f[2]) wids[j] = uint(f[4 + at]);
    \\  }
    \\}
    \\// Before the verdict: the target's picks take the span's tokens at its positions, up to a cancel token landing first.
    \\kernel void fz_span(device uint* picks [[buffer(0)]], device const int* ar [[buffer(1)]], constant uint2& c [[buffer(2)]],
    \\    uint tid [[thread_position_in_grid]]) {
    \\  device const int* f = ar + c.y;
    \\  if (tid != 0 || f[0] <= 0) return;
    \\  for (int i = 0; i < int(c.x); i++) {
    \\    const int at = ar[1] + 1 + i - f[0];
    \\    if (at >= 0 && at < f[2]) picks[i] = uint(f[4 + at]);
    \\    else if (at < 0 && int(picks[i]) == f[1]) return;
    \\  }
    \\}
    \\// A sampled window's draw positions: row i draws the token after it, at the cache length (ar[1]) + 1 + i.
    \\kernel void fz_positions(device uint* pos [[buffer(0)]], device const int* ar [[buffer(1)]],
    \\    constant uint& n [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
    \\  if (tid < n) pos[tid] = uint(ar[1] + 1 + int(tid));
    \\}
;

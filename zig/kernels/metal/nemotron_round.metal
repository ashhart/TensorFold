// A lone greedy stream's round on the GPU: the verify's arguments, the accept, the head's arguments and its confidence stop.
#include <metal_stdlib>
using namespace metal;

// state (u32): len, mtp_len, copy source, pending, kept, kept - 1, rounds, next rows, top-1 product, alive, copy rows,

// match, tail copy rows, second choice row, its chance; from 16: a missed copy row's place + 1, rounds left, first copy row
#define S_RESUME 16
#define S_GUESS 19  // a copied window that also runs the head's first level, for its guess beside the copy's first row
#define S_LEAF 20   // the leaf's kind: 1 a second choice, 2 the head's guess beside a copy

// record (u32): kept, rows, copy rows (0: the head's window), match length, the kept tokens from 4, the drafts from 64

// args (i32) blocks, each 256 bytes apart where a kernel reads them as constants (gpu_round.zig names them)
#define A_KV 0
#define A_DIMS 64
#define A_ABS 128
#define A_ROWS 192
#define A_MD 256
#define A_SEGI 320
#define A_VA 384
#define A_VB 640
#define A_VC 896
#define A_STEP 1024
#define A_STEP_STRIDE 128
#define A_TP 2048   // a tree window's shared attention pass dims: P, chunks, rows, 0, simdgroup tiles
#define A_TT 2112   // its tail and merge dims: len, P, chunks A, chunks B, rows
#define A_CMP 2176  // the kept path's compaction: len, rows, capacity, head dim
#define A_PAR 2240  // the window's rows' parents (row 0, the pending token: -1)
#define A_PATH 2304 // the kept path's rows (past it: its last row)
#define A_FOL 2368  // the token after each kept row, for the head (past them: 0)

// A window row's parent place in the Mamba tables: the replayed rows' last before row 0, else its parent's place.
inline int tf_place_parent(const device int* args, int k, int p) {
  if (p < k) return p - 1;
  const int q = args[A_PAR + (p - k)];
  return q < 0 ? k - 1 : k + q;
}

// The next verify's kv write and dims, its Mamba tables (the kept path replayed, the window by parent), tree tables.
kernel void tf_round_args(device uint* state [[buffer(0)]],
                          device int* args [[buffer(1)]],
                          constant uint* in [[buffer(2)]],  // R, qkv, koff, capacity, g, load slot, store slot, parity, rid, replay rows, virtual rows
                          device int* tdepths [[buffer(3)]],
                          device int* tpaths [[buffer(4)]],  // [rows, 64]
                          uint t [[thread_position_in_grid]]) {
  if (t != 0) return;
  const int R = int(in[0]), g = int(in[4]), load = int(in[5]), store = int(in[6]), parity = int(in[7]), rid = int(in[8]);
  const int len = int(state[0]), k = int(state[4]), rows = int(state[7]), vmax = int(in[10]);
  args[A_KV + 0] = int(in[1]); args[A_KV + 1] = int(in[2]); args[A_KV + 2] = int(in[3]); args[A_KV + 3] = len;
  const int keys = len + R;
  args[A_DIMS + 0] = keys; args[A_DIMS + 1] = (keys + 511) / 512; args[A_DIMS + 2] = R; args[A_DIMS + 3] = 1; args[A_DIMS + 4] = g * R / 16;
  args[A_ROWS] = rows;
  args[A_MD + 0] = k + rows; args[A_MD + 1] = int(in[9]);
  device int* sg = args + A_SEGI;
  sg[0] = 0; sg[1] = k; sg[2] = load; sg[3] = rid; sg[4] = parity; sg[5] = 1; sg[6] = 0; sg[7] = 0;
  device int4* va = (device int4*)(args + A_VA);
  device int4* vb = (device int4*)(args + A_VB);
  bool tree = false;
  for (int i = 1; i < rows; i++) tree = tree || args[A_PAR + i] != i - 1;
  if (!tree) {
    // a chain: each row's parent the row before it (the kept path's rows replayed by their input rows)
    for (int v = 0; v < vmax; v++) {
      const int l = min(v, k + R - 1);
      va[v] = int4(0, l, l == k - 1 ? store : -1, l - 1);
      vb[v] = int4(l - 3, l - 2, l - 1, l < k ? args[A_PATH + l] : 0);
      args[A_VC + v] = -1;
    }
    return;
  }
  for (int v = 0; v < vmax; v++) args[A_VC + v] = -1;
  int saves = 0;
  for (int v = 0; v < vmax; v++) {
    const int l = min(v, k + R - 1);  // rows past the window repeat its last (the conv grid is fixed, the scan stops at k + rows)
    const int par = tf_place_parent(args, k, l);
    const int anc2 = par < 0 ? par - 1 : tf_place_parent(args, k, par);
    const int anc3 = anc2 < 0 ? anc2 - 1 : tf_place_parent(args, k, anc2);
    va[v] = int4(0, l, l == k - 1 ? store : -1, par);
    vb[v] = int4(anc3, anc2, par, l < k ? args[A_PATH + l] : 0);
    // a parent whose child is not its next row keeps its state for the scan to reload
    if (v == l && v < k + rows && par >= 0 && par != l - 1 && args[A_VC + par] < 0) args[A_VC + par] = saves++;
  }
  // attention: each window row's depth and ancestors by depth, then the tree's dims
  int deepest = 0;
  for (int r = 0; r < R; r++) {
    const int p = args[A_PAR + r];
    const int d = p < 0 ? 0 : tdepths[p] + 1;
    tdepths[r] = d;
    for (int i = 0; i < d; i++) tpaths[r * 64 + i] = tpaths[p * 64 + i];
    tpaths[r * 64 + d] = r;
    deepest = max(deepest, d);
  }
  const int pt = len / 64 * 64, nca = (pt + 511) / 512, ncb = (len + deepest) / 512 - pt / 512 + 1;
  args[A_TP + 0] = pt; args[A_TP + 1] = nca; args[A_TP + 2] = R; args[A_TP + 3] = 0; args[A_TP + 4] = g * R / 16;
  args[A_TT + 0] = len; args[A_TT + 1] = pt; args[A_TT + 2] = nca; args[A_TT + 3] = ncb; args[A_TT + 4] = R;
}

// The kept path (each row the draw before it chose), bonus, record, the head's tokens, compaction, the next parents.
kernel void tf_round_accept(device uint* window [[buffer(0)]],
                            const device uint* draws [[buffer(1)]],
                            device uint* state [[buffer(2)]],
                            device uint* next_window [[buffer(3)]],
                            device uint* record [[buffer(4)]],
                            device int* args [[buffer(5)]],
                            constant uint* in [[buffer(6)]],  // depth, qkv, koff, head capacity, g
                            device uint* context [[buffer(7)]],
                            uint t [[thread_position_in_grid]]) {
  if (t != 0) return;
  const uint depth = in[0], g = in[4], R = in[5];
  const uint rows = state[7];
  uint kept = 1, node = 0;
  args[A_PATH] = 0;
  for (;;) {
    uint next = 0;
    for (uint c = node + 1; c < rows; c++)
      if (args[A_PAR + c] == int(node) && window[c] == draws[node]) { next = c; break; }
    if (next == 0) break;
    args[A_PATH + kept++] = int(next);
    node = next;
  }
  const uint bonus = draws[node];
  record[0] = kept;
  record[1] = rows;
  record[2] = state[10];
  record[3] = state[11];
  record[60] = state[12];  // copy rows after the head's drafts
  // a copied row the target replaced: where the copy stood in the context, to resume after the replacement
  if (state[S_RESUME + 1] > 0) state[S_RESUME + 1] -= 1;
  const uint copy_from = state[S_RESUME + 2], copy_n = state[10] != 0 ? state[10] : state[12];
  if (copy_n > 0 && kept < rows && kept >= copy_from && kept < copy_from + copy_n) {
    state[S_RESUME] = state[2] + (kept - copy_from) + 1;
    state[S_RESUME + 1] = 3;
  }
  record[61] = state[13];  // the leaf's row (0: none), its chance, its kind
  record[59] = state[14];
  record[58] = state[S_LEAF];
  state[S_LEAF] = 0;
  state[S_GUESS] = 0;
  state[12] = 0;
  const uint len = state[0];  // the context holds len + 1 tokens: the pending one last
  for (uint i = 1; i < kept; i++) {
    const uint tok = window[args[A_PATH + i]];
    record[3 + i] = tok;
    context[len + i] = tok;
    args[A_FOL + i - 1] = int(tok);
  }
  record[3 + kept] = bonus;
  context[len + kept] = bonus;
  args[A_FOL + kept - 1] = int(bonus);  // the head absorbs path row kept - 1 with the token after it: the chain's root
  for (uint i = kept; i < R; i++) {
    args[A_PATH + i] = args[A_PATH + kept - 1];
    args[A_FOL + i] = 0;
  }
  for (uint i = 1; i < rows; i++) record[63 + i] = window[i];  // every draft the window verified
  uint leaf_kept = 0;
  for (uint i = 1; i < kept; i++) leaf_kept |= uint(args[A_PATH + i]) == state[13] ? 1u : 0u;
  record[62] = state[13] != 0 ? leaf_kept : 0;  // it was kept
  state[13] = 0;
  args[A_CMP + 0] = int(len); args[A_CMP + 1] = int(kept); args[A_CMP + 2] = int(in[7]); args[A_CMP + 3] = int(in[8]);
  for (uint i = 0; i < 64; i++) args[A_PAR + i] = int(i) - 1;
  const uint mtp = state[1];
  state[0] += kept;
  state[1] = mtp + kept;
  state[3] = bonus;
  state[4] = kept;
  state[5] = kept - 1;
  state[6] += 1;
  state[7] = 1;
  state[8] = as_type<uint>(1.0f);
  state[9] = 1;
  next_window[0] = bonus;
  // the head takes the window's rows from mtp (kept - 1 of them stay), then chains from row kept - 1
  args[A_ABS + 0] = int(in[1]); args[A_ABS + 1] = int(in[2]); args[A_ABS + 2] = int(in[3]); args[A_ABS + 3] = int(mtp);
  for (uint j = 0; j < depth; j++) {
    device int* s = args + A_STEP + A_STEP_STRIDE * j;
    const int at = int(mtp + kept - 1 + j);
    s[0] = int(in[1]); s[1] = int(in[2]); s[2] = int(in[3]); s[3] = at;
    s[64] = at + 1; s[65] = (at + 1 + 511) / 512; s[66] = 1; s[67] = 1; s[68] = int(g / 16);
  }
}

// The better of two (value, index) candidates: NaN first, then the larger value, ties to the lower index (MLX argmax).
inline bool tf_first(float a, uint ia, float b, uint ib) {
  if (isnan(a)) return !isnan(b) || ia < ib;
  if (isnan(b)) return false;
  return a > b || (a == b && ia < ib);
}

// Level j's draft and chance (tf_topk_probs's first id), then the stop: the running top-1 product to `power` vs its bar.
kernel void tf_round_top1(const device bfloat* x [[buffer(0)]],
                          const device uint* map [[buffer(1)]],
                          device uint* ids [[buffer(2)]],
                          device float* probs [[buffer(3)]],
                          constant uint& vocab [[buffer(4)]],
                          device uint* draft [[buffer(5)]],
                          constant float* bars [[buffer(6)]],
                          device uint* state [[buffer(7)]],
                          const device uint* tmpl [[buffer(8)]],
                          device uint* live [[buffer(9)]],
                          constant uint4& how [[buffer(10)]],  // level j, the next level's dispatches (0: none), its first entry, power
                          device uint* second [[buffer(11)]],  // by level: the second choice's id, chance, the trunk's reach
                          constant uint& seconds [[buffer(12)]],
                          uint t [[thread_position_in_threadgroup]],
                          uint sgi [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float bv[32];
  threadgroup uint bi[32];
  threadgroup float top_v[1];
  threadgroup uint top_i[1];
  threadgroup bool go[1];
  threadgroup float total_v[1];
  threadgroup float reach_v[1];
  if (state[9] == 0) {  // a level past the stop: nothing to draft, the next level stays dead
    if (t < 3 * how.y) live[3 * how.z + t] = 0;
    return;
  }
  float best = -INFINITY;
  uint at = 0xffffffffu;
  for (uint i = t; i < vocab; i += 1024) {
    const float v = float(x[i]);
    if (tf_first(v, i, best, at)) { best = v; at = i; }
  }
  for (uint o = 16; o > 0; o >>= 1) {
    const float ov = simd_shuffle_xor(best, o);
    const uint oi = simd_shuffle_xor(at, o);
    if (tf_first(ov, oi, best, at)) { best = ov; at = oi; }
  }
  if (lane == 0) { bv[sgi] = best; bi[sgi] = at; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float b0 = bv[0];
    uint i0 = bi[0];
    for (int q = 1; q < 32; q++) if (tf_first(bv[q], bi[q], b0, i0)) { b0 = bv[q]; i0 = bi[q]; }
    top_v[0] = b0;
    top_i[0] = i0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum = 0.0f;
  const float top = top_v[0];
  for (uint i = t; i < vocab; i += 1024) sum += metal::exp(float(x[i]) - top);
  sum = simd_sum(sum);
  if (lane == 0) bv[sgi] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float total = 0.0f;
    for (int q = 0; q < 32; q++) total += bv[q];
    total_v[0] = total;
    const float p0 = metal::exp(top - top) / total;
    ids[0] = map[top_i[0]];
    probs[0] = p0;
    const uint j = how.x;
    reach_v[0] = as_type<float>(state[8]);
    bool alive = state[9] != 0;
    if (state[S_GUESS] != 0) {
      // a copied window's first level: its guess goes beside the copy's first row, no level runs after it
      second[24] = map[top_i[0]];
      second[25] = as_type<uint>(p0);
      state[9] = 0;
      alive = false;
    } else if (alive) {
      draft[0] = map[top_i[0]];
      const float p = as_type<float>(state[8]) * p0;
      state[8] = as_type<uint>(p);
      alive = metal::pow(p, as_type<float>(how.w)) >= bars[j];
      if (alive) state[7] = j + 2;
      state[9] = alive ? 1 : 0;
    }
    go[0] = alive;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 3 * how.y) live[3 * how.z + t] = go[0] ? tmpl[t] : 0;
  if (seconds == 0 || state[S_GUESS] != 0) return;
  // the second choice (tf_topk_probs's rank 1): the best id past the first, its chance
  float b1 = -INFINITY;
  uint a1 = 0xffffffffu;
  for (uint i = t; i < vocab; i += 1024) {
    const float v = float(x[i]);
    if (i != top_i[0] && tf_first(v, i, b1, a1)) { b1 = v; a1 = i; }
  }
  for (uint o = 16; o > 0; o >>= 1) {
    const float ov = simd_shuffle_xor(b1, o);
    const uint oi = simd_shuffle_xor(a1, o);
    if (tf_first(ov, oi, b1, a1)) { b1 = ov; a1 = oi; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) { bv[sgi] = b1; bi[sgi] = a1; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float v1 = bv[0];
    uint i1 = bi[0];
    for (int q = 1; q < 32; q++) if (tf_first(bv[q], bi[q], v1, i1)) { v1 = bv[q]; i1 = bi[q]; }
    second[3 * how.x] = map[i1];
    second[3 * how.x + 1] = as_type<uint>(metal::exp(v1 - top_v[0]) / total_v[0]);
    second[3 * how.x + 2] = as_type<uint>(reach_v[0]);
  }
}

// One leaf: the head's guess beside a copy's first row, else its likeliest second choice (reach x chance, scaled) past the bar.
kernel void tf_round_sib(device uint* state [[buffer(0)]],
                         device uint* next_window [[buffer(1)]],
                         device int* args [[buffer(2)]],
                         const device uint* second [[buffer(3)]],
                         constant float4& how [[buffer(4)]],  // bar, the stream's landed over offered chances, rows at most
                         uint t [[thread_position_in_grid]]) {
  if (t != 0) return;
  const uint rows = state[7];
  if (float(rows) >= how.z) return;
  if (state[10] != 0) {
    if (state[S_GUESS] == 0 || second[24] == next_window[1]) return;
    next_window[rows] = second[24];
    args[A_PAR + rows] = 0;
    state[7] = rows + 1;
    state[13] = rows;
    state[14] = second[25];
    state[S_LEAF] = 2;
    return;
  }
  const uint n = rows - 1 - state[12];  // the head's drafts (a tail copy after them)
  float best = 0.0f;
  uint at = 0;
  for (uint j = 0; j < n; j++) {
    const float v = as_type<float>(second[3 * j + 1]) * as_type<float>(second[3 * j + 2]);
    if (v > best) { best = v; at = j; }
  }
  if (n == 0 || best * how.y < how.x) return;
  next_window[rows] = second[3 * at];
  args[A_PAR + rows] = int(at);
  state[7] = rows + 1;
  state[13] = rows;
  state[14] = as_type<uint>(best);
  state[S_LEAF] = 1;
}

// The next window: the continuation after the latest longest match of the context's suffix, when long and priced.
kernel void tf_round_copy(const device uint* context [[buffer(0)]],
                          device uint* state [[buffer(1)]],
                          device uint* next_window [[buffer(2)]],
                          const device uint* tmpl [[buffer(3)]],
                          device uint* live [[buffer(4)]],
                          constant uint4& how [[buffer(5)]],  // copy rows at most (0: none), match needed, level 0's entries, guess below
                          uint t [[thread_position_in_threadgroup]],
                          uint sgi [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint bl[32];
  threadgroup uint be[32];
  threadgroup uint pick[2];
  const uint L = state[0] + 1;
  state[S_GUESS] = 0;
  uint best = 0, at = 0;
  for (uint e = 1 + t; e < L; e += 1024) {
    uint l = 0;
    while (l < 64 && l < e && context[e - 1 - l] == context[L - 1 - l]) l++;
    if (l > best || (l == best && e > at)) { best = l; at = e; }
  }
  for (uint o = 16; o > 0; o >>= 1) {
    const uint ol = simd_shuffle_xor(best, o), oe = simd_shuffle_xor(at, o);
    if (ol > best || (ol == best && oe > at)) { best = ol; at = oe; }
  }
  if (lane == 0) { bl[sgi] = best; be[sgi] = at; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    uint b0 = bl[0], e0 = be[0];
    for (int q = 1; q < 32; q++) if (bl[q] > b0 || (bl[q] == b0 && be[q] > e0)) { b0 = bl[q]; e0 = be[q]; }
    const uint m = (b0 >= how.y && how.x >= 2) ? min(how.x, L - e0) : 0;
    state[11] = b0;
    state[10] = m >= 2 ? m : 0;
    if (m >= 2) {
      state[2] = e0;
      state[S_RESUME + 2] = 1;
      for (uint i = 0; i < m; i++) next_window[1 + i] = context[e0 + i];
      state[7] = 1 + m;
      // a short match: the head's first level runs for a guess beside the copy's first row; else it drafts nothing
      const bool guess = b0 < how.w;
      state[S_GUESS] = guess ? 1 : 0;
      state[9] = guess ? 1 : 0;
      pick[0] = guess ? 0 : 1;
    } else pick[0] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 3 * how.z) live[t] = pick[0] != 0 ? 0 : tmpl[t];
}

// After the head's chain: its drafts continued by a copy when the context with them has a long earlier match.
kernel void tf_round_tail(const device uint* context [[buffer(0)]],
                          device uint* state [[buffer(1)]],
                          device uint* next_window [[buffer(2)]],
                          constant uint4& how [[buffer(3)]],  // rows a window takes at most, match length needed
                          uint t [[thread_position_in_threadgroup]],
                          uint sgi [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint bl[32];
  threadgroup uint be[32];
  const uint rows = state[7];
  if (state[10] != 0 || rows < 2 || rows >= how.x) return;  // a copied window, or no drafts to continue
  const uint L = state[0] + 1, n = rows - 1;  // the context and its n drafts after the pending token
  if (state[S_RESUME + 1] > 0) {
    // after a missed copy: the drafts past a short replacement (j drafts) again the context from near the missed row
    if (t != 0) return;
    const uint anchor = state[S_RESUME] - 1;
    for (uint j = 0; j + 2 <= n; j++) {
      for (uint d = 0; d <= 4; d++) {
        const uint s0 = anchor + d;
        if (s0 + (n - j) >= L) continue;
        bool ok = true;
        for (uint i = 0; i < n - j && ok; i++) ok = next_window[j + 1 + i] == context[s0 + i];
        if (!ok) continue;
        const uint e0 = s0 + (n - j);
        const uint k = min(how.x - rows, L - e0);
        for (uint i = 0; i < k; i++) next_window[rows + i] = context[e0 + i];
        state[2] = e0;
        state[S_RESUME + 2] = rows;
        state[7] = rows + k;
        state[12] = k;
        state[S_RESUME + 1] = 0;
        return;
      }
    }
    return;
  }
  uint best = 0, at = 0;
  for (uint e = 1 + t; e < L; e += 1024) {
    uint l = 0;
    while (l < 64 && l < e && context[e - 1 - l] == (l < n ? next_window[n - l] : context[L - 1 - (l - n)])) l++;
    if (l > best || (l == best && e > at)) { best = l; at = e; }
  }
  for (uint o = 16; o > 0; o >>= 1) {
    const uint ol = simd_shuffle_xor(best, o), oe = simd_shuffle_xor(at, o);
    if (ol > best || (ol == best && oe > at)) { best = ol; at = oe; }
  }
  if (lane == 0) { bl[sgi] = best; be[sgi] = at; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    uint b0 = bl[0], e0 = be[0];
    for (int q = 1; q < 32; q++) if (bl[q] > b0 || (bl[q] == b0 && be[q] > e0)) { b0 = bl[q]; e0 = be[q]; }
    if (b0 < how.y) return;
    const uint k = min(how.x - rows, L - e0);
    for (uint i = 0; i < k; i++) next_window[rows + i] = context[e0 + i];
    state[2] = e0;
    state[S_RESUME + 2] = rows;
    state[7] = rows + k;
    state[12] = k;
  }
}

// The verify's attention path: each layer's chain dispatches when the window is a chain, else its tree dispatches.
kernel void tf_round_attsel(const device uint* state [[buffer(0)]],
                            const device int* args [[buffer(1)]],
                            const device uint* tmpl [[buffer(2)]],
                            device uint* live [[buffer(3)]],
                            constant uint& layers [[buffer(4)]],
                            uint t [[thread_position_in_grid]]) {
  if (t != 0) return;
  const uint rows = state[7];
  bool tree = false;
  for (uint i = 1; i < rows; i++) tree = tree || args[A_PAR + i] != int(i) - 1;
  for (uint a = 0; a < layers; a++)
    for (uint e = 0; e < 5; e++)
      for (uint c = 0; c < 3; c++) live[(8 * a + e) * 3 + c] = (e >= 2) == tree ? tmpl[(8 * a + e) * 3 + c] : 0;
}

// Row state[5] (the last kept row) of the head's residual and q/k/v rows into row 0: the absorbed chain root's.
kernel void tf_round_root(device bfloat* h [[buffer(0)]],
                          device bfloat* qkv [[buffer(1)]],
                          const device uint* state [[buffer(2)]],
                          constant uint2& dims [[buffer(3)]],  // D, the q/k/v row
                          uint i [[thread_position_in_grid]]) {
  const uint r = state[5];
  if (r == 0) return;
  if (i < dims.x) h[i] = h[size_t(r) * dims.x + i];
  if (i < dims.y) qkv[i] = qkv[size_t(r) * dims.y + i];
}

// Hidden row state[5] (the last kept row) of `src` into `dst`.
kernel void tf_round_gather(const device bfloat* src [[buffer(0)]],
                            device bfloat* dst [[buffer(1)]],
                            const device uint* state [[buffer(2)]],
                            constant uint& d [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
  dst[i] = src[size_t(state[5]) * d + i];
}

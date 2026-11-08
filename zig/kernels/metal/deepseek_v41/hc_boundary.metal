// Hyper-Connection boundary: the release's hc_mixes/hc_pre/hc_post contract in two launches, mix partials then split.

// Written from the release's Block.hc_mixes/hc_pre/hc_post and hc_split_sinkhorn_kernel (revision 314113f8).
#include <metal_stdlib>
using namespace metal;

constant constexpr int kCopies = 4;                             // hc_mult
constant constexpr int kWidth = 5120;                          // one copy's width
constant constexpr int kStream = kCopies * kWidth;             // the flattened hc stream
constant constexpr int kMixes = (2 + kCopies) * kCopies;       // pre, post and the comb cells
constant constexpr int kSlices = kStream / 1024;               // the mix projection's column slices
constant constexpr int kParts = kMixes + 1;                    // the mixes and the stream's squares

struct HcArgs { float eps; float hc_eps; int iters; };
static_assert(sizeof(HcArgs) == 12, "HcArgs must mirror mhc.zig's BoundaryArgs");

inline float hc_round(float v) { return float(bfloat(v)); }

// One thread's four adjacent columns of a slice: the stream's squares and its products against the 24 hc_fn rows.
inline void hc_mix_column(const float* v, const device float* fn, int col, float* out) {
  out[kMixes] = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
  for (int q = 0; q < kMixes; q++) {
    const device float* f = fn + size_t(q) * kStream + col;
    out[q] = f[0] * v[0] + f[1] * v[1] + f[2] * v[2] + f[3] * v[3];
  }
}

// A slice's reduction: the warp sums, then the first kParts threads add the eight warps in index order and publish.
inline void hc_publish_slice(threadgroup float* warp, device float* part, const float* m,
                             uint lane, uint sg, uint t, uint r, uint j) {
  for (int q = 0; q <= kMixes; q++) {
    const float s = simd_sum(m[q]);
    if (lane == 0) warp[sg * kParts + q] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(t) <= kMixes) {
    float s = 0.0f;
    for (int k = 0; k < 8; k++) s += warp[k * kParts + int(t)];
    part[(size_t(r) * kSlices + j) * kParts + t] = s;
  }
}

// The mix projection's partials: a threadgroup a (1024-column slice, row), 256 threads, four columns a thread.
[[kernel]] void tf_hc_mix_part(const device bfloat* X [[buffer(0)]], const device float* FN [[buffer(1)]],
                               device float* PART [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint2 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float warp[8 * kParts];
  const int j = int(tg.x), r = int(tg.y);
  const uint t = tpos.x;
  const int col = 1024 * j + 4 * int(t);
  const device bfloat* x = X + size_t(r) * kStream + col;
  float v[4];
  for (int i = 0; i < 4; i++) v[i] = float(x[i]);
  float parts[kParts];
  hc_mix_column(v, FN, col, parts);
  hc_publish_slice(&warp[0], PART, parts, lane, sg, t, uint(r), uint(j));
}

// The pending branch into XO (the release's hc_post) and the same slice partials from the bf16 written.
[[kernel]] void tf_hc_expand_mix(const device bfloat* X [[buffer(0)]], const device bfloat* BR [[buffer(1)]],
                                 const device float* POST [[buffer(2)]], const device float* COMB [[buffer(3)]],
                                 const device float* FN [[buffer(4)]], device float* PART [[buffer(5)]],
                                 device bfloat* XO [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                 uint2 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float warp[8 * kParts];
  const int j = int(tg.x), r = int(tg.y);
  const uint t = tpos.x;
  const int col = 1024 * j + 4 * int(t);
  const int k = col / kWidth, d0 = col % kWidth;
  const device bfloat* x = X + size_t(r) * kStream;
  const device float* comb = COMB + r * kCopies * kCopies;
  float v[4];
  for (int i = 0; i < 4; i++) {
    const int d = d0 + i;
    const float branch = float(BR[size_t(r) * kWidth + d]);
    float stream[kCopies];
    for (int jj = 0; jj < kCopies; jj++) stream[jj] = float(x[jj * kWidth + d]);
    float mixed = 0.0f;
    for (int jj = 0; jj < kCopies; jj++) mixed += comb[jj * kCopies + k] * stream[jj];
    const bfloat out = bfloat(POST[r * kCopies + k] * branch + mixed);
    XO[size_t(r) * kStream + col + i] = out;
    v[i] = float(out);
  }
  float parts[kParts];
  hc_mix_column(v, FN, col, parts);
  hc_publish_slice(&warp[0], PART, parts, lane, sg, t, uint(r), uint(j));
}

// The release's split (kernel.py hc_split_sinkhorn_kernel): lanes 0-3 carry pre and post.
inline void hc_split_row(threadgroup const float* tot, constant const HcArgs& a,
                         const device float* base, const device float* scale, uint lane, uint r,
                         device float* pre_out, device float* post_out, device float* comb_out) {
  const float rstd = precise::rsqrt(tot[kMixes] / float(kStream) + a.eps);
  if (lane < uint(kCopies)) {
    pre_out[r * kCopies + lane] =
        1.0f / (1.0f + precise::exp(-(tot[lane] * rstd * scale[0] + base[lane]))) + a.hc_eps;
    post_out[r * kCopies + lane] =
        2.0f / (1.0f + precise::exp(-(tot[kCopies + lane] * rstd * scale[1] + base[kCopies + lane])));
  }
  const bool live = lane < uint(kCopies * kCopies);
  const uint cell = live ? lane : 0;
  const float z = tot[2 * kCopies + cell] * rstd * scale[2] + base[2 * kCopies + cell];
  float top = max(z, simd_shuffle_xor(z, ushort(1)));
  top = max(top, simd_shuffle_xor(top, ushort(2)));
  const float e = precise::exp(z - top);
  float row = e + simd_shuffle_xor(e, ushort(1));
  row = row + simd_shuffle_xor(row, ushort(2));
  float comb = live ? e / row + a.hc_eps : 0.0f;
  for (uint it = 0; it < uint(a.iters); it++) {
    if (it > 0) {
      float row_sum = comb + simd_shuffle_xor(comb, ushort(1));
      row_sum = row_sum + simd_shuffle_xor(row_sum, ushort(2));
      comb = live ? comb / (row_sum + a.hc_eps) : 0.0f;
    }
    float col_sum = comb + simd_shuffle_xor(comb, ushort(4));
    col_sum = col_sum + simd_shuffle_xor(col_sum, ushort(8));
    comb = live ? comb / (col_sum + a.hc_eps) : 0.0f;
  }
  if (live) comb_out[r * kCopies * kCopies + lane] = comb;
}

// A threadgroup a row, 1024 threads: the split, collapse and norm; MIXES=false consumes a pre made elsewhere.
template <bool MIXES>
[[kernel]] void tf_hc_boundary(const device bfloat* X [[buffer(0)]], const device float* PART [[buffer(1)]],
                               const device float* BASE [[buffer(2)]], const device float* SCALE [[buffer(3)]],
                               const device float* PRE_IN [[buffer(4)]], const device bfloat* NORMW [[buffer(5)]],
                               constant HcArgs& a [[buffer(6)]], device float* PRE_OUT [[buffer(7)]],
                               device float* POST_OUT [[buffer(8)]], device float* COMB_OUT [[buffer(9)]],
                               device bfloat* NORMED [[buffer(10)]], uint r [[threadgroup_position_in_grid]],
                               uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[32];
  threadgroup float slice[kParts];
  threadgroup float pre[kCopies];
  const device bfloat* x = X + size_t(r) * kStream;
  // The slices' totals beside the lagged collapse, which overlaps the norm's reduction.
  if (MIXES && int(t) <= kMixes) {
    float s = 0.0f;
    for (int j = 0; j < kSlices; j++) s += PART[(size_t(r) * kSlices + j) * kParts + t];
    slice[t] = s;
  }
  if (t < uint(kCopies)) pre[t] = PRE_IN[r * kCopies + t];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float y[5];
  float squares = 0.0f;
  for (int i = 0; i < 5; i++) {
    const int d = int(t) + 1024 * i;
    float v = 0.0f;
    for (int s = 0; s < kCopies; s++) v += pre[s] * float(x[s * kWidth + d]);
    y[i] = hc_round(v);
    squares += y[i] * y[i];
  }
  const float lane_squares = simd_sum(squares);
  if (lane == 0) sums[sg] = lane_squares;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (MIXES && sg == 0) hc_split_row(&slice[0], a, BASE, SCALE, lane, r, PRE_OUT, POST_OUT, COMB_OUT);
  if (t == 32) {  // simdgroup 1, while simdgroup 0 runs the split
    float s = 0.0f;
    for (int k = 0; k < 32; k++) s += sums[k];
    pre[0] = precise::rsqrt(s / float(kWidth) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = pre[0];
  for (int i = 0; i < 5; i++) {
    const int d = int(t) + 1024 * i;
    NORMED[size_t(r) * kWidth + d] = bfloat(float(NORMW[d]) * (y[i] * inv));
  }
}

template [[host_name("tf_hc_boundary_mix")]] [[kernel]] decltype(tf_hc_boundary<true>) tf_hc_boundary<true>;
template [[host_name("tf_hc_boundary_plain")]] [[kernel]] decltype(tf_hc_boundary<false>) tf_hc_boundary<false>;

// The launch-overhead probe: a work-free dispatch that reads the boundary's own partial sums and writes them back.
kernel void tf_hc_probe(const device float *part [[buffer(0)]], device float *sink [[buffer(1)]],
                        uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint at = g * (kSlices * kParts) + tid;
  if (tid <= uint(kMixes)) sink[at] = part[at];
}

// The independent form of the probe: each dispatch has its own sink, so the chain carries no memory dependency.
kernel void tf_hc_probe_indep(const device float *part [[buffer(0)]], device float *sink [[buffer(1)]],
                              uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  if (tid <= uint(kMixes)) sink[tid] = part[tid];
}

// DeepSeek V4.1's indexer scores on Metal: FP4xFP4 dots in one stated order, causal and candidate-block masks.
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

#ifndef TF_CTL
#define TF_CTL 0
#endif

// The release's indexer geometry (release-config.json): 32 index heads of 128 dims, candidate blocks of 8 keys.
#define IH 32u
#define ID 128u
#define CBS 8u
#define QROW 2048u // one query row: 32 heads x 128 FP4 values, two a byte
#define QSROW 128u // one query row: 32 heads x 4 per-32-group scales, e8m0
#define IXR 4u // simdgroups a threadgroup carries, each scoring a block of 8 rows
#define IXKT 2u // key fragments (8 keys each) a simdgroup scores together over one block's rows

// Fully unrolled: the matrix arrays below must stay in registers, indexed at compile time.
#define UNR _Pragma("clang loop unroll(full)")

struct TfDsv41IxArgs {
  uint rows, keys, start, ratio, chunk, out_stride, w_stride, ik_stride, mask_stride;
  float w_scale; // softmax_scale * n_heads^-0.5 (model.py:556, 513): scales each bf16 weight once, in fp32
};

// One E2M1 code by its bit fields: 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives (the format has no NaN or infinity).
static inline float tf_ix_e2m1(const uint c) {
  const uint e = (c >> 1) & 3u, m = c & 1u;
  const float v = (e == 0u) ? float(m) * 0.5f : ldexp(float(m) + 2.0f, int(e) - 2);
  return (c & 8u) ? -v : v;
}

// One e8m0 scale: 2^(c-127) from its exponent bits; code 255 is that dtype's NaN, code 0 the subnormal 2^-127.
static inline float tf_ix_e8m0(const uint c) {
  return c == 255u ? as_type<float>(0x7FC00000u) : (c == 0u ? as_type<float>(0x00400000u) : as_type<float>(c << 23));
}

// ReLU keeping a NaN: fmax() folds one to 0, which would hide a broken dot (gate 1's planted control).
static inline float tf_ix_relu(const float x) {
#if TF_CTL == 1
  return fmax(x, 0.0f);
#else
  return (x != x || x > 0.0f) ? x : 0.0f;
#endif
}

// One FP4 code's value as an fp16 (exact; the format has no NaN or infinity), by its code.
constant constexpr half tf_ix_lut[16] = {0.0h, 0.5h, 1.0h, 1.5h, 2.0h, 3.0h, 4.0h, 6.0h,
                                         -0.0h, -0.5h, -1.0h, -1.5h, -2.0h, -3.0h, -4.0h, -6.0h};

// Scores 8 rows on fp16 MMA, fp32 sums: FP4 codes are exact in fp16 and a group sum exact in any order.
[[kernel, max_total_threads_per_threadgroup(32 * IXR)]] void tf_dsv41_ix_scores(
    const device uchar* Q [[buffer(0)]], const device uchar* QS [[buffer(1)]], const device bfloat* W [[buffer(2)]],
    const device uchar* IK [[buffer(3)]], device float* S [[buffer(4)]], const device uchar* CM [[buffer(5)]],
    constant TfDsv41IxArgs& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint r0 = (tg.y * IXR + sg) * 8u;
  if (r0 >= a.rows) return; // uniform across the simdgroup, which has no barrier with the others
  const uint qid = lane / 4;
  const uint fm = (qid & 4u) + ((lane / 2u) % 4u);
  const uint fn = (qid & 2u) * 2u + (lane % 2u) * 2u;
  const bool row_ok = r0 + fm < a.rows;
  const uint row = row_ok ? r0 + fm : a.rows - 1u; // the lane's query row (an MMA row); a tail's lanes repeat the last
  // keys a query reaches: (start + row) / ratio (model.py:565 prefill with start 1, 568 decode with end_pos)
  const uint visible = (a.start + row) / a.ratio;
  const uint vmax = (a.start + min(r0 + 8u, a.rows) - 1u) / a.ratio; // the block's last row reaches the most
  const uint first = tg.x * a.chunk;
  const uint last = min(first + a.chunk, a.keys);
  const device uchar* qrow = Q + size_t(row) * QROW + fn / 2u;
  for (uint key0 = first; key0 < last; key0 += IXKT * 8u) {
    float sacc[IXKT][2];
    UNR for (uint kf = 0; kf < IXKT; ++kf) sacc[kf][0] = sacc[kf][1] = 0.0f;
    if (key0 < vmax) {
      simdgroup_matrix<half, 8, 8> bq[IXKT][16];
      float sk[IXKT][2][4];
      UNR for (uint kf = 0; kf < IXKT; ++kf) {
        const uint ka = min(key0 + 8u * kf + fn, a.keys - 1u), kb = min(ka + 1u, a.keys - 1u);
        const device uchar* ra = IK + size_t(ka) * a.ik_stride;
        const device uchar* rb = IK + size_t(kb) * a.ik_stride;
        UNR for (uint ks = 0; ks < 16u; ++ks) {
          const uint d = 8u * ks + fm;
          bq[kf][ks].thread_elements()[0] = tf_ix_lut[(ra[d >> 1] >> (4u * (d & 1u))) & 0xFu];
          bq[kf][ks].thread_elements()[1] = tf_ix_lut[(rb[d >> 1] >> (4u * (d & 1u))) & 0xFu];
        }
        UNR for (uint g = 0; g < 4u; ++g) { // a key's row: 64 FP4 bytes then its 4 e8m0 scales
          sk[kf][0][g] = tf_ix_e8m0(ra[ID / 2u + g]);
          sk[kf][1][g] = tf_ix_e8m0(rb[ID / 2u + g]);
        }
      }
#if TF_CTL == 2 // gate 3's control: the head fold in reverse, and head 0 (the last one folded) dropped
      for (uint h = IH - 1u; h >= 1u; --h) {
#else
      for (uint h = 0u; h < IH; ++h) {
#endif
        simdgroup_matrix<half, 8, 8> aq[16];
        UNR for (uint ks = 0; ks < 16u; ++ks) {
          const uchar b = qrow[h * (ID / 2u) + 4u * ks];
          aq[ks].thread_elements()[0] = tf_ix_lut[b & 0xFu];
          aq[ks].thread_elements()[1] = tf_ix_lut[b >> 4];
        }
        float qs[4];
        UNR for (uint g = 0; g < 4u; ++g) qs[g] = tf_ix_e8m0(QS[size_t(row) * QSROW + h * 4u + g]);
        // the row's weights times the softmax scale once, in fp32 (model.py:556), before any head meets a score
        const float wv = float(W[size_t(row) * a.w_stride + h]) * a.w_scale;
        simdgroup_matrix<float, 8, 8> acc[IXKT][4];
        UNR for (uint kf = 0; kf < IXKT; ++kf)
          UNR for (uint g = 0; g < 4u; ++g) acc[kf][g] = simdgroup_matrix<float, 8, 8>(0.0f);
        UNR for (uint kk = 0; kk < 4u; ++kk)
          UNR for (uint kf = 0; kf < IXKT; ++kf)
            UNR for (uint g = 0; g < 4u; ++g) {
              simdgroup_multiply_accumulate(acc[kf][g], aq[4u * g + kk], bq[kf][4u * g + kk], acc[kf][g]);
            }
        UNR for (uint kf = 0; kf < IXKT; ++kf)
          UNR for (uint e = 0; e < 2u; ++e) {
            float p[4];
            UNR for (uint g = 0; g < 4u; ++g) {
              p[g] = acc[kf][g].thread_elements()[e];
#if TF_CTL == 3 // gate 4's control: the group sums accumulated in half
              p[g] = float(half(p[g]));
#endif
            }
            const float prod = (((p[0] * qs[0]) * sk[kf][e][0]) + ((p[1] * qs[1]) * sk[kf][e][1])) +
                               (((p[2] * qs[2]) * sk[kf][e][2]) + ((p[3] * qs[3]) * sk[kf][e][3]));
            sacc[kf][e] += tf_ix_relu(prod) * wv;
          }
      }
    }
    if (!row_ok) continue;
    UNR for (uint kf = 0; kf < IXKT; ++kf)
      UNR for (uint e = 0; e < 2u; ++e) {
        const uint key = key0 + 8u * kf + fn + e;
        if (key >= last) continue;
        // keys a row reaches are still masked by the candidate blocks
        const bool live = key < visible && (a.mask_stride == 0u || CM[size_t(row) * a.mask_stride + key / CBS] != 0u);
#if TF_CTL == 4 // gate 2's control: one float past the LAST output, into the guard bytes
        if (row == a.rows - 1u && key == a.keys - 1u) S[size_t(a.rows) * a.out_stride] = 1.0f;
#endif
        S[size_t(row) * a.out_stride + key] = live ? sacc[kf][e] : -INFINITY; // model.py:566, 576 mask unreachable keys
      }
  }
}

#define TF_IX_QPAD 129u // a head's floats in threadgroup memory: one more than ID, so 32 lanes read 32 banks

// A query row's head h, dim d: the nibble's value times its group's power of two (both exact in fp32).
static inline float tf_ix_q(const device uchar* Q, const device uchar* QS, const size_t at, const uint h, const uint d) {
  return tf_ix_e2m1((Q[at + h * (ID / 2u) + (d >> 1)] >> (4u * (d & 1u))) & 0xFu) * tf_ix_e8m0(QS[at / QROW * QSROW + h * 4u + (d >> 5)]);
}

// Decode's few rows: eight simdgroups share a row's decoded query and score a key each in turn, the same bits as above.
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_ix_scores_dec(
    const device uchar* Q [[buffer(0)]], const device uchar* QS [[buffer(1)]], const device bfloat* W [[buffer(2)]],
    const device uchar* IK [[buffer(3)]], device float* S [[buffer(4)]], const device uchar* CM [[buffer(5)]],
    constant TfDsv41IxArgs& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint sgs [[simdgroups_per_threadgroup]]) {
  const uint row = tg.y;
  if (row >= a.rows) return;
  threadgroup float qf[IH * TF_IX_QPAD]; // 16 KiB: this row's query, decoded once for the whole chunk
  threadgroup float wf[IH];
  const size_t at = size_t(row) * QROW;
  for (uint i = t; i < IH * ID; i += sgs * 32u) qf[(i / ID) * TF_IX_QPAD + i % ID] = tf_ix_q(Q, QS, at, i / ID, i % ID);
  // the row's weights times the softmax scale once, in fp32 (model.py:556), before any head meets a score
  if (t < IH) wf[t] = float(W[size_t(row) * a.w_stride + t]) * a.w_scale;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint first = tg.x * a.chunk;
  const uint last = min(first + a.chunk, a.keys);
  // keys a query reaches: (start + row) / ratio (model.py:565 prefill with start 1, 568 decode with end_pos)
  const uint visible = (a.start + row) / a.ratio;
  const threadgroup float* qh = qf + size_t(lane) * TF_IX_QPAD;
  const float wh = wf[lane];
  for (uint key = first + sg; key < last; key += sgs) {
#if TF_CTL == 4 // gate 2's control: one float past the LAST output, into the guard bytes (lane 0, before any mask)
    if (lane == 0u && row == a.rows - 1u && key == a.keys - 1u) S[size_t(a.rows) * a.out_stride] = 1.0f;
#endif
    const bool live = key < visible && (a.mask_stride == 0u || CM[size_t(row) * a.mask_stride + key / CBS] != 0u);
    if (!live) {
      if (lane == 0u) S[size_t(row) * a.out_stride + key] = -INFINITY; // model.py:566, 576 mask unreachable keys
      continue;
    }
    const device uchar* ik = IK + size_t(key) * a.ik_stride;
    const device uchar* ks = ik + ID / 2u; // a key's row: 64 FP4 bytes then its 4 e8m0 scales
    float p[4];
    for (uint q = 0; q < 4u; ++q) {
#if TF_CTL == 3 // gate 4's control: the group sums accumulated in half
      half h = 0.0h;
      for (uint i = 32u * q; i < 32u * q + 32u; ++i) h += half(qh[i]) * half(tf_ix_e2m1((ik[i >> 1] >> (4u * (i & 1u))) & 0xFu));
      p[q] = float(h);
#else
      float acc = 0.0f;
      for (uint i = 32u * q; i < 32u * q + 32u; ++i) acc += qh[i] * tf_ix_e2m1((ik[i >> 1] >> (4u * (i & 1u))) & 0xFu);
      p[q] = acc;
#endif
    }
    const float prod = ((p[0] * tf_ix_e8m0(ks[0])) + (p[1] * tf_ix_e8m0(ks[1]))) + ((p[2] * tf_ix_e8m0(ks[2])) + (p[3] * tf_ix_e8m0(ks[3])));
    const float term = tf_ix_relu(prod) * wh;
    float acc = 0.0f;
#if TF_CTL == 2 // gate 3's control: the head fold in reverse, and head 0 (the last one folded) dropped
    for (uint h = IH - 1u; h >= 1u; --h) acc += simd_shuffle(term, ushort(h));
#else
    for (uint h = 0u; h < IH; ++h) acc += simd_shuffle(term, ushort(h));
#endif
    if (lane == 0u) S[size_t(row) * a.out_stride + key] = acc;
  }
}

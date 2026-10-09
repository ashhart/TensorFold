// Prompt rows' attention in absolute 512-key chunks: a row's bits never depend on how its prompt was cut.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
constant constexpr int PG = 6, PD = 256, PH = 128, PCK = 512, PTK = 32;
struct Q27PromptArgs { int start, rows, chunks, rp, key_stride, value_stride; float scale2; };
typedef vec<float, 8> q27f8;
typedef vec<bfloat, 8> q27b8;
typedef vec<half, 8> q27h8;

// Rows fm and fm+8, columns fn..fn+3 of a 16-row block; past `limit` at a chunk's edge, rows read zero.
template <bool EDGE>
inline q27b8 q27_frag(const device bfloat* p, int stride, short fm, short fn, int row, int limit) {
  const device bfloat* r0 = p + fm * stride + fn;
  if (!EDGE) return q27b8(*(const device vec<bfloat, 4>*)r0, *(const device vec<bfloat, 4>*)(r0 + 8 * stride));
  const vec<bfloat, 4> z = vec<bfloat, 4>(0);
  return q27b8(row + fm < limit ? *(const device vec<bfloat, 4>*)r0 : z, row + fm + 8 < limit ? *(const device vec<bfloat, 4>*)(r0 + 8 * stride) : z);
}

// S[16][32] += Q[16][16] K[32][16]^T, fp32 sums.
inline void q27_mma_s(thread q27f8& c0, thread q27f8& c1, q27b8 a, q27b8 b0, q27b8 b1) {
  constexpr auto d = matmul2d_descriptor(16, 32, 16, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<d, execution_simdgroup> op;
  auto ca = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto cb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto cc = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
  _Pragma("clang loop unroll(full)") // a runtime index into a cooperative tensor is several times slower
  for (int i = 0; i < 8; i++) { ca[i] = a[i]; cb[i] = b0[i]; cb[8 + i] = b1[i]; cc[i] = c0[i]; cc[8 + i] = c1[i]; }
  op.run(ca, cb, cc);
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < 8; i++) { c0[i] = cc[i]; c1[i] = cc[8 + i]; }
}

// O[16][32] += P[16][16] V[16][32], half P and fp32 sums.
inline void q27_mma_o(thread q27f8& c0, thread q27f8& c1, q27h8 a, q27b8 b0, q27b8 b1) {
  constexpr auto d = matmul2d_descriptor(16, 32, 16, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<d, execution_simdgroup> op;
  auto ca = op.template get_left_input_cooperative_tensor<half, bfloat, float>();
  auto cb = op.template get_right_input_cooperative_tensor<half, bfloat, float>();
  auto cc = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < 8; i++) { ca[i] = a[i]; cb[i] = b0[i]; cb[8 + i] = b1[i]; cc[i] = c0[i]; cc[8 + i] = c1[i]; }
  op.run(ca, cb, cc);
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < 8; i++) { c0[i] = cc[i]; c1[i] = cc[8 + i]; }
}

// Running state of one tile half: O over its 128 columns, and each row's max and sum (log2 domain).
struct Q27PromptRun { q27f8 O[8]; float m0, m1, l0, l1; };

// One 32-key step at `kt`: S halves summed low first, masked by each row's key count, then softmax and O += PV.
template <bool EDGE>
inline void q27_prompt_step(thread Q27PromptRun& r, int kt, int kend, const device bfloat* q, const device bfloat* k, const device bfloat* v,
    threadgroup float (*xs)[32][16], ushort sg, ushort lane, short fm, short fn, short h, int n0, int n1, float scale2) {
  q27f8 s0 = 0.0f, s1 = 0.0f;
  _Pragma("clang loop unroll_count(4)") // whole unrolling hoists every K load ahead of the products and runs slower
  for (int d = 0; d < 8; d++) q27_mma_s(s0, s1, q27_frag<false>(q + d * 16, PD, fm, fn, 0, 16), q27_frag<EDGE>(k + kt * PD + d * 16, PD, fm, fn, kt, kend), q27_frag<EDGE>(k + (kt + 16) * PD + d * 16, PD, fm, fn, kt + 16, kend));
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < 8; i++) { xs[sg][lane][i] = s0[i]; xs[sg][lane][8 + i] = s1[i]; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < 8; i++) {
    const float o0 = xs[sg ^ 1][lane][i], o1 = xs[sg ^ 1][lane][8 + i];
    s0[i] = h ? o0 + s0[i] : s0[i] + o0;
    s1[i] = h ? o1 + s1[i] : s1[i] + o1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float x0 = -INFINITY, x1 = -INFINITY;
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < 4; i++) {
    const int key = kt + fn + i;
    s0[i] = key < n0 ? s0[i] * scale2 : -INFINITY;
    s1[i] = key + 16 < n0 ? s1[i] * scale2 : -INFINITY;
    s0[4 + i] = key < n1 ? s0[4 + i] * scale2 : -INFINITY;
    s1[4 + i] = key + 16 < n1 ? s1[4 + i] * scale2 : -INFINITY;
    x0 = max(x0, max(s0[i], s1[i]));
    x1 = max(x1, max(s0[4 + i], s1[4 + i]));
  }
  x0 = max(x0, simd_shuffle_xor(x0, 1)); x0 = max(x0, simd_shuffle_xor(x0, 8));
  x1 = max(x1, simd_shuffle_xor(x1, 1)); x1 = max(x1, simd_shuffle_xor(x1, 8));
  const float nm0 = max(r.m0, x0), nm1 = max(r.m1, x1);
  const float f0 = (x0 == -INFINITY) ? 1.0f : fast::exp2(r.m0 - nm0);
  const float f1 = (x1 == -INFINITY) ? 1.0f : fast::exp2(r.m1 - nm1);
  q27h8 p0, p1;
  float y0 = 0.0f, y1 = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < 4; i++) {
    const float a0 = s0[i] == -INFINITY ? 0.0f : fast::exp2(s0[i] - nm0), a1 = s1[i] == -INFINITY ? 0.0f : fast::exp2(s1[i] - nm0);
    const float b0 = s0[4 + i] == -INFINITY ? 0.0f : fast::exp2(s0[4 + i] - nm1), b1 = s1[4 + i] == -INFINITY ? 0.0f : fast::exp2(s1[4 + i] - nm1);
    p0[i] = half(a0); p1[i] = half(a1); p0[4 + i] = half(b0); p1[4 + i] = half(b1);
    y0 += a0 + a1;
    y1 += b0 + b1;
  }
  y0 += simd_shuffle_xor(y0, 1); y0 += simd_shuffle_xor(y0, 8);
  y1 += simd_shuffle_xor(y1, 1); y1 += simd_shuffle_xor(y1, 8);
  if (x0 != -INFINITY) { r.l0 = r.l0 * f0 + y0; r.m0 = nm0; }
  if (x1 != -INFINITY) { r.l1 = r.l1 * f1 + y1; r.m1 = nm1; }
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 8; j++) for (int i = 0; i < 4; i++) { r.O[j][i] *= f0; r.O[j][4 + i] *= f1; }
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 8; j += 2) {
    q27_mma_o(r.O[j], r.O[j + 1], p0, q27_frag<EDGE>(v + kt * PD + j * 16, PD, fm, fn, kt, kend), q27_frag<EDGE>(v + kt * PD + (j + 1) * 16, PD, fm, fn, kt, kend));
    q27_mma_o(r.O[j], r.O[j + 1], p1, q27_frag<EDGE>(v + (kt + 16) * PD + j * 16, PD, fm, fn, kt + 16, kend), q27_frag<EDGE>(v + (kt + 16) * PD + (j + 1) * 16, PD, fm, fn, kt + 16, kend));
  }
}

// One chunk's partial per packed row: a simdgroup pair halves the head dim; only the last step reads bounded.
template <int SG>
kernel void q27_attn_prompt(device const bfloat* QA [[buffer(0)]], device const Q27AttnReadPtrs* caches [[buffer(1)]], constant Q27PromptArgs& args [[buffer(2)]],
    device float* PO [[buffer(3)]], device float* PM [[buffer(4)]], device float* PL [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
  const short qid = lane >> 2, fm = (qid & 4) | ((lane >> 1) & 3), fn = ((qid & 2) | (lane & 1)) * 4;
  const short h = sg & 1;
  const int tile = int(tg.z) * (SG / 2) + (sg >> 1);
  const int hk = int(tg.x), c = int(tg.y);
  const bool live = tile * 16 < args.rp;
  threadgroup float xs[SG][32][16];
  const int r0 = tile * 16 + fm, r1 = r0 + 8;
  const int n0 = live && r0 < args.rows * PG ? args.start + r0 / PG + 1 : 0;
  const int n1 = live && r1 < args.rows * PG ? args.start + r1 / PG + 1 : 0;
  const int kbeg = c * PCK, kend = min(kbeg + PCK, args.start + args.rows);
  const device bfloat* q = QA + ((int64_t)hk * args.rp + (live ? tile * 16 : 0)) * PD + h * PH;
  const device bfloat* k = caches[0].keys + (int64_t)hk * args.key_stride + h * PH;
  const device bfloat* v = caches[0].values + (int64_t)hk * args.value_stride + h * PH;
  Q27PromptRun r;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 8; j++) r.O[j] = 0.0f;
  r.m0 = -INFINITY; r.m1 = -INFINITY; r.l0 = 0.0f; r.l1 = 0.0f;
  int kt = kbeg;
  for (; kt + PTK <= kend; kt += PTK) q27_prompt_step<false>(r, kt, kend, q, k, v, xs, sg, lane, fm, fn, h, n0, n1, args.scale2);
  if (kt < kend) q27_prompt_step<true>(r, kt, kend, q, k, v, xs, sg, lane, fm, fn, h, n0, n1, args.scale2);
  if (!live) return;
  const int64_t base = ((int64_t)hk * args.chunks + c) * args.rp + tile * 16;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 8; j++) {
    device float* dst = PO + (base + fm) * PD + h * PH + j * 16 + fn;
    *(device float4*)dst = float4(r.O[j][0], r.O[j][1], r.O[j][2], r.O[j][3]);
    *(device float4*)(dst + 8 * PD) = float4(r.O[j][4], r.O[j][5], r.O[j][6], r.O[j][7]);
  }
  if (h == 0 && (lane & 9) == 0) {
    PM[base + fm] = r.m0; PL[base + fm] = r.l0;
    PM[base + fm + 8] = r.m1; PL[base + fm + 8] = r.l1;
  }
}
#define Q27_PROMPT(SG) template [[host_name("q27_attn_prompt_" #SG)]] kernel void q27_attn_prompt<SG>(device const bfloat*, device const Q27AttnReadPtrs*, constant Q27PromptArgs&, device float*, device float*, device float*, uint3, ushort, ushort);
Q27_PROMPT(8)

// A packed row's chunks merged in key order, normalized and written as the row's head output.
kernel void q27_attn_prompt_merge(device const float* PO [[buffer(0)]], device const float* PM [[buffer(1)]], device const float* PL [[buffer(2)]],
    constant Q27PromptArgs& args [[buffer(3)]], device bfloat* OUT [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const int hk = int(tg.x), r = int(tg.y);
  constexpr int DP = PD / 32;
  float m = -INFINITY, l = 0.0f, o[DP];
  for (int i = 0; i < DP; i++) o[i] = 0.0f;
  for (int c = 0; c < args.chunks; c++) {
    const int64_t row = ((int64_t)hk * args.chunks + c) * args.rp + r;
    const float mc = PM[row];
    if (mc == -INFINITY) continue;
    const float lc = PL[row];
    const float nm = max(m, mc);
    const float f1 = fast::exp2(m - nm), f2 = fast::exp2(mc - nm);
    l = l * f1 + lc * f2;
    for (int i = 0; i < DP; i++) o[i] = o[i] * f1 + PO[row * PD + lane * DP + i] * f2;
    m = nm;
  }
  const int node = r / PG, head = hk * PG + r % PG;
  for (int i = 0; i < DP; i++) OUT[((int64_t)node * (PG * 4) + head) * PD + lane * DP + i] = static_cast<bfloat>(o[i] / l);
}

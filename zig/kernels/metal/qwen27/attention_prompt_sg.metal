// Prompt rows' attention before the M5: attention_prompt.metal's 512-key chunks and merge in simdgroup matrices.
typedef simdgroup_matrix<float, 8, 8> q27_sgf;
typedef simdgroup_matrix<bfloat, 8, 8> q27_sgb;

// Running state of one tile half: O over its 128 columns (rows fm, then fm + 8), each row's max and sum (log2 domain).
struct Q27SgRun { q27_sgf O[2][16]; float m0, m1, l0, l1; };

// K^T's 8 dims by 8 keys from `key`; at a chunk's edge, keys from `kend` read zero.
template <bool EDGE>
inline q27_sgb q27_sg_keys(const device bfloat* k, int key, int kend, short fm, short fn) {
  q27_sgb b;
  if (!EDGE) {
    simdgroup_load(b, k + (int64_t)key * PD, PD, ulong2(0, 0), true);
    return b;
  }
  const int k0 = key + fn, k1 = k0 + 1;  // this lane's elements: dim fm, keys fn and fn + 1
  b.thread_elements()[0] = k0 < kend ? k[(int64_t)k0 * PD + fm] : bfloat(0.0f);
  b.thread_elements()[1] = k1 < kend ? k[(int64_t)k1 * PD + fm] : bfloat(0.0f);
  return b;
}

// V's 8 keys by 8 dims from `key` in fp32 (exact), keys from `kend` zero; under `mask` a non-finite value reads as 0.
template <bool EDGE, bool MASKED>
inline q27_sgf q27_sg_values(const device bfloat* v, int key, int kend, short fm, short fn, uint mask) {
  q27_sgf b;
  const int row = key + fm;  // this lane's elements: key fm, dims fn and fn + 1
  uint w = (!EDGE || row < kend) ? *(const device uint*)(v + (int64_t)row * PD + fn) : 0u;
  if (MASKED) w &= ~(((q27_inf2(w) & mask) >> 15) * 0x7FFFu);  // a non-finite half keeps its sign bit alone
  b.thread_elements()[0] = as_type<float>(w << 16);
  b.thread_elements()[1] = as_type<float>(w & 0xFFFF0000u);
  return b;
}

// Adds the step's non-finite values (zeros in its P V) times P where a row sees the key; P passes through xs.
template <bool EDGE>
inline void q27_sg_unmask(thread Q27SgRun& r, thread const q27_sgf (&P)[2][4], int kt, int kend, const device bfloat* v,
    threadgroup float (*xs)[32][16], ushort sg, ushort lane, short fm, short fn, int n0, int n1) {
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < 8; i++) { xs[sg][lane][i] = P[0][i / 2].thread_elements()[i % 2]; xs[sg][lane][8 + i] = P[1][i / 2].thread_elements()[i % 2]; }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const int keys = EDGE ? min(PTK, kend - kt) : PTK;
  for (int d = 0; d < keys; d++) {
    const int f = d & 6, i = (d >> 3) * 2 + (d & 1);
    const ushort from = ((fm & 4) << 2) | ((f & 4) << 1) | ((fm & 3) << 1) | ((f >> 1) & 1);
    const float a = xs[sg][from][i], b = xs[sg][from][8 + i];
    const int key = kt + d;
    const device bfloat* row = v + (int64_t)key * PD + fn;
    _Pragma("clang loop unroll(full)")
    for (int e = 0; e < 16; e++)
      for (int x = 0; x < 2; x++) {
        const float val = float(row[8 * e + x]);
        if (isfinite(val)) continue;
        if (key < n0) r.O[0][e].thread_elements()[x] += a * val;
        if (key < n1) r.O[1][e].thread_elements()[x] += b * val;
      }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
}

// One 32-key step at `kt`: S halves summed low first, masked by each row's key count, then softmax and O += PV.
template <bool EDGE, bool MASKED>
inline void q27_sg_step(thread Q27SgRun& r, int kt, int kend, const device bfloat* q, const device bfloat* k, const device bfloat* v,
    threadgroup float (*xs)[32][16], ushort sg, ushort lane, short fm, short fn, short h, int n0, int n1, float scale2, uint mask) {
  q27_sgf S[2][4];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 4; j++) { S[0][j] = q27_sgf(0.0f); S[1][j] = q27_sgf(0.0f); }
  for (int d = 0; d < PH / 8; d++) {
    q27_sgb a0, a1;
    simdgroup_load(a0, q + d * 8, PD);
    simdgroup_load(a1, q + 8 * PD + d * 8, PD);
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < 4; j++) {
      const q27_sgb b = q27_sg_keys<EDGE>(k + d * 8, kt + 8 * j, kend, fm, fn);
      simdgroup_multiply_accumulate(S[0][j], a0, b, S[0][j]);
      simdgroup_multiply_accumulate(S[1][j], a1, b, S[1][j]);
    }
  }
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 4; j++)
    for (int e = 0; e < 2; e++) { xs[sg][lane][2 * j + e] = S[0][j].thread_elements()[e]; xs[sg][lane][8 + 2 * j + e] = S[1][j].thread_elements()[e]; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float s0[8], s1[8];  // rows fm and fm + 8; key kt + 8 (i / 2) + fn + i % 2 at i
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < 8; i++) {
    const float a = S[0][i / 2].thread_elements()[i % 2], b = S[1][i / 2].thread_elements()[i % 2];
    const float o0 = xs[sg ^ 1][lane][i], o1 = xs[sg ^ 1][lane][8 + i];
    s0[i] = h ? o0 + a : a + o0;
    s1[i] = h ? o1 + b : b + o1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float x0 = -INFINITY, x1 = -INFINITY;
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < 8; i++) {
    const int key = kt + 8 * (i / 2) + fn + i % 2;
    s0[i] = key < n0 ? s0[i] * scale2 : -INFINITY;
    s1[i] = key < n1 ? s1[i] * scale2 : -INFINITY;
    x0 = max(x0, s0[i]);
    x1 = max(x1, s1[i]);
  }
  x0 = max(x0, simd_shuffle_xor(x0, 1)); x0 = max(x0, simd_shuffle_xor(x0, 8));
  x1 = max(x1, simd_shuffle_xor(x1, 1)); x1 = max(x1, simd_shuffle_xor(x1, 8));
  const float nm0 = max(r.m0, x0), nm1 = max(r.m1, x1);
  const float f0 = (x0 == -INFINITY) ? 1.0f : fast::exp2(r.m0 - nm0);
  const float f1 = (x1 == -INFINITY) ? 1.0f : fast::exp2(r.m1 - nm1);
  q27_sgf P[2][4];
  float y0 = 0.0f, y1 = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < 8; i++) {
    const float a = s0[i] == -INFINITY ? 0.0f : fast::exp2(s0[i] - nm0);
    const float b = s1[i] == -INFINITY ? 0.0f : fast::exp2(s1[i] - nm1);
    P[0][i / 2].thread_elements()[i % 2] = a;
    P[1][i / 2].thread_elements()[i % 2] = b;
    y0 += a;
    y1 += b;
  }
  y0 += simd_shuffle_xor(y0, 1); y0 += simd_shuffle_xor(y0, 8);
  y1 += simd_shuffle_xor(y1, 1); y1 += simd_shuffle_xor(y1, 8);
  if (x0 != -INFINITY) { r.l0 = r.l0 * f0 + y0; r.m0 = nm0; }
  if (x1 != -INFINITY) { r.l1 = r.l1 * f1 + y1; r.m1 = nm1; }
  _Pragma("clang loop unroll(full)")
  for (int e = 0; e < 16; e++)
    for (int i = 0; i < 2; i++) { r.O[0][e].thread_elements()[i] *= f0; r.O[1][e].thread_elements()[i] *= f1; }
  for (int j = 0; j < 4; j++) {
    _Pragma("clang loop unroll(full)")
    for (int e = 0; e < 16; e++) {
      const q27_sgf b = q27_sg_values<EDGE, MASKED>(v + 8 * e, kt + 8 * j, kend, fm, fn, mask);
      simdgroup_multiply_accumulate(r.O[0][e], P[0][j], b, r.O[0][e]);
      simdgroup_multiply_accumulate(r.O[1][e], P[1][j], b, r.O[1][e]);
    }
  }
  if (MASKED && mask != 0) q27_sg_unmask<EDGE>(r, P, kt, kend, v, xs, sg, lane, fm, fn, n0, n1);
}

// One chunk's partial per packed row, in attention_prompt.metal's layout for q27_attn_prompt_merge.
template <int SG>
kernel void q27_attn_prompt_sg(device const bfloat* QA [[buffer(0)]], device const Q27AttnReadPtrs* caches [[buffer(1)]], constant Q27PromptArgs& args [[buffer(2)]],
    device float* PO [[buffer(3)]], device float* PM [[buffer(4)]], device float* PL [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
  const short qid = lane >> 2, fm = (qid & 4) | ((lane >> 1) & 3), fn = (qid & 2) * 2 + (lane & 1) * 2;
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
  Q27SgRun r;
  _Pragma("clang loop unroll(full)")
  for (int e = 0; e < 16; e++) { r.O[0][e] = q27_sgf(0.0f); r.O[1][e] = q27_sgf(0.0f); }
  r.m0 = -INFINITY; r.m1 = -INFINITY; r.l0 = 0.0f; r.l1 = 0.0f;
  // 0 times a non-finite value is NaN: if one sits past `seen` (fewest keys a live row sees), mask it and unmask after
  const int seen = simd_min(min(n0 ? n0 : INT_MAX, n1 ? n1 : INT_MAX));
  const uint mask = q27_nonfinite(v, max(seen, kbeg), kend, lane) ? 0x80008000u : 0u;
  int kt = kbeg;
  const int seen_by_all = kbeg + max(min(seen, kend) - kbeg, 0) / PTK * PTK;  // steps below it: every live row sees all 32 keys
  for (; kt < seen_by_all; kt += PTK) q27_sg_step<false, false>(r, kt, kend, q, k, v, xs, sg, lane, fm, fn, h, n0, n1, args.scale2, 0);
  for (; kt < kend; kt += PTK) q27_sg_step<true, true>(r, kt, kend, q, k, v, xs, sg, lane, fm, fn, h, n0, n1, args.scale2, mask);
  if (!live) return;
  const int64_t base = ((int64_t)hk * args.chunks + c) * args.rp + tile * 16;
  _Pragma("clang loop unroll(full)")
  for (int e = 0; e < 16; e++) {
    simdgroup_store(r.O[0][e], PO + base * PD + h * PH + 8 * e, PD);
    simdgroup_store(r.O[1][e], PO + (base + 8) * PD + h * PH + 8 * e, PD);
  }
  if (h == 0 && (lane & 9) == 0) {
    PM[base + fm] = r.m0; PL[base + fm] = r.l0;
    PM[base + fm + 8] = r.m1; PL[base + fm + 8] = r.l1;
  }
}
#define Q27_PROMPT_SG(SG) template [[host_name("q27_attn_prompt_sg_" #SG)]] kernel void q27_attn_prompt_sg<SG>(device const bfloat*, device const Q27AttnReadPtrs*, constant Q27PromptArgs&, device float*, device float*, device float*, uint3, ushort, ushort);
Q27_PROMPT_SG(8)

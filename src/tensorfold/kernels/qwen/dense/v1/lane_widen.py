"""The lane matmul for 2-, 3-, 5-, 6- and 8-bit weights: each group widened exactly, then ``lane_qmm._MAIN``'s op."""

from __future__ import annotations

# 3- and 2-bit weights to nibbles for the bf16 x uint4 op, lane l widening column n0 + l (TILED: tile_weight's layout)
NIBBLES = r"""
  static_assert(NT == 32, "one column per lane");
  static_assert(BITS == 2 || BITS == 3, "4-bit weights go to the tensor op as they are");
  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;     // K slice
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  const int M = mdims[0], MP = mdims[1];
  constexpr int KG = K / 64;
  constexpr int NF = NT / 16;
  constexpr int WPG = 2 * BITS;                          // words per column per group: 64 values x BITS bits
  constexpr int KW = K * BITS / 32;                      // words per column
  const int n0 = threadgroup_position_in_grid.x * NT;
  const int rb = threadgroup_position_in_grid.y * 16 * TMR;
  const int g_begin = (sg * KG) / SK;
  const int g_end = ((sg + 1) * KG) / SK;
  constexpr auto desc = matmul2d_descriptor(16 * TMR, NT, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));
  threadgroup uint stage_all[SK * NT * 8];               // per K slice: NT columns x 64 nibbles
  threadgroup uint* stage = stage_all + sg * NT * 8;
  tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline> b((threadgroup uchar*)stage, dextents<int32_t, 2>(64, NT));
  const device uint* Wv = (const device uint*)Wq;
  const int n = n0 + lane;

  float C[TMR][NF * 8];
  for (int t = 0; t < TMR; t++) for (int i = 0; i < NF * 8; i++) C[t][i] = 0.0f;
  const device uint4* sbv = (const device uint4*)SBt;
  bool colok[NF];
  for (int f = 0; f < NF; f++) colok[f] = n0 + f * 16 + fn < N;
  for (int g = g_begin; g < g_end; g++) {
    uint w[WPG + 1];
    for (int i = 0; i <= WPG; i++) w[i] = 0;
    if (n < N) {
      const device uint* src = TILED ? Wv + ((int64_t)(threadgroup_position_in_grid.x * KG + g) * NT + lane) * WPG
                                     : Wv + (int64_t)n * KW + g * WPG;
      for (int i = 0; i < WPG; i++) w[i] = src[i];
    }
    if (BITS == 3) {
      for (int c = 0; c < 8; c++) {
        const int bit = 24 * c, i = bit >> 5, sh = bit & 31;
        uint pack = w[i] >> sh;
        if (sh > 8) pack |= w[i + 1] << (32 - sh);
        uint nib = 0;
        for (int j = 0; j < 8; j++) nib |= ((pack >> (3 * j)) & 7u) << (4 * j);
        stage[lane * 8 + c] = nib;
      }
    } else {
      for (int c = 0; c < 8; c++) {                     // word c/2's half c%2: 8 values of 2 bits -> 8 nibbles
        uint v = (w[c >> 1] >> (16 * (c & 1))) & 0xFFFFu;
        v = (v | (v << 8)) & 0x00FF00FFu;
        v = (v | (v << 4)) & 0x0F0F0F0Fu;
        v = (v | (v << 2)) & 0x33333333u;
        stage[lane * 8 + c] = v;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float s[NF][4], bb[NF][4];
    for (int f = 0; f < NF; f++) {
      const uint4 q = colok[f] ? sbv[(g * N + n0 + f * 16 + fn) / 4] : uint4(0);
      const vec<bfloat, 8> v = as_type<vec<bfloat, 8>>(q);
      for (int j = 0; j < 4; j++) { s[f][j] = float(v[2 * j]); bb[f][j] = float(v[2 * j + 1]); }
    }
    auto a = tA.slice(g * 64, 0);
    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    op.run(a, b, P);
    simdgroup_barrier(mem_flags::mem_threadgroup);   // the op has read the stage before the next group's widening
    for (int t = 0; t < TMR; t++) {
      // the last row block can run past MP (MP % 32 == 16): those rows are never stored, and XS ends at MP
      const bool live = rb + t * 16 < MP;
      const float xs0 = live ? XS[g * MP + rb + t * 16 + fm] : 0.0f;
      const float xs1 = live ? XS[g * MP + rb + t * 16 + fm + 8] : 0.0f;
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++)
          for (int j = 0; j < 4; j++) {
            const int i = f * 8 + r * 4 + j;
            C[t][i] = fma(s[f][j], P[t * NF * 8 + i], fma(bb[f][j], r ? xs1 : xs0, C[t][i]));
          }
    }
  }
  threadgroup float part[(SK > 1 ? SK - 1 : 1) * NF * 8 * 32];
  for (int t = 0; t < TMR; t++) {
    if (SK > 1) {
      if (sg > 0) for (int i = 0; i < NF * 8; i++) part[((sg - 1) * NF * 8 + i) * 32 + lane] = C[t][i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0)
        for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < NF * 8; i++) C[t][i] += part[((s2 - 1) * NF * 8 + i) * 32 + lane];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 0)
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++) {
          const int m = rb + t * 16 + fm + 8 * r;
          const int nn = n0 + f * 16 + fn;
          if (m < M && nn < N)
            for (int j = 0; j < 4; j++) Y[m * N + nn + j] = static_cast<bfloat>(C[t][f * 8 + r * 4 + j]);
        }
  }
"""

# 5-, 6- and 8-bit weights to bytes for the bf16 x uint8 op: a group is one little-endian bit stream, 4 values a word
_NIBBLE_WIDENING = NIBBLES[NIBBLES.index("    if (BITS == 3) {"):
                           NIBBLES.index("    simdgroup_barrier(mem_flags::mem_threadgroup);\n    float s[NF][4]")]
_BYTE_WIDENING = """    for (int c = 0; c < 16; c++) {
      const int bit = 4 * BITS * c, i = bit >> 5, sh = bit & 31;
      uint word = w[i] >> sh;
      if (sh + 4 * BITS > 32) word |= w[i + 1] << (32 - sh);
      word = (word & ((1u << (2 * BITS)) - 1u)) | (((word >> (2 * BITS)) & ((1u << (2 * BITS)) - 1u)) << 16);
      word = (word & ((0x10001u << BITS) - 0x10001u)) | (((word >> BITS) & ((0x10001u << BITS) - 0x10001u)) << 8);
      stage[lane * 16 + c] = word;
    }
"""


def _bytes(source: str) -> str:
    for old, new in (
        ('static_assert(BITS == 2 || BITS == 3, "4-bit weights go to the tensor op as they are");',
         'static_assert(BITS == 5 || BITS == 6 || BITS == 8, "bytes for 5-, 6- and 8-bit weights");'),
        ("threadgroup uint stage_all[SK * NT * 8];               // per K slice: NT columns x 64 nibbles",
         "threadgroup uint stage_all[SK * NT * 16];              // per K slice: NT columns x 64 bytes"),
        ("threadgroup uint* stage = stage_all + sg * NT * 8;", "threadgroup uint* stage = stage_all + sg * NT * 16;"),
        ("tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline> b((threadgroup uchar*)stage,",
         "tensor<threadgroup uint8_t, dextents<int32_t, 2>, tensor_inline> b((threadgroup uint8_t*)stage,"),
        (_NIBBLE_WIDENING, _BYTE_WIDENING),
    ):
        if source.count(old) != 1:
            raise AssertionError(f"lane_widen: the nibble kernel changed ({old[:60]!r})")
        source = source.replace(old, new)
    return source


BYTES = _bytes(NIBBLES)


def grouped(source: str) -> str:
    """A widening kernel for groups of GS values (32 or 64) instead of 64; the group-64 kernels keep their source."""

    for old, new in (
        ("  constexpr int KG = K / 64;\n", "  constexpr int KG = K / GS;\n"),
        ("  constexpr int WPG = 2 * BITS;                          // words per column per group: 64 values x BITS bits\n",
         "  constexpr int WPG = GS * BITS / 32;                    // words per column per group: GS values x BITS bits\n"),
        ("matmul2d_descriptor(16 * TMR, NT, 64,", "matmul2d_descriptor(16 * TMR, NT, GS,"),
        ("stage_all[SK * NT * 8];               // per K slice: NT columns x 64 nibbles",
         "stage_all[SK * NT * (GS / 8)];        // per K slice: NT columns x GS nibbles"),
        ("stage_all[SK * NT * 16];              // per K slice: NT columns x 64 bytes",
         "stage_all[SK * NT * (GS / 4)];        // per K slice: NT columns x GS bytes"),
        ("stage_all + sg * NT * 8;", "stage_all + sg * NT * (GS / 8);"),
        ("stage_all + sg * NT * 16;", "stage_all + sg * NT * (GS / 4);"),
        ("dextents<int32_t, 2>(64, NT));", "dextents<int32_t, 2>(GS, NT));"),
        ("    auto a = tA.slice(g * 64, 0);\n", "    auto a = tA.slice(g * GS, 0);\n"),
    ):
        if old in source:
            source = source.replace(old, new)
    for old, new in (("for (int c = 0; c < 8; c++) {\n        const int bit = 24 * c,",
                      "for (int c = 0; c < GS / 8; c++) {\n        const int bit = 24 * c,"),
                     ("stage[lane * 8 + c] = nib;", "stage[lane * (GS / 8) + c] = nib;"),
                     ("for (int c = 0; c < 8; c++) {                     // word c/2's half",
                      "for (int c = 0; c < GS / 8; c++) {                // word c/2's half"),
                     ("stage[lane * 8 + c] = v;", "stage[lane * (GS / 8) + c] = v;"),
                     ("for (int c = 0; c < 16; c++) {\n      const int bit = 4 * BITS * c,",
                      "for (int c = 0; c < GS / 4; c++) {\n      const int bit = 4 * BITS * c,"),
                     ("stage[lane * 16 + c] = word;", "stage[lane * (GS / 4) + c] = word;")):
        if source.count(old) > 1:
            raise AssertionError(f"lane_widen: ambiguous edit ({old[:50]!r})")
        source = source.replace(old, new)
    if "64" in source.replace("int64_t", "").replace("uint64", ""):
        raise AssertionError("lane_widen: a group-64 constant is left in the grouped kernel")
    return source


NIBBLES_GROUPED, BYTES_GROUPED = grouped(NIBBLES), grouped(BYTES)


def sources() -> dict[str, str]:
    return {"nibbles": NIBBLES, "bytes": BYTES, "nibbles_grouped": NIBBLES_GROUPED, "bytes_grouped": BYTES_GROUPED}


__all__ = ["BYTES", "BYTES_GROUPED", "NIBBLES", "NIBBLES_GROUPED", "grouped", "sources"]

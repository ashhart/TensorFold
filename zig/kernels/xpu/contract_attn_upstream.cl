// The replaced attention from 1a5644892, retaining its (slices, heads, rows) decode grid.

// Qwen3.8-27B decode kernels on Intel Xe2: serial GDN (a few tokens) and attention over the cache.
#pragma OPENCL FP_CONTRACT OFF
#include "shape.h"

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort f2bf(float f) {
  return intel_convert_bfloat16_as_ushort(f); // hardware conversion (round to nearest even)
}

// Tree table of a verified block (<= 16 rows): TREE[t*17] = depth | SAVE | LOAD flags, then the ancestor row per depth.
#define DEPTH(t) (TREE[(t) * 17] & 255)
#define SAVE(t) (TREE[(t) * 17] & 256)
#define LOAD(t) (TREE[(t) * 17] & 512)
#define ANC(t, d) TREE[(t) * 17 + 1 + (d)]

// Decode attention by key slices: one subgroup per (slice, K/V head, query part, row). Grid y is AKV * QPARTS.
#define SPLIT 32
KERNEL(16) void attn_decode(global const ushort *Q, global const ushort *KC, global const ushort *VC, global float *PM,
                            global float *PL, global float *PO, int p0, int NS, float scale, global const int *TREE,
                            global const ulong *MT, global const int *POS, int cap) {
  // MT (if non-null): several streams, one row each; with TREE, tree windows; cap > 0: int8 cache (256 bytes per row).
#if Q27_QPARTS == 1
  const int sp = get_group_id(0), hk = get_group_id(1), t = get_group_id(2), lane = get_sub_group_local_id();
#else
  const int sp = get_group_id(0), hk = get_group_id(1) / Q27_QPARTS, t = get_group_id(2), lane = get_sub_group_local_id();
  const int qh0 = hk * Q27_GQA + (get_group_id(1) % Q27_QPARTS) * Q27_QPG; // the first query head held
#endif
  int base = p0, wf = 0;
  if (MT) {
    KC = (global const ushort *)MT[t * 8];
    VC = (global const ushort *)MT[t * 8 + 1];
    wf = (int)MT[t * 8 + 6];
    base = TREE ? POS[wf] : 0x7fffffff;
  }
  const int last = MT ? POS[t] : p0 + DEPTH(t), j0 = sp * SPLIT, j1 = min(j0 + SPLIT, last + 1);
  float q[Q27_QPG][16], o[Q27_QPG][16], m[Q27_QPG], l[Q27_QPG];
  for (int a = 0; a < Q27_QPG; a++) {
    for (int i = 0; i < 16; i++) {
#if Q27_QPARTS == 1
      q[a][i] = bf2f(Q[((long)t * Q27_AQ + hk * Q27_GQA + a) * 256 + i * 16 + lane]);
#else
      q[a][i] = bf2f(Q[((long)t * Q27_AQ + qh0 + a) * 256 + i * 16 + lane]);
#endif
      o[a][i] = 0.0f;
    }
    m[a] = -INFINITY;
    l[a] = 0.0f;
  }
  for (int j = j0; j < j1; j++) {
    const long kr = j < base ? j : base + ANC(t, j - base) - wf; // cache row of key j
    float kv[16];
    if (cap) {
      const global char *k8 = (const global char *)KC + (kr * Q27_AKV + hk) * 256;
      const float ks = ((const global float *)((const global char *)KC + (long)cap * Q27_KVROW))[kr * Q27_AKV + hk];
      for (int i = 0; i < 16; i++) kv[i] = (float)k8[i * 16 + lane] * ks;
    } else
      for (int i = 0; i < 16; i++) kv[i] = bf2f(KC[(kr * Q27_AKV + hk) * 256 + i * 16 + lane]);
    float sc[Q27_QPG];
    for (int a = 0; a < Q27_QPG; a++) {
      float d = 0.0f;
      for (int i = 0; i < 16; i++) d += q[a][i] * kv[i];
      sc[a] = sub_group_reduce_add(d) * scale;
    }
    if (cap) {
      const global char *v8 = (const global char *)VC + (kr * Q27_AKV + hk) * 256;
      const float vs = ((const global float *)((const global char *)VC + (long)cap * Q27_KVROW))[kr * Q27_AKV + hk];
      for (int i = 0; i < 16; i++) kv[i] = (float)v8[i * 16 + lane] * vs;
    } else
      for (int i = 0; i < 16; i++) kv[i] = bf2f(VC[(kr * Q27_AKV + hk) * 256 + i * 16 + lane]);
    for (int a = 0; a < Q27_QPG; a++) {
      const float mn = fmax(m[a], sc[a]);
      const float al = exp(m[a] - mn), p = exp(sc[a] - mn);
      l[a] = l[a] * al + p;
      for (int i = 0; i < 16; i++) o[a][i] = o[a][i] * al + p * kv[i];
      m[a] = mn;
    }
  }
  for (int a = 0; a < Q27_QPG; a++) {
#if Q27_QPARTS == 1
    const long base = ((long)t * Q27_AQ + hk * Q27_GQA + a) * NS + sp;
#else
    const long base = ((long)t * Q27_AQ + qh0 + a) * NS + sp;
#endif
    if (lane == 0) {
      PM[base] = m[a];
      PL[base] = l[a];
    }
    for (int i = 0; i < 16; i++) PO[base * 256 + i * 16 + lane] = o[a][i];
  }
}

// Merge of the slices, in order: O bf16 [T, Q27_AQ, 256]. One subgroup per (head, row).
KERNEL(16) void attn_merge(global const float *PM, global const float *PL, global const float *PO, global ushort *O,
                           int p0, int NS, global const int *TREE, global const int *POS) {
  const int hh = get_group_id(0), t = get_group_id(1), lane = get_sub_group_local_id();
  const int used = (POS ? POS[t] : p0 + DEPTH(t)) / SPLIT + 1; // non-empty slices of row t
  const long base = ((long)t * Q27_AQ + hh) * NS;
  float m = -INFINITY;
  for (int s = 0; s < used; s++) m = fmax(m, PM[base + s]);
  float l = 0.0f, o[16];
  for (int i = 0; i < 16; i++) o[i] = 0.0f;
  for (int s = 0; s < used; s++) {
    const float w = exp(PM[base + s] - m);
    l += PL[base + s] * w;
    for (int i = 0; i < 16; i++) o[i] += PO[(base + s) * 256 + i * 16 + lane] * w;
  }
  for (int i = 0; i < 16; i++) O[((long)t * Q27_AQ + hh) * 256 + i * 16 + lane] = f2bf(o[i] / l);
}

// Qwen3.8 Flash Next (qwen4_exp) kernels for Intel Xe2 (Level Zero / OpenCL C), supporting TP={1,2,4}.

#pragma OPENCL FP_CONTRACT OFF
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define SG 16
#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float bf(ushort h) { return as_float((uint)h << 16); }
static inline ushort to_bf(float f) { return intel_convert_bfloat16_as_ushort(f); }
static inline float rbf(float f) { return bf(to_bf(f)); }
static inline float tf_silu(float x) { return x / (1.0f + exp(-x)); }
static inline float tf_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
static inline float bsilu(float x) { return rbf(tf_silu(x)); }
static inline float bsig(float x) { return rbf(tf_sigmoid(x)); }
static inline float softplus_f(float x) { return x > 20.0f ? x : log(1.0f + exp(x)); }

// Deterministic work-group sum over N threads (N a multiple of 16, up to 1024).
static inline float wg_sum(float v, local float *red, int nsg) {
  const float s = sub_group_reduce_add(v);
  const int sg = get_sub_group_id();
  if (get_sub_group_local_id() == 0) red[sg] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  float tot = 0.0f;
  for (int i = 0; i < nsg; i++) tot += red[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  return tot;
}

// MLX affine 4-bit group-32 embedding copied into `copies` streams: out [rows, copies, D] bf16.
KERNEL(256) void fn_embed_q4(global const int *IDS, global const uint *W, global const ushort *S, global const ushort *B,
                             global ushort *OUT, int D, int copies) {
  const int r = get_group_id(0);
  const long id = (long)IDS[r];
  const int words = D / 8, groups = D / 32;
  for (int d = get_local_id(0); d < D; d += 256) {
    const uint w = W[id * words + d / 8];
    const float q = (float)((w >> (4 * (d & 7))) & 15u);
    const int g = d / 32;
    const ushort val = to_bf(q * bf(S[id * groups + g]) + bf(B[id * groups + g]));
    for (int s = 0; s < copies; s++) {
      OUT[((long)r * copies + s) * D + d] = val;
    }
  }
}

KERNEL(256) void fn_embed_bf16(global const int *IDS, global const ushort *W, global ushort *OUT, int D, int copies) {
  const int r = get_group_id(0);
  const long id = (long)IDS[r];
  for (int d = get_local_id(0); d < D; d += 256) {
    const ushort val = W[id * D + d];
    for (int s = 0; s < copies; s++) {
      OUT[((long)r * copies + s) * D + d] = val;
    }
  }
}

// Whole-row RMSNorm (used by MTP pre_fc_norm_embedding [D=2560] and pre_fc_norm_hidden [wide=10240]).
KERNEL(256) void fn_rmsnorm_rows(global const ushort *X, int x_stride, global const float *W, global ushort *OUT,
                                 float eps, int K) {
  local float red[16];
  const int r = get_group_id(0), lid = get_local_id(0);
  const global ushort *xr = X + (long)r * x_stride;
  float ss = 0.0f;
  for (int i = lid; i < K; i += 256) {
    const float v = bf(xr[i]);
    ss = fma(v, v, ss);
  }
  const float tot = wg_sum(ss, red, 16);
  const float rinv = rsqrt(tot / (float)K + eps);
  global ushort *yr = OUT + (long)r * K;
  for (int i = lid; i < K; i += 256) {
    yr[i] = to_bf(bf(xr[i]) * rinv * W[i]);
  }
}

// MTP stream add: OUT[r, s, d] = bf16(E[r, d] + HS[r, s, d]), grid (rows, (D + 255) / 256).
KERNEL(256) void fn_add_streams(global const ushort *E, global const ushort *HS, global ushort *OUT, int D, int S) {
  const int r = get_group_id(0), d = get_group_id(1) * 256 + get_local_id(0);
  if (d >= D) return;
  const float ev = bf(E[(long)r * D + d]);
  for (int s = 0; s < S; s++) {
    const long at = ((long)r * S + s) * D + d;
    OUT[at] = to_bf(ev + bf(HS[at]));
  }
}

// Stream-parallel hc_writeback + normalize for D=2560, S=4 (WIDE=10240): grid (4, rows), WG = 256.
KERNEL(256) void fn_hc_writeback_norm(global ushort *H, global const float *BR, global const ushort *INJ,
                                      global const float *SCALE, global ushort *NORM, float eps, int has_br) {
  local float red[16];
  const int s = get_group_id(0), r = get_group_id(1), lid = get_local_id(0);
  const int D = 2560;
  const float inj_v = has_br ? bf(INJ[(long)r * 4 + s]) : 0.0f;
  const long base = (long)r * (4 * D) + (long)s * D;
  float hv[10];
  float ss = 0.0f;
  for (int c = 0; c < 10; c++) {
    const int d = c * 256 + lid;
    float v = bf(H[base + d]);
    if (has_br) {
      const float branch = rbf(BR[(long)r * D + d]);
      v = rbf(fma(branch, inj_v, v));
      H[base + d] = to_bf(v);
    }
    hv[c] = v;
    ss = fma(v, v, ss);
  }
  const float tot = wg_sum(ss, red, 16);
  const float rinv = rsqrt(tot / (float)D + eps);
  for (int c = 0; c < 10; c++) {
    const int d = c * 256 + lid;
    NORM[base + d] = to_bf((hv[c] * rinv) * SCALE[s * D + d]);
  }
}

// Writeback only (without NORM), used before final head or Layer 1 PLE: grid (rows, 10), WG = 256.
KERNEL(256) void fn_hc_writeback_only(global ushort *H, global const float *BR, global const ushort *INJ, int has_br) {
  if (!has_br) return;
  const int r = get_group_id(0), c = get_group_id(1), lid = get_local_id(0);
  const int D = 2560, d = c * 256 + lid;
  const float branch = rbf(BR[(long)r * D + d]);
  for (int s = 0; s < 4; s++) {
    const long at = (long)r * (4 * D) + (long)s * D + d;
    const float v = rbf(fma(branch, bf(INJ[(long)r * 4 + s]), bf(H[at])));
    H[at] = to_bf(v);
  }
}

// Two-stage argmax + log-sum-exp over BF16 logits [R, V]: Stage 1 grid (256, R), WG = 256.
KERNEL(256) void fn_logits_part(global const ushort *L, int V, global float *PV, global int *PI, global float *PS) {
  local float bv[256], bs[256];
  local int bi[256];
  const int lid = get_local_id(0), gid = get_group_id(0), row = get_group_id(1);
  const int per = (V + 255) / 256, lo = gid * per, hi = min(V, lo + per);
  float best = -INFINITY;
  int idx = 0x7fffffff;
  for (int i = lo + lid; i < hi; i += 256) {
    const float v = bf(L[(long)row * V + i]);
    if (v > best) {
      best = v;
      idx = i;
    }
  }
  bv[lid] = best;
  bi[lid] = idx;
  barrier(CLK_LOCAL_MEM_FENCE);
  float gmax = bv[0];
  int gidx = bi[0];
  for (int i = 1; i < 256; i++) {
    if (bv[i] > gmax || (bv[i] == gmax && bi[i] < gidx)) {
      gmax = bv[i];
      gidx = bi[i];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float s = 0.0f;
  if (gmax > -INFINITY) {
    for (int i = lo + lid; i < hi; i += 256) s += exp(bf(L[(long)row * V + i]) - gmax);
  }
  bs[lid] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0) {
    float tot = 0.0f;
    for (int i = 0; i < 256; i++) tot += bs[i];
    PV[row * 256 + gid] = gmax;
    PI[row * 256 + gid] = gidx;
    PS[row * 256 + gid] = tot;
  }
}

// Stage 2: grid (R), WG = 256 -> OUT_TOK[r] = argmax, OUT_BEST[r*4 + 0..2] = (max, lse, id_bits).
KERNEL(256) void fn_logits_best(global const float *PV, global const int *PI, global const float *PS,
                                global int *OUT_TOK, global float *OUT_BEST) {
  local float bv[256], bs[256];
  local int bi[256];
  const int lid = get_local_id(0), row = get_group_id(0);
  bv[lid] = PV[row * 256 + lid];
  bi[lid] = PI[row * 256 + lid];
  bs[lid] = PS[row * 256 + lid];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0) {
    float best = bv[0];
    int idx = bi[0];
    for (int i = 1; i < 256; i++) {
      if (bv[i] > best || (bv[i] == best && bi[i] < idx)) {
        best = bv[i];
        idx = bi[i];
      }
    }
    float sum = 0.0f;
    if (best > -INFINITY) {
      for (int i = 0; i < 256; i++) {
        if (bv[i] > -INFINITY) sum += bs[i] * exp(bv[i] - best);
      }
    }
    OUT_TOK[row] = idx;
    OUT_BEST[row * 4 + 0] = best;
    OUT_BEST[row * 4 + 1] = best + log(sum);
    OUT_BEST[row * 4 + 2] = as_float(idx);
    OUT_BEST[row * 4 + 3] = 0.0f;
  }
}

// Top-64 candidates of one BF16 logits row `L + row * V`: Stage 1 keeps top-64 per 256-slice.
KERNEL(64) void fn_topk64_part(global const ushort *L, int V, int row, global float *TV, global int *TI) {
  local float sv[64];
  local int si[64];
  const int tid = get_local_id(0), gid = get_group_id(0);
  const int per = (V + 255) / 256, lo = gid * per, hi = min(V, lo + per);
  float my_v[32];
  int my_i[32];
  int cnt = 0;
  for (int i = lo + tid; i < hi; i += 64) {
    my_v[cnt] = bf(L[(long)row * V + i]);
    my_i[cnt] = i;
    cnt++;
  }
  for (int k = 0; k < 64; k++) {
    float bv = -INFINITY;
    int bi = 0x7fffffff, bj = -1;
    for (int j = 0; j < cnt; j++) {
      if (my_v[j] > bv || (my_v[j] == bv && my_i[j] < bi)) {
        bv = my_v[j];
        bi = my_i[j];
        bj = j;
      }
    }
    sv[tid] = bv;
    si[tid] = bi;
    barrier(CLK_LOCAL_MEM_FENCE);
    float wv = sv[0];
    int wi = si[0];
    for (int u = 1; u < 64; u++) {
      if (sv[u] > wv || (sv[u] == wv && si[u] < wi)) {
        wv = sv[u];
        wi = si[u];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (bj >= 0 && my_i[bj] == wi) my_v[bj] = -INFINITY;
    if (tid == 0) {
      TV[gid * 64 + k] = wv;
      TI[gid * 64 + k] = wi;
    }
  }
}

// Convert BF16 row [V] to FP32 row [V] (for full-vocab host sampling when top_k == 0 or top_k > 64).
KERNEL(256) void fn_bf16_to_f32(global const ushort *IN, global float *OUT, int N) {
  const int i = get_global_id(0);
  if (i < N) OUT[i] = bf(IN[i]);
}

#include "fn_dpas.cl"
#include "fn_mixers.cl"
#include "fn_moe_head.cl"

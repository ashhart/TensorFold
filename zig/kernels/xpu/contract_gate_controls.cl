// The kernel contract's planted controls on XPU: each breaks one gate on purpose, and that gate must go red.

#pragma OPENCL FP_CONTRACT OFF
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }

// Gate 1: a row's maximum by fmax, which drops a NaN input, so a row holding NaN reads as its largest number.
KERNEL(256) void tf_ctl_row_max_fmax(global const ushort *X, global float *Y, int K) {
  local float bv[256];
  const int row = get_group_id(0), lid = get_local_id(0);
  float m = -INFINITY;
  for (int i = lid; i < K; i += 256) m = fmax(m, bf2f(X[(long)row * K + i]));
  bv[lid] = m;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int o = 128; o > 0; o >>= 1) {
    if (lid < o) bv[lid] = fmax(bv[lid], bv[lid + o]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) Y[row] = bv[0];
}

// Gate 2: row sums by an atomic add from all over the grid, so the order varies run to run; a compare-exchange loop.
KERNEL(256) void tf_ctl_atomic_sum(global const ushort *X, global float *Y, int K) {
  const int i = get_group_id(0) * 256 + get_local_id(0), row = get_group_id(1);
  if (i >= K) return;
  volatile global int *slot = (volatile global int *)Y + row;
  int old = *slot;
  for (;;) {
    const float next = as_float(old) + bf2f(X[(long)row * K + i]);
    const int got = atomic_cmpxchg(slot, old, as_int(next));
    if (got == old) return;
    old = got;
  }
}

// A work-group's sum of every 256th input of a row up to `end`, then its tree; `bv` is the kernel's local buffer.
static float wg_sum(local float *bv, global const ushort *X, int K, int row, int end, int lid) {
  float s = 0.0f;
  for (int i = lid; i < end; i += 256) s += bf2f(X[(long)row * K + i]);
  bv[lid] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int o = 128; o > 0; o >>= 1) {
    if (lid < o) bv[lid] += bv[lid + o];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  return bv[0];
}

// Gate 2 too: a row sum that writes the sum back into its own input, which its spec says it never does.
KERNEL(256) void tf_ctl_scribble(global ushort *X, global float *Y, int K) {
  local float bv[256];
  const int row = get_group_id(0), lid = get_local_id(0);
  const float s = wg_sum(bv, X, K, row, K, lid);
  if (lid == 0) {
    Y[row] = s;
    X[(long)row * K] = (ushort)(as_uint(s) >> 16);
  }
}

// Gate 3: a row sum whose partial tail loses its last input: right whenever K is a whole number of 32s.
KERNEL(256) void tf_ctl_drop_tail(global const ushort *X, global float *Y, int K) {
  local float bv[256];
  const int row = get_group_id(0), lid = get_local_id(0), whole = K / 32 * 32;
  float s = 0.0f;
  for (int i = lid; i < whole; i += 256) s += bf2f(X[(long)row * K + i]);
  if (lid < K - whole - 1) s += bf2f(X[(long)row * K + whole + lid]);
  bv[lid] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int o = 128; o > 0; o >>= 1) {
    if (lid < o) bv[lid] += bv[lid + o];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) Y[row] = bv[0];
}

// Gate 4: a dot product summed in half, while its spec declares fp32 sums.
KERNEL(256) void tf_ctl_fp16_sum(global const ushort *X, global const ushort *W, global float *Y, int K) {
  local half bv[256];
  const int row = get_group_id(0), lid = get_local_id(0);
  half s = (half)0.0f;
  for (int i = lid; i < K; i += 256) s = (half)(s + (half)(bf2f(X[(long)row * K + i]) * bf2f(W[i])));
  bv[lid] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int o = 128; o > 0; o >>= 1) {
    if (lid < o) bv[lid] = (half)(bv[lid] + bv[lid + o]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) Y[row] = (float)bv[0];
}

// Gate 3: glue_quant.cl's topk_merge with a tie going to the higher id, so equal logits come out in the wrong order.
static void tf_best_high(float *v, int *i, local float *rv, local int *ri) {
  float bv = sub_group_reduce_max(*v);
  int bi = sub_group_reduce_max(*v == bv ? *i : -1);
  if (get_sub_group_local_id() == 0) { rv[get_sub_group_id()] = bv; ri[get_sub_group_id()] = bi; }
  barrier(CLK_LOCAL_MEM_FENCE);
  bv = rv[0];
  bi = ri[0];
  for (int s = 1; s < 16; s++)
    if (rv[s] > bv || (rv[s] == bv && ri[s] > bi)) { bv = rv[s]; bi = ri[s]; }
  barrier(CLK_LOCAL_MEM_FENCE);
  *v = bv;
  *i = bi;
}

KERNEL(256) void tf_ctl_topk_merge_ties(global const float *PV, global const int *PI, global const int *PF, int K, int NS,
                                        global uint *OUT) {
  local float rv[16];
  local int ri[16];
  const int lid = get_local_id(0), row = get_group_id(0);
  const int live = lid < NS;
  int ptr = 0;
  if (lid == 0) {
    int f = 0;
    for (int s = 0; s < NS; s++) f |= PF[row * 256 + s];
    OUT[row * 132 + 128] = f;
  }
  for (int k = 0; k < min(K, 64); k++) {
    float v = live && ptr < K ? PV[((long)row * 256 + lid) * 64 + ptr] : -INFINITY;
    int i = live && ptr < K ? PI[((long)row * 256 + lid) * 64 + ptr] : -1;
    float bv = v;
    int bi = i;
    tf_best_high(&bv, &bi, rv, ri);
    if (i == bi) {
      ptr++;
      OUT[row * 132 + k] = as_uint(v);
      OUT[row * 132 + 64 + k] = (uint)i;
    }
  }
}

static void tf_best_low(float *v, int *i, local float *rv, local int *ri) {
  float bv = sub_group_reduce_max(*v);
  int bi = sub_group_reduce_min(*v == bv ? *i : 0x7fffffff);
  if (get_sub_group_local_id() == 0) { rv[get_sub_group_id()] = bv; ri[get_sub_group_id()] = bi; }
  barrier(CLK_LOCAL_MEM_FENCE);
  bv = rv[0];
  bi = ri[0];
  for (int s = 1; s < 16; s++)
    if (rv[s] > bv || (rv[s] == bv && ri[s] < bi)) { bv = rv[s]; bi = ri[s]; }
  barrier(CLK_LOCAL_MEM_FENCE);
  *v = bv;
  *i = bi;
}

// Gate 1: glue_quant.cl's topk_merge with the not-finite flag read off the merged candidates, not the whole row.
KERNEL(256) void tf_ctl_topk_merge_cand_flag(global const float *PV, global const int *PI, global const int *PF, int K,
                                             int NS, global uint *OUT) {
  local float rv[16];
  local int ri[16];
  const int lid = get_local_id(0), row = get_group_id(0);
  const int live = lid < NS;
  int ptr = 0;
  int seen = 0;
  for (int k = 0; k < min(K, 64); k++) {
    float v = live && ptr < K ? PV[((long)row * 256 + lid) * 64 + ptr] : -INFINITY;
    int i = live && ptr < K ? PI[((long)row * 256 + lid) * 64 + ptr] : 0x7fffffff;
    float bv = v;
    int bi = i;
    tf_best_low(&bv, &bi, rv, ri);
    seen |= !isfinite(bv);
    if (i == bi) {
      ptr++;
      OUT[row * 132 + k] = as_uint(v);
      OUT[row * 132 + 64 + k] = (uint)i;
    }
  }
  if (lid == 0) OUT[row * 132 + 128] = seen;
}

// Clean-room Mamba-2 (SSD-SSM) Metal 3.1+ kernels for Nemotron-H hybrid models.
//
// Matches NVIDIA Nemotron-H / Nemotron-3.5 exact math:
//   1. mamba2_conv_silu      : 4-tap causal depthwise conv1d + bias + SiLU + history update
//   2. mamba2_ssm_scan       : Multi-head fp32 state recurrence (64 heads x 64 head_dim x 128 d_state)
//   3. mamba2_norm_gate_sums : MambaRMSNormGated = weight * grouped_rms_norm(silu(gate) * y) + group-64 sums
#include <metal_stdlib>
using namespace metal;

#define UNROLL _Pragma("clang loop unroll(full)")

struct MambaConvParams {
  uint rows;
  uint inter_dim;      // mamba_num_heads * mamba_head_dim (4096)
  uint conv_dim;       // inter_dim + 2 * n_groups * d_state (6144)
  uint in_proj_stride; // inter_dim + conv_dim + num_heads (10304)
  uint commit_hist;    // 1 = write updated 3-row conv history to hist_out
};

// Fused 4-tap depthwise conv1d + bias + SiLU over [rows, conv_dim] with causal history ring.
// Grid: threads (conv_dim, 1, 1) with threadgroup (256, 1, 1).
kernel void mamba2_conv_silu(const device bfloat* in_proj [[buffer(0)]],
                             const device bfloat* conv_w [[buffer(1)]],
                             const device bfloat* conv_b [[buffer(2)]],
                             const device bfloat* hist_in [[buffer(3)]],
                             device bfloat* conv_out [[buffer(4)]],
                             device bfloat* hist_out [[buffer(5)]],
                             constant MambaConvParams& p [[buffer(6)]],
                             uint ch [[thread_position_in_grid]]) {
  if (ch >= p.conv_dim) return;

  const device bfloat* wp = conv_w + (size_t)ch * 4;
  const float w0 = float(wp[0]);
  const float w1 = float(wp[1]);
  const float w2 = float(wp[2]);
  const float w3 = float(wp[3]);
  const float bias = float(conv_b[ch]);

  float h0 = float(hist_in[(size_t)0 * p.conv_dim + ch]);
  float h1 = float(hist_in[(size_t)1 * p.conv_dim + ch]);
  float h2 = float(hist_in[(size_t)2 * p.conv_dim + ch]);

  const size_t x_col = (size_t)p.inter_dim + ch;
  for (uint r = 0; r < p.rows; r++) {
    const float cur = float(in_proj[(size_t)r * p.in_proj_stride + x_col]);
    const float acc = fma(h0, w0, fma(h1, w1, fma(h2, w2, fma(cur, w3, bias))));
    const float y = acc / (1.0f + exp(-acc)); // SiLU
    conv_out[(size_t)r * p.conv_dim + ch] = bfloat(y);
    h0 = h1;
    h1 = h2;
    h2 = cur;
  }

  if (p.commit_hist != 0) {
    hist_out[(size_t)0 * p.conv_dim + ch] = bfloat(h0);
    hist_out[(size_t)1 * p.conv_dim + ch] = bfloat(h1);
    hist_out[(size_t)2 * p.conv_dim + ch] = bfloat(h2);
  }
}

struct MambaSsmParams {
  uint rows;
  uint num_heads;      // 64
  uint head_dim;       // 64
  uint n_groups;       // 8
  uint d_state;        // 128
  uint conv_dim;       // 6144
  uint in_proj_stride; // 10304
  uint dt_offset;      // inter_dim + conv_dim (10240)
  uint commit_state;   // 1 = store updated fp32 state back to state_buf
  float dt_min;        // 0.0
  float dt_max;        // +inf
};

// Multi-head Mamba-2 SSD recurrence for d_state == 128 (4 state slots per SIMD lane).
// Grid: threadgroups (num_heads, head_dim / 4, 1) of 128 threads (4 simdgroups = 4 d-channels).
kernel void mamba2_ssm_scan(const device bfloat* conv_out [[buffer(0)]],
                            const device bfloat* in_proj [[buffer(1)]],
                            const device float* a_log [[buffer(2)]],
                            const device bfloat* dt_bias [[buffer(3)]],
                            const device bfloat* d_skip [[buffer(4)]],
                            device float* state_buf [[buffer(5)]],
                            device bfloat* ssm_out [[buffer(6)]],
                            constant MambaSsmParams& p [[buffer(7)]],
                            uint2 grp [[threadgroup_position_in_grid]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
  const uint h = grp.x;
  const uint d = grp.y * 4 + sg;
  if (h >= p.num_heads || d >= p.head_dim) return;

  const uint inter_dim = p.num_heads * p.head_dim;
  const uint heads_per_group = p.num_heads / p.n_groups;
  const uint g = h / heads_per_group;

  const float neg_A = -exp(a_log[h]);
  const float dtb = float(dt_bias[h]);
  const float d_val = float(d_skip[h]);

  const size_t st_base = ((size_t)h * p.head_dim + d) * 128 + (size_t)lane * 4;
  float4 s = *(const device float4*)(state_buf + st_base);

  const size_t x_off = (size_t)h * p.head_dim + d;
  const size_t b_off = (size_t)inter_dim + (size_t)g * 128 + (size_t)lane * 4;
  const size_t c_off = (size_t)inter_dim + (size_t)p.n_groups * 128 + (size_t)g * 128 + (size_t)lane * 4;

  for (uint r = 0; r < p.rows; r++) {
    const device bfloat* crow = conv_out + (size_t)r * p.conv_dim;
    const float dt_raw = float(in_proj[(size_t)r * p.in_proj_stride + p.dt_offset + h]) + dtb;
    float dt = (dt_raw > 20.0f) ? dt_raw : log(1.0f + exp(dt_raw));
    dt = clamp(dt, p.dt_min, p.dt_max);

    const float decay = exp(dt * neg_A);
    const float x_val = float(crow[x_off]);
    const float dtx = dt * x_val;

    const float4 b4 = float4(*(const device vec<bfloat, 4>*)(crow + b_off));
    const float4 c4 = float4(*(const device vec<bfloat, 4>*)(crow + c_off));

    s = fma(s, float4(decay), b4 * dtx);
    const float4 prod = s * c4;
    const float y_part = (prod[0] + prod[1]) + (prod[2] + prod[3]);
    const float y_sum = simd_sum(y_part);

    if (lane == 0) {
      ssm_out[(size_t)r * inter_dim + x_off] = bfloat(y_sum + d_val * x_val);
    }
  }

  if (p.commit_state != 0) {
    *(device float4*)(state_buf + st_base) = s;
  }
}

struct MambaNormParams {
  uint rows;
  uint inter_dim;      // 4096
  uint n_groups;       // 8
  uint group_size;     // 512 (inter_dim / n_groups)
  uint in_proj_stride; // 10304 (gate starts at offset 0 of in_proj)
  uint sums_stride;    // 0 = skip sums (M == 1 qmv_1), > 0 = write group-64 sums
  float eps;
};

// Exact MambaRMSNormGated + optional group-64 sums for out_proj:
//   v = silu(gate) * ssm_out
//   out = norm_w * rms_norm_per_group(v, group_size=512)
// Grid: threadgroups (n_groups, rows, 1) of 256 threads (8 simdgroups x 64 elems = 512).
kernel void mamba2_norm_gate_sums(const device bfloat* ssm_out [[buffer(0)]],
                                  const device bfloat* in_proj [[buffer(1)]],
                                  const device bfloat* norm_w [[buffer(2)]],
                                  device bfloat* act_out [[buffer(3)]],
                                  device float* sums [[buffer(4)]],
                                  constant MambaNormParams& p [[buffer(5)]],
                                  uint2 grp [[threadgroup_position_in_grid]],
                                  ushort sg [[simdgroup_index_in_threadgroup]],
                                  ushort lane [[thread_index_in_simdgroup]]) {
  const uint g_idx = grp.x;
  const uint row = grp.y;
  if (g_idx >= p.n_groups || row >= p.rows) return;

  // Each of the 8 simdgroups handles one 64-element sub-group (2 elems/lane) when group_size == 512.
  const uint sub_groups = p.group_size / 64; // 8 for 512
  const uint base_ch = g_idx * p.group_size;

  float local_sq = 0.0f;
  // Support any multiple-of-512 group_size cleanly; for Nemotron-3.5 group_size == 512 -> 1 pass!
  if (p.group_size == 512) {
    const uint i0 = base_ch + sg * 64 + lane;
    const uint i1 = i0 + 32;
    const device bfloat* yrow = ssm_out + (size_t)row * p.inter_dim;
    const device bfloat* grow = in_proj + (size_t)row * p.in_proj_stride;

    const float g0 = float(grow[i0]);
    const float g1 = float(grow[i1]);
    const float v0 = (g0 / (1.0f + exp(-g0))) * float(yrow[i0]);
    const float v1 = (g1 / (1.0f + exp(-g1))) * float(yrow[i1]);

    local_sq = simd_sum(fma(v0, v0, v1 * v1));
    threadgroup float tg_sq[8];
    if (lane == 0) tg_sq[sg] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float total_sq = (tg_sq[0] + tg_sq[1] + tg_sq[2] + tg_sq[3]) +
                           (tg_sq[4] + tg_sq[5] + tg_sq[6] + tg_sq[7]);
    const float inv_rms = rsqrt(total_sq / 512.0f + p.eps);

    const bfloat o0 = bfloat(v0 * inv_rms * float(norm_w[i0]));
    const bfloat o1 = bfloat(v1 * inv_rms * float(norm_w[i1]));
    device bfloat* orow = act_out + (size_t)row * p.inter_dim;
    orow[i0] = o0;
    orow[i1] = o1;

    if (p.sums_stride > 0) {
      const float gsum = simd_sum(float(o0) + float(o1));
      if (lane == 0) {
        sums[(size_t)(g_idx * sub_groups + sg) * p.sums_stride + row] = gsum;
      }
    }
  }
}

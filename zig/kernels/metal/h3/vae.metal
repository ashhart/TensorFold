// H3 video VAE on Metal: fp32 matmul kernels for the ViT decoder, pattern from h3_rows.metal.

// Y[r, n] = B[n] + sum_k X[r, k] W[n, k]. Threads [N, R, 1], fp32 throughout.
[[kernel]] void h3_vae_matmul(
  const device float* X [[buffer(0)]],   // [R, K]
  const device float* W [[buffer(1)]],   // [N, K] (row-major, the safetensors layout)
  const device float* B [[buffer(2)]],   // [N]
  device float* Y [[buffer(3)]],         // [R, N]
  constant uint* P [[buffer(4)]],        // P[0]=R, P[1]=K, P[2]=N, P[3]=row0
  uint3 at [[thread_position_in_grid]]) {
  const uint R = P[0], K = P[1], N = P[2], row0 = P[3];
  if (at.x >= N || at.y >= R) return;
  const device float* x = X + (uint)at.y * K;
  const device float* w = W + (uint)at.x * K;
  float sum = 0.0f;
  for (uint k = 0; k < K; k++) sum += x[k] * w[k];
  Y[(row0 + at.y) * N + at.x] = sum + B[at.x];
}

// SwiGLU elementwise: out[r, f] = fused[r, f + F] * silu(fused[r, f]).
[[kernel]] void h3_vae_swiglu(
  const device float* FUSED [[buffer(0)]],
  device float* OUT [[buffer(1)]],
  constant uint* P [[buffer(2)]],        // P[0] = F
  uint3 at [[thread_position_in_grid]]) {
  const uint F = P[0];
  const uint r = at.y, f = at.x;
  const float gate = FUSED[r * 2 * F + f];
  const float value = FUSED[r * 2 * F + F + f];
  OUT[r * F + f] = value * gate / (1.0f + exp(-gate));
}

// RMSNorm in fp32 with a learned weight: one threadgroup per row.
[[kernel]] void h3_vae_rmsnorm(
  const device float* X [[buffer(0)]],
  const device float* W [[buffer(1)]],
  device float* Y [[buffer(2)]],
  constant uint* P [[buffer(3)]],        // P[0] = D
  uint tid [[thread_index_in_threadgroup]],
  uint tg [[threadgroup_position_in_grid]]) {
  const uint D = P[0];
  threadgroup float red[256];
  float acc = 0.0f;
  for (uint i = tid; i < D; i += 256) acc += X[tg * D + i] * X[tg * D + i];
  red[tid] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride /= 2) {
    if (tid < stride) red[tid] += red[tid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float inv = rsqrt(red[0] / float(D) + 1e-5f);
  for (uint i = tid; i < D; i += 256) Y[tg * D + i] = X[tg * D + i] * inv * W[i];
}

// Full attention on the interleaved qkv layout: softmax q k per head, v per head; writes attended[h][r][i] ...
[[kernel]] void h3_vae_attention_il(
  const device float* QKV [[buffer(0)]],
  device float* Y [[buffer(1)]],
  const device float* ROPE [[buffer(2)]],
  constant uint* P [[buffer(3)]],
  uint at1 [[thread_position_in_grid]]) {
  const uint rows = P[0];
  const uint hd = 64, heads = 32, dim = heads * hd;
  const uint r = at1 / 32;
  const uint head = at1 % 32;
  if (r >= rows || head >= heads) return;
  const device float* q_raw = QKV + r * 3 * dim + head * 3 * hd;
  float q_sq = 0.0f;
  for (uint i = 0; i < hd; i++) q_sq += q_raw[i] * q_raw[i];
  const float q_rstd = rsqrt(q_sq / float(hd) + 1e-5f);
  float q_normed[64];
  for (uint i = 0; i < hd; i++) q_normed[i] = q_raw[i] * q_rstd;
  // apply RoPE: cos/sin tables are [rows, 48] cos then [rows, 48] sin rotary_dim = 48 (75% of 64), half_dim = 24
  const device float* COS = ROPE + r * 48;
  const device float* SIN = ROPE + rows * 48 + r * 48;
  float q_rot[64];
  float k_rot_base[64];  // only used if we need per-key rotation — skip for now
  for (uint i = 0; i < hd; i++) {
    if (i < 48) {
      const uint half_dim = 24;
      const uint fi = i % half_dim;
      const float c = COS[fi];
      const float sn = SIN[fi];
      if (i < half_dim) {
        q_rot[i] = q_normed[i] * c - q_normed[i + half_dim] * sn;
      } else {
        q_rot[i] = q_normed[i] * c + q_normed[i - half_dim] * sn;
      }
    } else {
      q_rot[i] = q_normed[i];
    }
  }
  // three-pass softmax (no large per-thread arrays): pass 1: global max of dots
  float gmax = -1e30f;
  for (uint k = 0; k < rows; k++) {
    float dot = 0.0f;
    const device float* kk_raw = QKV + k * 3 * dim + head * 3 * hd + hd;
    float k_sq = 0.0f;
    for (uint i = 0; i < hd; i++) k_sq += kk_raw[i] * kk_raw[i];
    const float k_rstd = rsqrt(k_sq / float(hd) + 1e-5f);
    const device float* KCOS = ROPE + k * 48;
    const device float* KSIN = ROPE + rows * 48 + k * 48;
    for (uint i = 0; i < hd; i++) {
      float kv;
      if (i < 48) {
        const uint half_dim = 24;
        const uint fi = i % half_dim;
        const float c = KCOS[fi];
        const float sn = KSIN[fi];
        if (i < half_dim) {
          kv = kk_raw[i] * k_rstd * c - kk_raw[i + half_dim] * k_rstd * sn;
        } else {
          kv = kk_raw[i] * k_rstd * c + kk_raw[i - half_dim] * k_rstd * sn;
        }
      } else {
        kv = kk_raw[i] * k_rstd;
      }
      dot += q_rot[i] * kv;
    }
    dot *= rsqrt(float(hd));
    if (dot > gmax) gmax = dot;
  }
  // pass 2: exp weights sum
  float run_sum = 0.0f;
  for (uint k = 0; k < rows; k++) {
    float dot = 0.0f;
    const device float* kk_raw = QKV + k * 3 * dim + head * 3 * hd + hd;
    float k_sq = 0.0f;
    for (uint i = 0; i < hd; i++) k_sq += kk_raw[i] * kk_raw[i];
    const float k_rstd = rsqrt(k_sq / float(hd) + 1e-5f);
    const device float* KCOS = ROPE + k * 48;
    const device float* KSIN = ROPE + rows * 48 + k * 48;
    for (uint i = 0; i < hd; i++) {
      float kv;
      if (i < 48) {
        const uint half_dim = 24;
        const uint fi = i % half_dim;
        const float c = KCOS[fi];
        const float sn = KSIN[fi];
        if (i < half_dim) {
          kv = kk_raw[i] * k_rstd * c - kk_raw[i + half_dim] * k_rstd * sn;
        } else {
          kv = kk_raw[i] * k_rstd * c + kk_raw[i - half_dim] * k_rstd * sn;
        }
      } else {
        kv = kk_raw[i] * k_rstd;
      }
      dot += q_rot[i] * kv;
    }
    run_sum += exp(dot * rsqrt(float(hd)) - gmax);
  }
  // pass 3: weighted value accumulation
  float acc[64];
  for (uint i = 0; i < hd; i++) acc[i] = 0.0f;
  for (uint k = 0; k < rows; k++) {
    float dot = 0.0f;
    const device float* kk_raw = QKV + k * 3 * dim + head * 3 * hd + hd;
    float k_sq = 0.0f;
    for (uint i = 0; i < hd; i++) k_sq += kk_raw[i] * kk_raw[i];
    const float k_rstd = rsqrt(k_sq / float(hd) + 1e-5f);
    const device float* KCOS = ROPE + k * 48;
    const device float* KSIN = ROPE + rows * 48 + k * 48;
    for (uint i = 0; i < hd; i++) {
      float kv;
      if (i < 48) {
        const uint half_dim = 24;
        const uint fi = i % half_dim;
        const float c = KCOS[fi];
        const float sn = KSIN[fi];
        if (i < half_dim) {
          kv = kk_raw[i] * k_rstd * c - kk_raw[i + half_dim] * k_rstd * sn;
        } else {
          kv = kk_raw[i] * k_rstd * c + kk_raw[i - half_dim] * k_rstd * sn;
        }
      } else {
        kv = kk_raw[i] * k_rstd;
      }
      dot += q_rot[i] * kv;
    }
    const float wgt = exp(dot * rsqrt(float(hd)) - gmax);
    const device float* vv = QKV + k * 3 * dim + head * 3 * hd + 2 * hd;
    for (uint i = 0; i < hd; i++) acc[i] += wgt * vv[i];
  }
  if (P[1] == 1) {
    // debug mode: write q_rot directly
    for (uint i = 0; i < hd; i++) Y[r * dim + head * hd + i] = q_rot[i];
    return;
  }
  if (P[1] == 2) {
    // debug: write first 64 dots for this (r, head)
    float dots_dbg[64];
    for (uint i = 0; i < 64; i++) dots_dbg[i] = 0.0f;
    for (uint k = 0; k < rows && k < 64; k++) {
      float dot = 0.0f;
      const device float* kk_raw = QKV + k * 3 * dim + head * 3 * hd + hd;
      float k_sq = 0.0f;
      for (uint i = 0; i < hd; i++) k_sq += kk_raw[i] * kk_raw[i];
      const float k_rstd = rsqrt(k_sq / float(hd) + 1e-5f);
      const device float* KCOS = ROPE + k * 48;
      const device float* KSIN = ROPE + rows * 48 + k * 48;
      for (uint i = 0; i < hd; i++) {
        float kv;
        if (i < 48) {
          const uint fi = i % 24;
          const float c = KCOS[fi];
          const float sn = KSIN[fi];
          if (i < 24) kv = kk_raw[i] * k_rstd * c - kk_raw[i + 24] * k_rstd * sn;
          else kv = kk_raw[i] * k_rstd * c + kk_raw[i - 24] * k_rstd * sn;
        } else kv = kk_raw[i] * k_rstd;
        dot += q_rot[i] * kv;
      }
      dots_dbg[k] = dot * rsqrt(float(hd));
    }
    for (uint i = 0; i < hd; i++) Y[r * dim + head * hd + i] = dots_dbg[i];
    return;
  }
  if (P[1] == 4) {
    // debug: write rotated key k = (r % 64) for this head
    const uint kdbg = r % 64;
    const device float* kk_raw = QKV + kdbg * 3 * dim + head * 3 * hd + hd;
    float k_sq = 0.0f;
    for (uint i = 0; i < hd; i++) k_sq += kk_raw[i] * kk_raw[i];
    const float k_rstd = rsqrt(k_sq / float(hd) + 1e-5f);
    const device float* KCOS = ROPE + kdbg * 48;
    const device float* KSIN = ROPE + rows * 48 + kdbg * 48;
    for (uint i = 0; i < hd; i++) {
      float kv;
      if (i < 48) {
        const uint fi = i % 24;
        const float c = KCOS[fi];
        const float sn = KSIN[fi];
        if (i < 24) kv = kk_raw[i] * k_rstd * c - kk_raw[i + 24] * k_rstd * sn;
        else kv = kk_raw[i] * k_rstd * c + kk_raw[i - 24] * k_rstd * sn;
      } else kv = kk_raw[i] * k_rstd;
      Y[r * dim + head * hd + i] = kv;
    }
    return;
  }
  if (P[1] == 3) {
    // debug mode: read back RoPE tables as seen by this thread
    const device float* COSD = ROPE + r * 48;
    const device float* SIND = ROPE + rows * 48 + r * 48;
    for (uint i = 0; i < hd; i++) {
      Y[r * dim + head * hd + i] = (i < 24) ? COSD[i] : SIND[i % 24];
    }
    return;
  }
  for (uint i = 0; i < hd; i++) Y[r * dim + head * hd + i] = run_sum > 0.0f ? acc[i] / run_sum : 0.0f;
}

// LayerNorm affine: mean/var in fp32, then (x - mean) / std * w + b.
[[kernel]] void h3_vae_layernorm(
  const device float* X [[buffer(0)]],
  const device float* W [[buffer(1)]],
  const device float* B [[buffer(2)]],
  device float* Y [[buffer(3)]],
  constant uint* P [[buffer(4)]],        // P[0] = D, P[1] = eps as bits? no: eps folded into P[1] unused; eps is compile-time 1e-5
  uint tid [[thread_index_in_threadgroup]],
  uint tg [[threadgroup_position_in_grid]]) {
  const uint D = P[0];
  threadgroup float red[256];
  float sum = 0.0f;
  for (uint i = tid; i < D; i += 256) sum += X[tg * D + i];
  red[tid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride /= 2) {
    if (tid < stride) red[tid] += red[tid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float mean = red[0] / float(D);
  float m2 = 0.0f;
  for (uint i = tid; i < D; i += 256) { const float t = X[tg * D + i] - mean; m2 += t * t; }
  red[tid] = m2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride /= 2) {
    if (tid < stride) red[tid] += red[tid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float rstd = rsqrt(red[0] / float(D) + 1e-5f);
  for (uint i = tid; i < D; i += 256) Y[tg * D + i] = (X[tg * D + i] - mean) * rstd * W[i] + B[i];
}

// post_quant 1x1x1 conv: out[v, o] = sum_i in[v, i] w[o, i] + b[o]. Threads [V, O, 1].
[[kernel]] void h3_vae_postquant(
  const device float* IN [[buffer(0)]],   // [V, 24]
  const device float* W [[buffer(1)]],    // [24, 24]
  const device float* B [[buffer(2)]],    // [24]
  device float* OUT [[buffer(3)]],        // [V, 24]
  constant uint* P [[buffer(4)]],        // P[0]=V, P[1]=C
  uint at1 [[thread_position_in_grid]]) {
  const uint V = P[0], C = P[1];
  const uint v = at1 / C;
  const uint o = at1 % C;
  if (v >= V || o >= C) return;
  float sum = 0.0f;
  for (uint i = 0; i < C; i++) sum += IN[v * C + i] * W[o * C + i];
  OUT[v * C + o] = sum + B[o];
}

// unpatchify + denormalize: token output to RGB bytes, one thread per byte.
[[kernel]] void h3_vae_unpatch(
  const device float* PROJ [[buffer(0)]],
  const device float* MEAN [[buffer(1)]],
  const device float* STD [[buffer(2)]],
  device uchar* RGB [[buffer(3)]],
  constant uint* P [[buffer(4)]],
  uint at1 [[thread_position_in_grid]]) {
  const uint depth = P[0], h = P[1], w = P[2], out_w = P[3], hb = P[4];
  const uint pt = 4, p = 16;
  const uint total_cols = w * p * 3;
  const uint total_rows = h * p;
  const uint f = at1 / (total_rows * total_cols);
  const uint rem = at1 % (total_rows * total_cols);
  const uint row = rem / total_cols;
  const uint col3 = rem % total_cols;
  const uint t = f / pt;
  const uint dt = f % pt;
  const uint hh = row / p;
  const uint dh = row % p;
  const uint c = col3 % 3;
  const uint col = col3 / 3;
  const uint ww = col / p;
  const uint dw = col % p;
if (t >= depth || hh >= h || ww >= w) return;
  const uint v = (t * h + hh) * w + ww;
  const float o = PROJ[v * hb + c * pt * p * p + dt * p * p + dh * p + dw];
  const float value = o * STD[c] + MEAN[c];
  RGB[(f * h * p + row) * out_w + col3] = (uchar)clamp(value * 255.0f, 0.0f, 255.0f);
}


// V extraction from the interleaved qkv layout, one thread per element.
[[kernel]] void h3_vae_vextract(
  const device float* QKV [[buffer(0)]],
  device float* OUT [[buffer(1)]],
  constant uint* P [[buffer(2)]],          // P[0] = rows
  uint3 at [[thread_position_in_grid]]) {
  const uint rows = P[0];
  const uint hd = 64, heads = 32, dim = heads * hd;
  const uint r = at.y;
  const uint i = at.x % hd;
  const uint h = at.x / hd;
  OUT[(h * rows + r) * hd + i] = QKV[r * 3 * dim + h * 3 * hd + 2 * hd + i];
}
[[kernel]] void h3_vae_enc_conv(
  const device float* IN [[buffer(0)]],
  const device float* W [[buffer(1)]],
  const device float* B [[buffer(2)]],
  device float* OUT [[buffer(3)]],
  constant uint* P [[buffer(4)]],
  uint3 at3 [[thread_position_in_grid]]) {
  const uint c_in = P[0], t = P[1], hh = P[2], ww = P[3], c_out = P[4];
  const uint st = P[5], sh = P[6], sw = P[7];
  const uint causal = P[8];
  const uint pad_l = P[9], pad_r = P[10], pad_b = P[11];
  const uint kt = 3, kh = 3, kw = 3;
  const uint pt = kt / 2;
  const uint t_pre = (causal == 1) ? pt * 2 : pt;
  const uint tp = t + t_pre;
  const uint hp = hh + pad_l + pad_r;
  const uint wp = ww + pad_l + pad_r;
  const uint co = at3.x, oh = at3.y, ozw = at3.z;
  const uint out_w_calc = (wp - kw) / sw + 1;
  const uint ot = ozw / out_w_calc;
  const uint ow = ozw % out_w_calc;
  if (co >= c_out) return;
  if (ot >= (tp - kt) / st + 1) return;
  if (oh >= (hp - kh) / sh + 1) return;
  if (ow >= out_w_calc) return;
  float sum = B[co];
  for (uint ci = 0; ci < c_in; ci++) {
    for (uint ktt = 0; ktt < kt; ktt++) {
      const uint tt = ot * st + ktt;
      // causal temporal: source frame index = tt - t_pre
      if (tt < t_pre) continue;
      const uint stt = tt - t_pre;
      if (stt >= t) continue;
      for (uint khh = 0; khh < kh; khh++) {
        // spatial: source = oh*sh + khh - pad_l
        const int shh = (int)(oh * sh + khh) - (int)pad_l;
        if (shh < 0 || shh >= (int)hh) continue;
        for (uint kww = 0; kww < kw; kww++) {
          const int sww = (int)(ow * sw + kww) - (int)pad_l;
          if (sww < 0 || sww >= (int)ww) continue;
          const float iv = IN[(ci * t + stt) * hh * ww + shh * ww + sww];
          const float wv = W[((co * c_in + ci) * kt + ktt) * kh * kw + khh * kw + kww];
          sum += iv * wv;
        }
      }
    }
  }
  const uint out_t = (tp - kt) / st + 1;
  const uint out_h = (hp - kh) / sh + 1;
  const uint out_w = (wp - kw) / sw + 1;
  OUT[(co * out_t + ot) * out_h * out_w + oh * out_w + ow] = sum;
}

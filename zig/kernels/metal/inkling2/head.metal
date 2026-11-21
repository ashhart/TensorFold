// Inkling 2 layer residual joins and the logit softcap, elementwise over rows of the given width.

struct I2ElemArgs {
  uint rows, cols, in_ld, out_ld;
  uint has_block, pad0, pad1, pad2;
  float cap;
};

// Inkling2Softcap on the lm_head's bf16 logits, kept in fp32: out = tanh(float(y) / cap) * cap.
[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_softcap(
    const device bfloat* y [[buffer(0)]], device float* out [[buffer(1)]], constant I2ElemArgs& a [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  if (at.x >= a.cols || at.y >= a.rows) return;
  out[ulong(at.y) * a.out_ld + at.x] = metal::precise::tanh(float(y[ulong(at.y) * a.in_ld + at.x]) / a.cap) * a.cap;
}

// The attention half's partial state: the post-attention conv output as fp32, plus the layer's input unless the
// layer starts a block (has_block == 0): out = (has_block ? float(block) : 0) + float(a); a fp32 sum, never rounded.
[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_residual_attn(
    const device bfloat* conv [[buffer(0)]], const device bfloat* block [[buffer(1)]], device float* out [[buffer(2)]],
    constant I2ElemArgs& a [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= a.cols || at.y >= a.rows) return;
  const ulong i = ulong(at.y) * a.cols + at.x;
  const float c = float(conv[i]);
  out[i] = a.has_block != 0 ? float(block[i]) + c : c;
}

// The layer's output: bf16(partial + float(post-MLP conv output)).
[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_residual_mlp(
    const device float* h [[buffer(0)]], const device bfloat* conv [[buffer(1)]], device bfloat* out [[buffer(2)]],
    constant I2ElemArgs& a [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= a.cols || at.y >= a.rows) return;
  const ulong i = ulong(at.y) * a.cols + at.x;
  out[i] = bfloat(h[i] + float(conv[i]));
}

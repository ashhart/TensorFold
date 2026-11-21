// Inkling 2 short causal convolutions (Inkling2ShortConvolution, Inkling2DynamicShortConvolution), four taps.
// A stream's history is a ring of `ring` (32) input rows indexed by absolute position: row t of a call sits at
// position pos0 + t, the three rows before it come from the call's own rows or, before row 0, from slot p % ring,
// and a position below 0 reads as zero. Reads of history never depend on stale slots, so rolling a stream back to an
// earlier position needs no repair. convStore (inkling2_conv_store) is a separate launch after the conv.

constant constexpr uint I2_CONV_TAPS = 4;

struct I2ConvArgs {
  uint rows, channels, pos0, ring;
  uint kv_heads, pad0, pad1, pad2;
  float divisor; // sqrt(taps + 1) for the static conv, sqrt(2 * taps + 1) for the dynamic one
};

// The input `lag` rows before call row t (lag 0 is row t itself), in fp32.
inline float i2_conv_in(const device bfloat* x, const device bfloat* hist, constant I2ConvArgs& a, uint t, uint lag,
                        uint c) {
  if (t >= lag) return float(x[ulong(t - lag) * a.channels + c]);
  const long position = long(a.pos0) + long(t) - long(lag);
  if (position < 0) return 0.0f;
  return float(hist[ulong(position % long(a.ring)) * a.channels + c]);
}

// y[t][c] = (sum_j w[c][j] * x[t - 3 + j] + x[t]) / sqrt(taps + 1); w is the checkpoint's [channels, 1, 4] conv1d
// weight, tap 3 weighing the current row. Taps run oldest first, one fp32 chain.
[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_conv_static(
    const device bfloat* x [[buffer(0)]], const device bfloat* hist [[buffer(1)]], const device bfloat* w [[buffer(2)]],
    device bfloat* y [[buffer(3)]], constant I2ConvArgs& a [[buffer(4)]], uint2 at [[thread_position_in_grid]]) {
  const uint c = at.x, t = at.y;
  if (c >= a.channels || t >= a.rows) return;
  float sum = 0.0f;
  for (uint j = 0; j < I2_CONV_TAPS; ++j) sum = fma(float(w[ulong(c) * I2_CONV_TAPS + j]), i2_conv_in(x, hist, a, t, I2_CONV_TAPS - 1 - j, c), sum);
  y[ulong(t) * a.channels + c] = bfloat((sum + float(x[ulong(t) * a.channels + c])) / a.divisor);
}

// The key/value convolution of a full-attention layer: tap weights are static[lag][c] + dyn[t][head][lag], lag 0 the
// current row (the reference's flip), shared by the channels of a key/value head; (sum + x[t]) / sqrt(2 * taps + 1).
// static_w is [4, channels]; dyn is the weight projection's bf16 output [rows, kv_heads * 4].
[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_conv_dynamic(
    const device bfloat* x [[buffer(0)]], const device bfloat* hist [[buffer(1)]],
    const device bfloat* static_w [[buffer(2)]], const device bfloat* dyn [[buffer(3)]], device bfloat* y [[buffer(4)]],
    constant I2ConvArgs& a [[buffer(5)]], uint2 at [[thread_position_in_grid]]) {
  const uint c = at.x, t = at.y;
  if (c >= a.channels || t >= a.rows) return;
  const uint head = c / (a.channels / a.kv_heads);
  float sum = 0.0f;
  for (uint lag = 0; lag < I2_CONV_TAPS; ++lag) {
    const float weight = float(static_w[ulong(lag) * a.channels + c]) +
                         float(dyn[ulong(t) * a.kv_heads * I2_CONV_TAPS + head * I2_CONV_TAPS + lag]);
    sum = fma(weight, i2_conv_in(x, hist, a, t, lag, c), sum);
  }
  y[ulong(t) * a.channels + c] = bfloat((sum + float(x[ulong(t) * a.channels + c])) / a.divisor);
}

// hist[(pos0 + t) % ring] = x[t] for the newest min(rows, ring) rows. Its own launch: the conv above must have read
// the old history of every row first.
[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_conv_store(
    const device bfloat* x [[buffer(0)]], device bfloat* hist [[buffer(1)]], constant I2ConvArgs& a [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  const uint c = at.x;
  if (c >= a.channels) return;
  const uint kept = min(a.rows, a.ring);
  if (at.y >= kept) return;
  const uint t = a.rows - kept + at.y;
  hist[ulong((a.pos0 + t) % a.ring) * a.channels + c] = x[ulong(t) * a.channels + c];
}

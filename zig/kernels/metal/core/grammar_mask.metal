// Structured output: a row's tokens its grammar disallows become -inf (bf16 0xff80) before the row's draw.
#include <metal_stdlib>
using namespace metal;

// dims: vocab, words a row, rows; bits: token t at bit t % 32 of word t / 32; one thread a word of 32 logits.
kernel void tf_grammar_mask(device ushort* logits [[buffer(0)]],
                            const device uint* bits [[buffer(1)]],
                            constant uint4& dims [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
  const uint w = gid.x, r = gid.y;
  if (w >= dims.y || r >= dims.z) return;
  const uint allowed = bits[size_t(r) * dims.y + w];
  if (allowed == 0xffffffffu) return;
  device ushort* row = logits + size_t(r) * dims.x;
  for (uint b = 0; b < 32; b++) {
    const uint t = w * 32 + b;
    if (t < dims.x && ((allowed >> b) & 1u) == 0) row[t] = 0xff80;
  }
}

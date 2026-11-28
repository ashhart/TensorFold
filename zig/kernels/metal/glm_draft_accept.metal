// GLM's DFlash2 chains accepted on the GPU: a stream's kept drafts end at the first row whose pick differs.
#include <metal_stdlib>
using namespace metal;

// stream s: rows[s] rows at offsets[s] of tokens and picks; out[2s] drafts kept, out[2s + 1] the next pending token
kernel void glm_draft_accept(device const uint* tokens [[buffer(0)]], device const uint* picks [[buffer(1)]],
                             device const uint* offsets [[buffer(2)]], device const uint* rows [[buffer(3)]],
                             device uint* out [[buffer(4)]], constant uint& streams [[buffer(5)]],
                             uint s [[thread_position_in_grid]]) {
  if (s >= streams) return;
  const uint at = offsets[s];
  const uint n = rows[s];
  uint kept = 0;
  while (kept + 1 < n && tokens[at + kept + 1] == picks[at + kept]) kept++;
  out[2 * s] = kept;
  out[2 * s + 1] = picks[at + kept];
}

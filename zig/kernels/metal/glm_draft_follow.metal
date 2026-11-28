// GLM's DFlash2 block after a window accepted on the GPU: its ids, rope positions and attention mask from the count.
#include <metal_stdlib>
using namespace metal;

// a stream's plan: window rows from `first`, context keys [lo, lo + context) then its block rows, `begin` its start
struct glm_follow_stream { ulong first, lo, begin; uint context, keys; };
struct glm_follow_args { uint block, streams, trained, window, mask_token, mask_stride, rows, pad; };

// thread (key, row): row = stream * block + q; accepted[2s] drafts kept, accepted[2s + 1] the block's anchor
kernel void glm_draft_follow(device const uint* accepted [[buffer(0)]], constant glm_follow_args& a [[buffer(1)]],
                             constant glm_follow_stream* st [[buffer(2)]], device uint* ids [[buffer(3)]],
                             device ulong* positions [[buffer(4)]], device uchar* mask [[buffer(5)]],
                             uint2 at [[thread_position_in_grid]]) {
  const uint key = at.x;
  const uint row = at.y;
  if (key >= a.mask_stride || row >= a.rows) return;
  const uint s = row / a.block;
  const uint q = row % a.block;
  const glm_follow_stream t = st[s];
  const ulong end = t.first + accepted[2 * s] + 1;
  if (key == 0) {
    ids[row] = q == 0 ? accepted[2 * s + 1] : a.mask_token;
    positions[row] = end + q;
  }
  const ulong reach = end + q + 1;
  const ulong from = max(t.begin, reach > a.window ? reach - a.window : 0ul);
  uchar visible = 0;
  if (key < t.context) {
    const ulong p = t.lo + key;
    visible = p >= from && p < end;
  } else if (key < t.keys) {
    visible = !(q < a.trained && a.block > a.trained && key - t.context >= a.trained);
  }
  mask[row * a.mask_stride + key] = visible;
}

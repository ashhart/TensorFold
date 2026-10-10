// Accepted metadata and activation movement copy only authoritative verified-path indices and BF16 bits.
#include <metal_stdlib>
using namespace metal;
struct TfCommitResult { uint status, stop, nonfinite, consumed_count, emitted_count, matched_count, bonus_emitted, pending_valid, pending_token, reserved; uint path[16]; uint tokens[16]; };
struct TfCommitMeta { uint rows, slot; };
struct TfCommitTaps { uint rows, width, capacity, planes; };
struct TfCommitHead { uint rows, width, stride; };
struct TfCommitKeep { uint first, rows, state_slot, next_slot; };
struct TfCommitMap { uint stream, source, destination, row; };
inline bool tf_commit_valid(device const TfCommitResult& r, uint rows) {
  if (r.status || r.nonfinite || r.reserved || !r.consumed_count || r.consumed_count > rows || r.consumed_count != r.emitted_count || r.path[0] != 0) return false;
  for (uint i = 0; i < r.consumed_count; ++i) if (r.path[i] >= rows || (i && r.path[i] <= r.path[i - 1])) return false;
  return true;
}
kernel void tf_round_keep_metadata(device const TfCommitResult& result [[buffer(0)]], device TfCommitKeep* keep [[buffer(1)]], device uint* rows [[buffer(2)]], device TfCommitMap* map [[buffer(3)]], constant TfCommitMeta& p [[buffer(4)]], constant uint& with_map [[buffer(5)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.rows) return;
  const uint count = tf_commit_valid(result, p.rows) ? result.consumed_count : 0;
  if (i == 0) keep[0] = {0, count, p.slot, p.slot};
  rows[i] = i < count ? result.path[i] : 0;
  if (with_map) map[i] = {i < count ? 0u : 0xffffffffu, i < count ? result.path[i] : 0u, i, i};
}
kernel void tf_round_keep_taps(device const TfCommitResult& result [[buffer(0)]], device const ushort* taps [[buffer(1)]], device ushort* out [[buffer(2)]], constant TfCommitTaps& p [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  if (!tf_commit_valid(result, p.rows) || at.y >= result.consumed_count || at.x >= p.width * p.planes) return;
  const uint plane = at.x / p.width, column = at.x % p.width;
  out[ulong(at.y) * p.width * p.planes + at.x] = taps[(ulong(plane) * p.capacity + result.path[at.y]) * p.width + column];
}
kernel void tf_round_keep_head(device const TfCommitResult& result [[buffer(0)]], device ushort* logits [[buffer(1)]], constant TfCommitHead& p [[buffer(2)]], uint column [[thread_position_in_grid]]) {
  if (!tf_commit_valid(result, p.rows) || column >= p.width) return;
  const uint row = result.path[result.consumed_count - 1];
  logits[column] = logits[ulong(row) * p.stride + column];
}

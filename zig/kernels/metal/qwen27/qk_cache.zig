//! A cache-writing twin of the head norm/RoPE kernel, with its arithmetic retained verbatim.
const std = @import("std");
const edit = @import("../core/source_edit.zig").edit;
const raw = @embedFile("glue.metal");
const start = std.mem.indexOf(u8, raw, "kernel void q27_qk_norm_rope(").?;

/// Flat or paged destinations; the enclosing attention library supplies the cache pointer type and helpers.
pub fn source(comptime pages: bool) []const u8 {
    var s = edit(raw[start..], "q27_qk_norm_rope(", "q27_qk_cache_rope(", 1);
    s = edit(s, "                            uint2 group [[threadgroup_position_in_grid]],",
        \\                            device Q27CachePtrs* caches [[buffer(10)]], device const int* meta [[buffer(11)]],
        \\                            device const int* nodes [[buffer(12)]], device const int* q_rows [[buffer(13)]],
        \\                            device bfloat* QA [[buffer(14)]],
        \\                            uint2 group [[threadgroup_position_in_grid]],
    , 1);
    s = edit(s, "  const uint threads = thread_shape.x;",
        \\  const uint threads = thread_shape.x;
        \\  const uint st = uint(nodes[4 * group.y + 1]), mb = 8 + st * 12;
        \\  const uint first_node = uint(meta[mb + 6]), local_node = group.y - first_node;
        \\  const uint cache_row = uint(meta[mb + 4]) + local_node;
        \\  if (group.x >= a.heads + a.kv_heads) {
        \\    const uint hk = group.x - a.heads - a.kv_heads;
        \\    for (uint at = t; at < a.width; at += threads)
        \\      caches[st].values[ulong(hk) * uint(meta[mb + 8]) + cache_row * a.width + at] =
        \\          ((device const bfloat*)K)[group.y * a.key_stride + (a.kv_heads + hk) * a.width + at];
        \\    return;
        \\  }
    , 1);
    s = edit(s, "    q27_write(Y, row * a.width + at, value, 0);",
        \\    q27_write(Y, row * a.width + at, value, 0);
        \\    if (!is_q) {
        \\      caches[st].keys[ulong(head) * uint(meta[mb + 7]) + cache_row * a.width + at] = bfloat(value);
        \\    } else {
        \\      const uint G = a.heads / a.kv_heads, hk = head / G, local_head = head % G;
        \\      const uint first_slot = uint(meta[mb + 3]) * 16, live = uint(meta[mb + 2]) * G;
        \\      const uint mapped = group.y * G + local_head;
        \\      const uint slot = first_slot + local_node * G + local_head;
        \\      if (uint(q_rows[slot]) == mapped)
        \\        QA[(hk * uint(meta[5]) + slot) * a.width + at] = bfloat(value);
        \\      // Plan.init repeats the first query head of each stream in all its tile padding.
        \\      if (local_node == 0 && local_head == 0) {
        \\        for (uint pad = first_slot + live; pad < first_slot + ((live + 15) / 16) * 16; ++pad)
        \\          if (uint(q_rows[pad]) == mapped)
        \\            QA[(hk * uint(meta[5]) + pad) * a.width + at] = bfloat(value);
        \\      }
        \\    }
    , 1);
    if (pages) {
        s = edit(s, "caches[st].keys[ulong(head) * uint(meta[mb + 7]) + cache_row * a.width + at]", "q27_paged(caches[st].keys, caches[st], head, cache_row, a.width)[at]", 1);
        s = edit(s, "caches[st].values[ulong(hk) * uint(meta[mb + 8]) + cache_row * a.width + at]", "q27_paged(caches[st].values, caches[st], hk, cache_row, a.width)[at]", 1);
    }
    return s;
}

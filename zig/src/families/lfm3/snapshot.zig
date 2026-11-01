//! Kept prompt states: a stream's conv states and its keys and values through `at` tokens, copied to host memory.
const std = @import("std");
const c = @import("config.zig");
const st = @import("state.zig");

/// One state: conv states by conv layer, then keys and values [kv head][at][head dim] by attention layer.
pub const Snap = struct { at: u32, bytes: []u8 };

/// A row's keys or values for every kv head of one attention layer.
const row_bytes = c.kv_heads * c.head_dim * 2;

pub fn bytes(layer_types: *const [c.layers]c.LayerType, at: u32) u64 {
    var n: u64 = 0;
    for (layer_types) |kind| n += if (kind != .conv) 2 * @as(u64, at) * row_bytes else st.conv_bytes;
    return n;
}

/// Copy `cache`'s state after `at` tokens (cache.len == at, nothing in flight).
pub fn save(gpa: std.mem.Allocator, layer_types: *const [c.layers]c.LayerType, cache: *const st.Cache, at: u32) !*Snap {
    if (cache.len != at or at > cache.capacity) return error.LfmSnapshotPosition;
    const snap = try gpa.create(Snap);
    errdefer gpa.destroy(snap);
    snap.* = .{ .at = at, .bytes = try gpa.alloc(u8, bytes(layer_types, at)) };
    var off: usize = 0;
    for (cache.blocks) |b| switch (b) {
        .conv => |conv| {
            @memcpy(snap.bytes[off..][0..st.conv_bytes], conv.contents()[0..st.conv_bytes]);
            off += st.conv_bytes;
        },
        .attention => |kv| for ([_]@TypeOf(kv.keys){ kv.keys, kv.values }) |buf| for (0..c.kv_heads) |h| {
            const n = at * c.head_dim * 2;
            @memcpy(snap.bytes[off..][0..n], (buf.contents() + h * cache.capacity * c.head_dim * 2)[0..n]);
            off += n;
        },
    };
    return snap;
}

/// Make a fresh `cache` hold `snap`'s state (cache.len becomes snap.at).
pub fn restore(cache: *st.Cache, snap: *const Snap) !void {
    if (cache.len != 0 or snap.at > cache.capacity) return error.LfmSnapshotPosition;
    var expected: usize = 0;
    for (cache.blocks) |b| expected += if (b == .conv) st.conv_bytes else 2 * snap.at * row_bytes;
    if (snap.bytes.len != expected) return error.LfmSnapshotSize;
    var off: usize = 0;
    for (&cache.blocks) |*b| switch (b.*) {
        .conv => |conv| {
            @memcpy(conv.contents()[0..st.conv_bytes], snap.bytes[off..][0..st.conv_bytes]);
            off += st.conv_bytes;
        },
        .attention => |kv| for ([_]@TypeOf(kv.keys){ kv.keys, kv.values }) |buf| for (0..c.kv_heads) |h| {
            const n = snap.at * c.head_dim * 2;
            @memcpy((buf.contents() + h * cache.capacity * c.head_dim * 2)[0..n], snap.bytes[off..][0..n]);
            off += n;
        },
    };
    cache.len = snap.at;
    cache.last = null;
}

pub fn drop(gpa: std.mem.Allocator, snap: *Snap) void {
    gpa.free(snap.bytes);
    gpa.destroy(snap);
}

test "snapshots store full-addressed KV for full and sliding layers and seven-tap convolution history" {
    try std.testing.expectEqual(@as(u64, 13 * st.conv_bytes), bytes(&c.schedule, 0));
    try std.testing.expectEqual(@as(u64, 11 * 2 * row_bytes), bytes(&c.schedule, 1) - bytes(&c.schedule, 0));
}

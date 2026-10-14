//! Kept prompt states: a stream's conv states and its keys and values through `at` tokens, copied to host memory.
const std = @import("std");
const c = @import("config.zig");
const st = @import("state.zig");

/// One state: conv states by conv layer, then keys and values [kv head][at][head dim] by attention layer.
pub const Snap = struct { at: u32, bytes: []u8 };

/// A row's keys or values for every kv head of one attention layer.
const row_bytes = c.kv_heads * c.head_dim * 2;

pub fn bytes(attention: *const [c.layers]bool, at: u32) u64 {
    var n: u64 = 0;
    for (attention) |a| n += if (a) 2 * @as(u64, at) * row_bytes else st.conv_bytes;
    return n;
}

/// Copy `cache`'s state after `at` tokens (cache.len == at, nothing in flight).
pub fn save(gpa: std.mem.Allocator, attention: *const [c.layers]bool, cache: *const st.Cache, at: u32) !*Snap {
    if (cache.len != at) return error.LfmSnapshotPosition;
    const snap = try gpa.create(Snap);
    errdefer gpa.destroy(snap);
    snap.* = .{ .at = at, .bytes = try gpa.alloc(u8, bytes(attention, at)) };
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

test "a state's bytes: fixed conv states, then a key and value row per kv head and attention layer a token" {
    var attention: [c.layers]bool = @splat(false);
    attention[2] = true;
    attention[6] = true;
    const fixed = (c.layers - 2) * st.conv_bytes;
    try std.testing.expectEqual(@as(u64, fixed), bytes(&attention, 0));
    try std.testing.expectEqual(@as(u64, 2 * 2 * c.kv_heads * c.head_dim * 2), bytes(&attention, 1) - bytes(&attention, 0));
}

//! Kept prompt states: a stream's keys and values through `at` tokens, copied to host memory.
const std = @import("std");
const c = @import("config.zig");
const st = @import("state.zig");

/// One state: keys and values [kv head][at][head dim] by layer.
pub const Snap = struct { at: u32, bytes: []u8 };

pub fn bytes(at: u32) u64 {
    return @as(u64, at) * st.row_bytes;
}

/// Copy `cache`'s state after `at` tokens (cache.len == at, nothing in flight).
pub fn save(gpa: std.mem.Allocator, cache: *const st.Cache, at: u32) !*Snap {
    if (cache.len != at) return error.CosmosSnapshotPosition;
    const snap = try gpa.create(Snap);
    errdefer gpa.destroy(snap);
    snap.* = .{ .at = at, .bytes = try gpa.alloc(u8, bytes(at)) };
    var off: usize = 0;
    for (cache.blocks) |kv| for ([_]@TypeOf(kv.keys){ kv.keys, kv.values }) |buf| for (0..c.kv_heads) |h| {
        const n = at * c.head_dim * 2;
        @memcpy(snap.bytes[off..][0..n], (buf.contents() + h * cache.capacity * c.head_dim * 2)[0..n]);
        off += n;
    };
    return snap;
}

/// Make a fresh `cache` hold `snap`'s state (cache.len becomes snap.at).
pub fn restore(cache: *st.Cache, snap: *const Snap) !void {
    if (cache.len != 0 or snap.at > cache.capacity) return error.CosmosSnapshotPosition;
    var off: usize = 0;
    for (&cache.blocks) |*kv| for ([_]@TypeOf(kv.keys){ kv.keys, kv.values }) |buf| for (0..c.kv_heads) |h| {
        const n = snap.at * c.head_dim * 2;
        @memcpy((buf.contents() + h * cache.capacity * c.head_dim * 2)[0..n], snap.bytes[off..][0..n]);
        off += n;
    };
    cache.len = snap.at;
    cache.last = null;
}

pub fn drop(gpa: std.mem.Allocator, snap: *Snap) void {
    gpa.free(snap.bytes);
    gpa.destroy(snap);
}

//! Qwen3.5 CUDA prompt states for the shared cache in pinned host memory: recurrences, tails, keys, values, rope shift.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const geometry = @import("cuda_geometry.zig");
const engine = @import("cuda_engine.zig");

const Engine = engine.Engine;
const Cache = engine.Cache;
const Geometry = geometry.Geometry;
const head_dim = geometry.head_dim;

/// One kept state: its bytes on the host, the position it stands at, its rotary shift.
pub const Copy = struct {
    host: cuda.HostBuffer,
    at: usize,
    delta: i32,
};

/// The engine and the streams' caches a save copies (cuda_native.zig's Lanes owns the map).
pub const Target = struct {
    e: *Engine,
    caches: *std.AutoHashMapUnmanaged(*lanes.Stream, *Cache),
};

/// One piece of a stream's state: where it lives on the device, its length.
const Piece = struct { dev: u64, len: usize };

/// Key heads a kept state copies at most, and its pieces: the state, then a key and a value block a (layer, head).
const max_kv_heads = 8;
const max_pieces = 1 + 2 * geometry.max_layers * max_kv_heads;

/// A state's pieces: recurrences and tails whole, then each (layer, key/value, head)'s first `at` rows; null: lengths.
fn pieces(g: Geometry, cache: ?*const Cache, at: usize, out: *[max_pieces]Piece) []const Piece {
    const row = head_dim * 2;
    out[0] = .{ .dev = if (cache) |c| c.state.ptr else 0, .len = g.stateBytes() };
    var n: usize = 1;
    for (0..g.attentions()) |l| for (0..2) |kv| for (0..g.kv_heads) |h| {
        // qwen35.cu's layout: a layer's keys then values, kv_heads blocks of cap rows each
        const block: usize = if (cache) |c| ((l * 2 + kv) * g.kv_heads + h) * c.cap * row else 0;
        out[n] = .{ .dev = if (cache) |c| c.kv.ptr + block else 0, .len = at * row };
        n += 1;
    };
    return out[0..n];
}

/// A state's bytes at `at` tokens.
fn total(g: Geometry, at: usize) u64 {
    var buf: [max_pieces]Piece = undefined;
    var n: u64 = 0;
    for (pieces(g, null, at, &buf)) |p| n += p.len;
    return n;
}

/// Whether this engine's states fit a kept state's pieces (cuda_native.zig offers no cache otherwise).
pub fn fits(g: Geometry) bool {
    return g.kv_heads <= max_kv_heads;
}

/// Host bytes a save at `at` takes.
pub fn bytes(ptr: *anyopaque, at: u32) u64 {
    const t: *Target = @ptrCast(@alignCast(ptr));
    return total(t.e.g, at);
}

/// Copy the stream's state after `at` prompt tokens; its prompt pass stands there.
pub fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    const t: *Target = @ptrCast(@alignCast(ptr));
    const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
    if (!fits(t.e.g)) return error.UnsupportedQwenCache;
    const cache = t.caches.get(s) orelse return error.UnknownStream;
    if (cache.len != at or cache.last != null) return error.NotAtMark;
    const e = t.e;
    const copy = try e.gpa.create(Copy);
    errdefer e.gpa.destroy(copy);
    copy.* = .{ .host = try cuda.HostBuffer.alloc(e.d, total(e.g, at)), .at = at, .delta = cache.delta };
    errdefer copy.host.free();
    var buf: [max_pieces]Piece = undefined;
    var off: usize = 0;
    for (pieces(e.g, cache, at, &buf)) |p| {
        if (p.len > 0) try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(copy.host.bytes.ptr + off, p.dev, p.len, e.stream.handle), "cuMemcpyDtoHAsync");
        off += p.len;
    }
    try e.stream.synchronize();
    return copy;
}

/// Make `cache` the kept state's: its next prompt chunk starts at the copy's position.
pub fn restoreCache(e: *Engine, cache: *Cache, saved: *anyopaque) !void {
    const copy: *Copy = @ptrCast(@alignCast(saved));
    if (copy.at >= cache.cap) return error.BadReuse;
    var buf: [max_pieces]Piece = undefined;
    var off: usize = 0;
    for (pieces(e.g, cache, copy.at, &buf)) |p| {
        if (p.len > 0) try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(p.dev, copy.host.bytes.ptr + off, p.len, e.stream.handle), "cuMemcpyHtoDAsync");
        off += p.len;
    }
    try e.stream.synchronize(); // the host copy may be dropped once this returns
    cache.len = copy.at;
    cache.written = copy.at;
    cache.last = null;
    cache.delta = copy.delta;
}

/// The store's restore: the owner's cache made the kept state's.
pub fn restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    const t: *Target = @ptrCast(@alignCast(ptr));
    const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
    return restoreCache(t.e, t.caches.get(s) orelse return error.UnknownStream, saved);
}

/// Free a kept state's host copy.
pub fn drop(ptr: *anyopaque, saved: *anyopaque) void {
    const t: *Target = @ptrCast(@alignCast(ptr));
    const copy: *Copy = @ptrCast(@alignCast(saved));
    copy.host.free();
    t.e.gpa.destroy(copy);
}

fn testGeometry() Geometry {
    var g: Geometry = undefined;
    g.hidden = 5120;
    g.k_heads = 16;
    g.v_heads = 48;
    g.q_heads = 24;
    g.kv_heads = 4;
    g.layers = 8;
    @memset(&g.linear, false);
    for (0..8) |i| g.linear[i] = i % 4 != 3; // the 27B's pattern: three linear layers, then attention
    return g;
}

test "a kept state holds the recurrences and tails whole and each attention head's rows it read" {
    const g = testGeometry();
    var buf: [max_pieces]Piece = undefined;
    try std.testing.expectEqual(@as(usize, 1 + 2 * 2 * 4), pieces(g, null, 7, &buf).len);
    try std.testing.expectEqual(@as(u64, g.stateBytes()), total(g, 0));
    try std.testing.expectEqual(@as(u64, g.stateBytes() + 2 * 2 * 4 * 1000 * head_dim * 2), total(g, 1000));
}

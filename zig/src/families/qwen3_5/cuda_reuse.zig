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

/// A cache's attention rows: bf16, or e4m3 with an fp32 scale each (--kv-cache fp8, where the engine has it).
pub const Format = enum(u8) { bf16, fp8 };

/// The engine's cache format; an engine without fp8 caches keeps bf16 rows.
pub fn formatOf(e: *const Engine) Format {
    if (comptime @hasField(Engine, "kv8")) return if (e.kv8) .fp8 else .bf16;
    return .bf16;
}

/// An image a kept state's prompt holds: the row its placeholders start at, its digest (lanes.Image.digest).
pub const Seen = struct { at: u32, digest: [32]u8 };

/// One kept state: its bytes on the host, the position it stands at, its rotary shift, its cache format and images.
pub const Copy = struct {
    host: cuda.HostBuffer,
    len: u64,
    at: usize,
    delta: i32,
    format: Format,
    images: []Seen,
};

/// The engine, the streams' caches a save copies, and their prompts' images (cuda_native.zig's Lanes owns the maps).
pub const Target = struct {
    e: *Engine,
    caches: *std.AutoHashMapUnmanaged(*lanes.Stream, *Cache),
    seen: *std.AutoHashMapUnmanaged(*lanes.Stream, []Seen),
};

/// One piece of a stream's state: its offset in the state or key/value buffer, its length.
const Piece = struct { kv: bool, off: usize, len: usize };

/// A state's pieces by index, any geometry: the state, then each (layer, key/value, head)'s first `at` rows.
const Layout = struct {
    g: Geometry,
    format: Format,
    cap: usize,
    at: usize,

    fn parts(l: Layout) usize {
        return if (l.format == .fp8) 2 else 1;
    }

    fn count(l: Layout) usize {
        return 1 + l.g.attentions() * 2 * l.g.kv_heads * l.parts();
    }

    /// Piece `i`: qwen35.cu's layout (a layer's keys then values, kv_heads blocks of cap rows), or qwen35_kv8.cu's.
    fn piece(l: Layout, i: usize) Piece {
        if (i == 0) return .{ .kv = false, .off = 0, .len = l.g.stateBytes() };
        const j = (i - 1) / l.parts();
        const kvh = l.g.kv_heads;
        const block = j % (2 * kvh); // key heads, then value heads
        const layer = j / (2 * kvh);
        return switch (l.format) {
            .bf16 => .{ .kv = true, .off = (layer * 2 * kvh + block) * l.cap * head_dim * 2, .len = l.at * head_dim * 2 },
            .fp8 => blk: {
                const base = layer * 2 * kvh * l.cap * (head_dim + 4);
                break :blk if ((i - 1) % 2 == 0)
                    .{ .kv = true, .off = base + block * l.cap * head_dim, .len = l.at * head_dim }
                else
                    .{ .kv = true, .off = base + 2 * kvh * l.cap * head_dim + block * l.cap * 4, .len = l.at * 4 };
            },
        };
    }

    /// The bytes a save at `at` takes.
    fn total(l: Layout) u64 {
        var n: u64 = 0;
        for (0..l.count()) |i| n += l.piece(i).len;
        return n;
    }

    /// The key/value buffer's bytes at `cap` (Geometry.kvBytes).
    fn kvBytes(l: Layout) usize {
        return l.g.attentions() * 2 * l.g.kv_heads * l.cap * @as(usize, if (l.format == .fp8) head_dim + 4 else head_dim * 2);
    }
};

/// Whether a request's images match a kept state's: the same digests at the same rows, for every image before `at`.
pub fn sameImages(kept: []const Seen, request: []const Seen, at: usize) bool {
    var n: usize = 0;
    for (request) |r| {
        if (r.at >= at) continue;
        if (n >= kept.len or kept[n].at != r.at or !std.mem.eql(u8, &kept[n].digest, &r.digest)) return false;
        n += 1;
    }
    return n == kept.len;
}

/// Host bytes a save at `at` takes.
pub fn bytes(ptr: *anyopaque, at: u32) u64 {
    const t: *Target = @ptrCast(@alignCast(ptr));
    return (Layout{ .g = t.e.g, .format = formatOf(t.e), .cap = 0, .at = at }).total();
}

/// Copy the stream's state after `at` prompt tokens; its prompt pass stands there.
pub fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    const t: *Target = @ptrCast(@alignCast(ptr));
    const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
    const cache = t.caches.get(s) orelse return error.UnknownStream;
    return @ptrCast(try saveCache(t.e, cache, at, t.seen.get(s) orelse &.{}));
}

/// A host copy of `cache` at `at` rows, naming the prompt's images that start before it.
pub fn saveCache(e: *Engine, cache: *const Cache, at: usize, seen: []const Seen) !*Copy {
    if (cache.len != at or cache.last != null or at > cache.cap) return error.NotAtMark;
    const l: Layout = .{ .g = e.g, .format = formatOf(e), .cap = cache.cap, .at = at };
    var n: usize = 0;
    for (seen) |im| n += @intFromBool(im.at < at);
    const copy = try e.gpa.create(Copy);
    errdefer e.gpa.destroy(copy);
    const images = try e.gpa.alloc(Seen, n);
    errdefer e.gpa.free(images);
    n = 0;
    for (seen) |im| if (im.at < at) {
        images[n] = im;
        n += 1;
    };
    copy.* = .{ .host = try cuda.HostBuffer.alloc(e.d, l.total()), .len = l.total(), .at = at, .delta = cache.delta, .format = l.format, .images = images };
    errdefer copy.host.free();
    var off: usize = 0;
    for (0..l.count()) |i| {
        const p = l.piece(i);
        const dev = (if (p.kv) cache.kv.ptr else cache.state.ptr) + p.off;
        if (p.len > 0) try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(copy.host.bytes.ptr + off, dev, p.len, e.stream.handle), "cuMemcpyDtoHAsync");
        off += p.len;
    }
    try e.stream.synchronize();
    return copy;
}

/// Whether `cache` can take the kept state: the same cache format and bytes, a position inside it, the same images.
pub fn restorable(e: *const Engine, cache: *const Cache, copy: *const Copy, seen: []const Seen) bool {
    if (copy.format != formatOf(e) or copy.at >= cache.cap) return false;
    const l: Layout = .{ .g = e.g, .format = copy.format, .cap = cache.cap, .at = copy.at };
    return copy.len == l.total() and copy.host.bytes.len >= copy.len and sameImages(copy.images, seen, copy.at);
}

/// Make `cache` the kept state's: its next prompt chunk starts at the copy's position; refused before any copy.
pub fn restoreCache(e: *Engine, cache: *Cache, saved: *anyopaque, seen: []const Seen) !void {
    const copy: *Copy = @ptrCast(@alignCast(saved));
    if (copy.format != formatOf(e)) return error.KeptCacheFormat;
    if (!restorable(e, cache, copy, seen)) return error.BadReuse;
    const l: Layout = .{ .g = e.g, .format = copy.format, .cap = cache.cap, .at = copy.at };
    var off: usize = 0;
    for (0..l.count()) |i| {
        const p = l.piece(i);
        const dev = (if (p.kv) cache.kv.ptr else cache.state.ptr) + p.off;
        if (p.len > 0) try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(dev, copy.host.bytes.ptr + off, p.len, e.stream.handle), "cuMemcpyHtoDAsync");
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
    return restoreCache(t.e, t.caches.get(s) orelse return error.UnknownStream, saved, t.seen.get(s) orelse &.{});
}

/// Free a kept state's host copy.
pub fn drop(ptr: *anyopaque, saved: *anyopaque) void {
    const t: *Target = @ptrCast(@alignCast(ptr));
    dropCopy(t.e, @ptrCast(@alignCast(saved)));
}

pub fn dropCopy(e: *Engine, copy: *Copy) void {
    copy.host.free();
    e.gpa.free(copy.images);
    e.gpa.destroy(copy);
}

fn testGeometry(layers: usize, attention_every: usize, kv_heads: usize) Geometry {
    var g: Geometry = undefined;
    g.hidden = 5120;
    g.k_heads = 16;
    g.v_heads = 48;
    g.q_heads = 32;
    g.kv_heads = kv_heads;
    g.layers = layers;
    @memset(&g.linear, false);
    for (0..layers) |i| g.linear[i] = i % attention_every != attention_every - 1;
    return g;
}

test "a kept state holds the recurrences and tails whole and each attention head's rows it read" {
    const g = testGeometry(8, 4, 4); // the 27B's pattern: three linear layers, then attention
    const l: Layout = .{ .g = g, .format = .bf16, .cap = 2048, .at = 1000 };
    try std.testing.expectEqual(@as(usize, 1 + 2 * 2 * 4), l.count());
    try std.testing.expectEqual(@as(u64, g.stateBytes()), (Layout{ .g = g, .format = .bf16, .cap = 2048, .at = 0 }).total());
    try std.testing.expectEqual(@as(u64, g.stateBytes() + 2 * 2 * 4 * 1000 * head_dim * 2), l.total());
    // fp8: e4m3 rows and a scale each, 260 bytes a position and head
    const l8: Layout = .{ .g = g, .format = .fp8, .cap = 2048, .at = 1000 };
    try std.testing.expectEqual(@as(usize, 1 + 2 * 2 * 2 * 4), l8.count());
    try std.testing.expectEqual(@as(u64, g.stateBytes() + 2 * 2 * 4 * 1000 * (head_dim + 4)), l8.total());
}

test "every geometry's pieces lie inside its allocation, apart, with no fixed limit on layers or heads" {
    const a = std.testing.allocator;
    // 33 attention layers of 32 key heads passed the old fixed array; the widest the engine reads is 128 layers
    for ([_][3]usize{ .{ 8, 4, 4 }, .{ 99, 3, 32 }, .{ 33, 1, 32 }, .{ 128, 1, 32 }, .{ 24, 4, 2 } }) |shape| {
        const g = testGeometry(shape[0], shape[1], shape[2]);
        for ([_]Format{ .bf16, .fp8 }) |f| for ([_]usize{ 1, 7, 64 }) |at| {
            const l: Layout = .{ .g = g, .format = f, .cap = 64, .at = at };
            const end = l.kvBytes();
            if (comptime @typeInfo(@TypeOf(Geometry.kvBytes)).@"fn".param_types.len == 3) try std.testing.expectEqual(g.kvBytes(64, f == .fp8), end);
            const spans = try a.alloc([2]usize, l.count() - 1);
            defer a.free(spans);
            try std.testing.expectEqual(@as(usize, g.stateBytes()), l.piece(0).len);
            for (spans, 1..) |*sp, i| {
                const p = l.piece(i);
                try std.testing.expect(p.kv and p.len > 0 and p.off + p.len <= end);
                sp.* = .{ p.off, p.off + p.len };
            }
            std.mem.sort([2]usize, spans, {}, struct {
                fn lt(_: void, x: [2]usize, y: [2]usize) bool {
                    return x[0] < y[0];
                }
            }.lt);
            for (spans[1..], spans[0 .. spans.len - 1]) |sp, prev| try std.testing.expect(sp[0] >= prev[1]);
        };
    }
}

test "fp8 pieces follow qwen35_kv8.cu: a layer's key then value rows, then their scales" {
    const g = testGeometry(4, 4, 2); // one attention layer
    const l: Layout = .{ .g = g, .format = .fp8, .cap = 100, .at = 10 };
    // (key, head 1): rows at 1 * cap * 256, its scales after both blocks' rows, at 2 * 2 * cap * 256 + 1 * cap * 4
    try std.testing.expectEqual(Piece{ .kv = true, .off = 100 * 256, .len = 10 * 256 }, l.piece(1 + 2 * 1));
    try std.testing.expectEqual(Piece{ .kv = true, .off = 4 * 100 * 256 + 100 * 4, .len = 10 * 4 }, l.piece(2 + 2 * 1));
    // (value, head 0): the third block's rows and scales
    try std.testing.expectEqual(Piece{ .kv = true, .off = 2 * 100 * 256, .len = 10 * 256 }, l.piece(1 + 2 * 2));
    try std.testing.expectEqual(Piece{ .kv = true, .off = 4 * 100 * 256 + 2 * 100 * 4, .len = 10 * 4 }, l.piece(2 + 2 * 2));
}

test "a state resumes only a request with the same images before its position" {
    const d = [_][32]u8{ @splat(1), @splat(2), @splat(3) };
    const kept = [_]Seen{.{ .at = 5, .digest = d[0] }};
    try std.testing.expect(sameImages(&kept, &.{ .{ .at = 5, .digest = d[0] }, .{ .at = 40, .digest = d[2] } }, 20)); // a later image is the pass's
    try std.testing.expect(!sameImages(&kept, &.{.{ .at = 5, .digest = d[1] }}, 20)); // another image under the same ids
    try std.testing.expect(!sameImages(&kept, &.{.{ .at = 6, .digest = d[0] }}, 20));
    try std.testing.expect(!sameImages(&kept, &.{}, 20)); // a text prompt never resumes an image's state
    try std.testing.expect(!sameImages(&.{}, &.{.{ .at = 5, .digest = d[0] }}, 20)); // nor an image prompt a text state past its image
    try std.testing.expect(sameImages(&.{}, &.{.{ .at = 5, .digest = d[0] }}, 5)); // a text prefix before it
}

test "a restore refuses another cache format, another length, a position past the cache and other images" {
    var e: Engine = undefined;
    e.g = testGeometry(8, 4, 4);
    if (comptime @hasField(Engine, "kv8")) e.kv8 = false;
    var cache: Cache = undefined;
    cache.cap = 64;
    const d: [32]u8 = @splat(9);
    var images = [_]Seen{.{ .at = 2, .digest = d }};
    const l: Layout = .{ .g = e.g, .format = .bf16, .cap = 64, .at = 10 };
    const host = try std.testing.allocator.alignedAlloc(u8, .@"16", @intCast(l.total()));
    defer std.testing.allocator.free(host);
    var copy: Copy = .{ .host = .{ .d = undefined, .bytes = host }, .len = l.total(), .at = 10, .delta = 0, .format = .bf16, .images = &images };
    const same = [_]Seen{.{ .at = 2, .digest = d }};
    try std.testing.expect(restorable(&e, &cache, &copy, &same));
    try std.testing.expect(!restorable(&e, &cache, &copy, &.{.{ .at = 2, .digest = @splat(8) }}));
    copy.format = .fp8; // a state kept from fp8 rows never lands in bf16 ones, nor the other way
    try std.testing.expect(!restorable(&e, &cache, &copy, &same));
    copy.format = .bf16;
    copy.len -= 1;
    try std.testing.expect(!restorable(&e, &cache, &copy, &same));
    copy.len += 1;
    cache.cap = 10;
    try std.testing.expect(!restorable(&e, &cache, &copy, &same));
}

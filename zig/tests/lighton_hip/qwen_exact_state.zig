//! Exact logical Qwen state and owned snapshots: all GDN bytes and committed KV rows, independent of page ids.
const std = @import("std");
const hip = @import("hip");
const q = @import("qwen3_5");
const Engine = q.engine.Engine;
const Allocator = std.mem.Allocator;

pub const Lane = struct {
    caches: q.state.Caches,
    hidden: hip.DeviceBuffer,
    len: usize = 0,

    pub fn init(e: *Engine) !Lane {
        var caches = try e.newCaches(e.o.capacity);
        errdefer caches.deinit(e.gpa);
        var hidden = try hip.DeviceBuffer.alloc(&e.driver, e.model().spec.hidden * e.model().act.size());
        errdefer hidden.free();
        try hidden.fill8(0);
        try caches.setHidden(hidden.base());
        return .{ .caches = caches, .hidden = hidden };
    }

    pub fn deinit(l: *Lane, e: *Engine) void {
        e.drain();
        l.caches.deinit(e.gpa);
        l.hidden.free();
    }
};

/// Hash every conv/recurrent byte, the last hidden row, and just the committed part of every KV head.
pub fn digest(a: Allocator, e: *Engine, lane: *const Lane) ![32]u8 {
    try e.stream.synchronize();
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(std.mem.asBytes(&lane.len));
    for (lane.caches.layers, 0..) |layer, index| switch (layer) {
        .linear => |x| for ([_]hip.DeviceBuffer{ x.conv, x.state }) |b| {
            const bytes = try a.alloc(u8, b.len);
            defer a.free(bytes);
            try b.download(0, bytes);
            try finite(bytes, .f32, true);
            hash.update(bytes);
        },
        .full => {
            const pool = &e.pool;
            const page = try a.alloc(u8, 2 * pool.layerBytes());
            defer a.free(page);
            const head_bytes = q.pages.tokens * pool.head_dim * pool.act_bytes;
            for (lane.caches.table.items[0..q.pages.pagesFor(lane.len)], 0..) |id, p| {
                try pool.read(index, id, page);
                const rows = @min(q.pages.tokens, lane.len - p * q.pages.tokens);
                for (0..2 * pool.kv_heads) |h| {
                    const committed = page[h * head_bytes ..][0 .. rows * pool.head_dim * pool.act_bytes];
                    try finite(committed, .bf16, true);
                    hash.update(committed);
                }
            }
        },
    };
    const hidden = try a.alloc(u8, lane.hidden.len);
    defer a.free(hidden);
    try lane.hidden.download(0, hidden);
    try finite(hidden, .bf16, false);
    hash.update(hidden);
    return hash.finalResult();
}

/// A host KV copy plus the production linear save/load buffer; it owns no reference to a source lane's pages.
pub const Snapshot = struct {
    a: Allocator,
    linear: hip.DeviceBuffer,
    hidden: []u8,
    kv: [][]u8,
    len: usize,

    pub fn save(a: Allocator, e: *Engine, lane: *const Lane) !Snapshot {
        try e.stream.synchronize();
        var linear = try hip.DeviceBuffer.alloc(&e.driver, lane.caches.linearBytes());
        errdefer linear.free();
        try lane.caches.saveLinear(linear, e.stream.handle);
        try e.stream.synchronize();
        const hidden = try a.alloc(u8, lane.hidden.len);
        errdefer a.free(hidden);
        try lane.hidden.download(0, hidden);
        const kv = try a.alloc([]u8, lane.caches.layers.len);
        errdefer a.free(kv);
        @memset(kv, &.{});
        errdefer for (kv) |bytes| a.free(bytes);
        const page_bytes = 2 * e.pool.layerBytes();
        const pages = q.pages.pagesFor(lane.len);
        for (kv, 0..) |*bytes, index| if (e.model().spec.full(index)) {
            bytes.* = try a.alloc(u8, pages * page_bytes);
            for (lane.caches.table.items[0..pages], 0..) |id, p| try e.pool.read(index, id, bytes.*[p * page_bytes ..][0..page_bytes]);
        };
        return .{ .a = a, .linear = linear, .hidden = hidden, .kv = kv, .len = lane.len };
    }

    pub fn restore(s: *const Snapshot, e: *Engine, lane: *Lane) !void {
        try e.stream.synchronize();
        const pool = &e.pool;
        const n = q.pages.tokens * pool.head_dim * pool.act_bytes;
        const pages = q.pages.pagesFor(s.len);
        const page_bytes = 2 * pool.layerBytes();
        for (s.kv, 0..) |bytes, index| {
            if (bytes.len == 0) continue;
            for (lane.caches.table.items[0..pages], 0..) |id, p| for (0..pool.kv_heads) |h| {
                const dst = (h * pool.count + id) * n;
                const src = p * page_bytes;
                try pool.keys[index].upload(dst, bytes[src + h * n ..][0..n]);
                try pool.values[index].upload(dst, bytes[src + (pool.kv_heads + h) * n ..][0..n]);
            };
        }
        try lane.hidden.upload(0, s.hidden);
        try lane.caches.loadLinear(s.linear, e.stream.handle);
        try e.stream.synchronize();
        lane.len = s.len;
    }

    pub fn deinit(s: *Snapshot, e: *Engine) void {
        e.drain();
        s.linear.free();
        s.a.free(s.hidden);
        for (s.kv) |bytes| s.a.free(bytes);
        s.a.free(s.kv);
    }
};

/// Positive checks prevent equal NaNs, infinities or untouched zero buffers from passing exact comparisons.
pub fn finite(bytes: []const u8, kind: hip.ops.Kind, require_nonzero: bool) !void {
    var nonzero = false;
    const size = kind.size();
    if (bytes.len == 0 or bytes.len % size != 0) return error.BadStateBytes;
    var at: usize = 0;
    while (at < bytes.len) : (at += size) {
        const value: f32 = switch (kind) {
            .f32 => @bitCast(std.mem.readInt(u32, bytes[at..][0..4], .little)),
            .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, bytes[at..][0..2], .little)) << 16),
            else => return error.UnsupportedStateKind,
        };
        if (!std.math.isFinite(value)) return error.NonfiniteState;
        nonzero = nonzero or value != 0;
    }
    if (require_nonzero and !nonzero) return error.UntouchedZeroState;
}

test "exact state positive checks reject identical NaNs and untouched zero buffers" {
    const zero = [_]u8{ 0, 0, 0, 0 };
    try std.testing.expectError(error.UntouchedZeroState, finite(&zero, .f32, true));
    const nan = [_]u8{ 0, 0, 0xc0, 0x7f };
    try std.testing.expectError(error.NonfiniteState, finite(&nan, .f32, false));
    const bf16 = [_]u8{ 0x80, 0x3f };
    try finite(&bf16, .bf16, true);
}

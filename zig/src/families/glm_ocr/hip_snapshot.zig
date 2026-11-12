//! Immutable prompt copies include one keyed prepared row and the full prompt arithmetic shape.
const std = @import("std");
const hip = @import("hip");
const c = @import("config.zig");
const Cache = @import("hip_text.zig").Cache;

pub const Snapshot = struct {
    cache: Cache,
    hidden: hip.DeviceBuffer,
    at: u32,
    math_rows: usize,

    pub fn bytes(at: u32) u64 {
        return (@as(u64, at) + 1) * c.layers * c.kv * 4 + c.hidden * 2;
    }
    pub fn take(source: *const Cache, normed: u64, at: u32, math_rows: usize, budget: usize) !Snapshot {
        if (at >= math_rows or source.len != math_rows or normed == 0) return error.BadPromptSnapshot;
        if (bytes(at) > budget) return error.GlmOcrSnapshotDoesNotFit;
        var cache = try Cache.init(source.stream, @as(usize, at) + 1, budget - c.hidden * 2);
        errdefer cache.deinit();
        try cache.copyPrefix(source, @as(usize, at) + 1);
        var hidden = try hip.DeviceBuffer.alloc(source.stream.r, c.hidden * 2);
        errdefer hidden.free();
        errdefer source.stream.synchronize() catch {};
        try hidden.copyFromAsync(0, normed + @as(u64, at) * c.hidden * 2, c.hidden * 2, source.stream);
        try source.stream.synchronize();
        return .{ .cache = cache, .hidden = hidden, .at = at, .math_rows = math_rows };
    }
    pub fn restore(s: *const Snapshot, dest: *Cache, at: u32, math_rows: usize) !void {
        if (s.at != at or s.math_rows != math_rows or at >= math_rows or s.cache.len != @as(usize, at) + 1)
            return error.GlmOcrPromptShapeChanged;
        try dest.copyPrefix(&s.cache, @as(usize, at) + 1);
    }
    pub fn deinit(s: *Snapshot) void {
        s.cache.deinit();
        s.hidden.free();
        s.* = undefined;
    }
};

test "GLM-OCR HIP snapshot: incompatible arithmetic and budgets fail before copying" {
    var source: Cache = undefined;
    source.len = 20;
    try std.testing.expectError(error.BadPromptSnapshot, Snapshot.take(&source, 2, 20, 20, 0));
    try std.testing.expectError(error.GlmOcrSnapshotDoesNotFit, Snapshot.take(&source, 2, 19, 20, 0));
    const s: Snapshot = .{ .cache = source, .hidden = undefined, .at = 10, .math_rows = 20 };
    try std.testing.expectError(error.GlmOcrPromptShapeChanged, s.restore(&source, 10, 21));
    try std.testing.expectError(error.GlmOcrPromptShapeChanged, s.restore(&source, 9, 20));
    try std.testing.expectEqual(@as(u64, 11 * c.layers * c.kv * 4 + c.hidden * 2), Snapshot.bytes(10));
}

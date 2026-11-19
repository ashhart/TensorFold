//! Translate lane-local token positions to physical latent KV slots for Metal and CUDA.
const std = @import("std");

pub const Cache = struct {
    /// Offset of each logical page's first row in the layer's contiguous latent arena.
    page_bases: []const u32,
    page_tokens: u32,
    capacity: u32,

    pub fn slot(self: Cache, position: u32) !u32 {
        if (self.page_tokens == 0) return error.BadHy4CachePage;
        const page = position / self.page_tokens;
        if (page >= self.page_bases.len) return error.BadHy4CachePage;
        const row = @as(u64, self.page_bases[page]) + position % self.page_tokens;
        if (row >= self.capacity or row > std.math.maxInt(i32)) return error.BadHy4CacheSlot;
        return @intCast(row);
    }

    /// Includes the query row itself. Length is the final per-row attention count.
    pub fn sliding(self: Cache, visible_end: u32, width: u32, output: []u32) !u32 {
        const count = @min(visible_end, width);
        if (count > output.len) return error.ShortHy4KeyList;
        const first = visible_end - count;
        for (output[0..count], 0..) |*dst, i| dst.* = try self.slot(first + @as(u32, @intCast(i)));
        return count;
    }

    /// Compact the sparse selector's positions (also used for DFly2 reuse) in their original order.
    pub fn retained(self: Cache, visible_end: u32, positions: []const i32, output: []u32) !u32 {
        var count: u32 = 0;
        for (positions) |position| {
            if (position == -1) continue;
            if (position < 0 or position >= visible_end) return error.NonCausalHy4Key;
            if (count == output.len) return error.ShortHy4KeyList;
            output[count] = try self.slot(@intCast(position));
            count += 1;
        }
        return count;
    }
};

test "a 512-token window crosses a physically reordered cache page" {
    const cache: Cache = .{ .page_bases = &.{ 512, 0 }, .page_tokens = 512, .capacity = 1024 };
    var keys: [513]u32 = @splat(0xffffffff);
    for ([_]u32{ 511, 512, 513 }) |end| {
        const count = try cache.sliding(end, 512, &keys);
        try std.testing.expectEqual(@min(end, 512), count);
        try std.testing.expectEqual(@as(u32, if (end > 512) 513 else 512), keys[0]);
        try std.testing.expectEqual(@as(u32, if (end > 512) 0 else 512 + end - 1), keys[count - 1]);
        try std.testing.expectEqual(@as(u32, 0xffffffff), keys[count]);
    }
}

test "draft attention compacts masked target keys and future draft rows" {
    const dfly2 = @import("dfly2_inputs.zig");
    const cache: Cache = .{ .page_bases = &.{ 8, 12, 0, 4, 16 }, .page_tokens = 4, .capacity = 20 };
    const frame: dfly2.Frame = .{ .source_row = 2, .anchor = 13, .query_rows = 6 };
    var positions: [16]i32 = undefined;
    try dfly2.reuseIndices(frame, &.{ 0, 8, 10, 12, 13, -1 }, 3, 2, &positions);
    var keys: [16]u32 = @splat(0xffffffff);
    const count = try cache.retained(frame.anchor + 3, &positions, &keys);
    try std.testing.expectEqualSlices(u32, &.{ 8, 0, 2, 3, 4, 5, 6, 7 }, keys[0..count]);
    try std.testing.expectEqual(@as(u32, 0xffffffff), keys[count]);
    try std.testing.expectError(error.NonCausalHy4Key, cache.retained(frame.anchor + 2, &positions, &keys));
    try std.testing.expectError(error.BadHy4CachePage, cache.slot(20));
}

//! The launch shape for absorbed attention over retained physical KV slots.
const std = @import("std");
const weights = @import("mla_weights.zig");

pub const chunk_keys = 512;
pub const score_scale: f32 = 0.0625; // 1 / sqrt(192 + 64), before absorbing the key projection.

pub const CacheWrite = extern struct {
    rows: u32,
    capacity: u32,
    epsilon: f32,

    pub fn check(self: CacheWrite) !void {
        if (self.rows == 0 or self.rows > std.math.maxInt(u16) or self.capacity == 0 or
            self.capacity > std.math.maxInt(i32) or !std.math.isFinite(self.epsilon) or self.epsilon <= 0)
            return error.BadHy4CacheWrite;
    }
};

pub const Shape = extern struct {
    rows: u32,
    heads: u32,
    key_stride: u32,
    chunks: u32,

    pub fn init(rows: u32, heads: u32, key_stride: u32) !Shape {
        if (rows == 0 or heads == 0 or key_stride == 0) return error.EmptyHy4AttentionShape;
        const chunks = (key_stride - 1) / chunk_keys + 1;
        if (chunks > std.math.maxInt(u16) or rows > std.math.maxInt(u16) or heads > 64)
            return error.UnsupportedHy4AttentionShape;
        return .{ .rows = rows, .heads = heads, .key_stride = key_stride, .chunks = chunks };
    }

    pub fn partialHeads(self: Shape) usize {
        return @as(usize, self.rows) * self.heads * self.chunks;
    }

    pub fn scratchBytes(self: Shape) usize {
        return self.partialHeads() * (weights.latent + 2) * @sizeOf(f32);
    }

    pub fn check(self: Shape) !void {
        const derived = try init(self.rows, self.heads, self.key_stride);
        if (self.chunks != derived.chunks) return error.WrongHy4AttentionChunks;
    }
};

test "dispatch geometry covers sliding and sparse tails" {
    for ([_]u32{ 511, 512, 513, 2047, 2048, 2049 }, [_]u32{ 1, 1, 2, 4, 4, 5 }) |len, chunks| {
        const shape = try Shape.init(8, 32, len);
        try shape.check();
        try std.testing.expectEqual(chunks, shape.chunks);
        try std.testing.expectEqual(@as(usize, 8 * 32 * (512 + 2) * 4) * chunks, shape.scratchBytes());
    }
    try std.testing.expectError(error.EmptyHy4AttentionShape, Shape.init(1, 64, 0));
    try std.testing.expectError(error.UnsupportedHy4AttentionShape, Shape.init(1, 65, 512));
}

//! Native operator ABI and row-local dispatch geometry, without a Metal device dependency.
const std = @import("std");
const DType = @import("checkpoint.zig").DType;

pub const Float = enum(u32) { bf16 = 0, f16 = 1, f32 = 2 };

pub fn floatType(dtype: DType) !Float {
    return switch (dtype) {
        .bf16 => .bf16,
        .f16 => .f16,
        .f32 => .f32,
        else => error.UnsupportedFloat,
    };
}

pub const Norm = extern struct { width: u32, rows: u32, activation: Float, gain: Float, residual: u32, eps: f32 };
pub const Head = extern struct { width: u32, rows: u32, stride: u32, activation: Float, gain: Float, eps: f32 };
pub const Embed = extern struct { width: u32, rows: u32, vocab: u32, bits: u32, group: u32, metadata: Float };
pub const Rope = extern struct { width: u32, heads: u32, rows: u32, rotary: u32, activation: Float, theta: f32 };
pub const Element = extern struct { width: u32, heads: u32, rows: u32, activation: Float };
pub const Post = extern struct { width: u32, heads: u32, rows: u32, stride: u32, offset: u32, activation: Float, gain: Float, eps: f32 };

pub fn normThreads(a: Norm) !u32 {
    if (a.width == 0 or a.width > 16384 or a.width % 512 != 0 or a.rows == 0 or a.residual > 1 or !std.math.isFinite(a.eps) or a.eps <= 0) return error.BadNorm;
    return a.width / 16;
}

pub fn headThreads(a: Head) !u32 {
    if (a.width == 0 or a.width > 16384 or a.width % 32 != 0 or a.stride < a.width or a.rows == 0 or a.eps <= 0) return error.BadHead;
    return @min(1024, @max(32, (a.width + 127) / 128 * 32));
}

pub fn checkEmbed(a: Embed) !void {
    try @import("affine.zig").Spec.check(.{ .bits = std.math.cast(u8, a.bits) orelse return error.BadAffine, .group_size = a.group });
    if (a.width == 0 or a.rows == 0 or a.vocab == 0 or a.width % a.group != 0 or @as(u64, a.width) * a.bits % 32 != 0) return error.BadEmbedding;
}

test "operator parameters match the six-word Metal ABI" {
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(Norm));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(Head));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(Embed));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(Rope));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(Element));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(Post));
    try std.testing.expectEqual(@as(usize, 20), @offsetOf(Norm, "eps"));
}

test "one and many rows use identical norm reduction geometry" {
    var a = Norm{ .width = 5120, .rows = 1, .activation = .bf16, .gain = .f16, .residual = 1, .eps = 1e-6 };
    const one = try normThreads(a);
    a.rows = 128;
    try std.testing.expectEqual(one, try normThreads(a));
    try std.testing.expectEqual(@as(u32, 320), one);
    a.width = 5119;
    try std.testing.expectError(error.BadNorm, normThreads(a));
}

test "native types and packed shapes are explicit" {
    try std.testing.expectEqual(Float.f16, try floatType(.f16));
    try std.testing.expectError(error.UnsupportedFloat, floatType(.u32));
    try checkEmbed(.{ .width = 5120, .rows = 8, .vocab = 4, .bits = 3, .group = 64, .metadata = .f16 });
    try std.testing.expectError(error.BadEmbedding, checkEmbed(.{ .width = 63, .rows = 1, .vocab = 4, .bits = 4, .group = 64, .metadata = .bf16 }));
}

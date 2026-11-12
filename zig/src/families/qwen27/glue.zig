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

pub const Head = extern struct { width: u32, rows: u32, stride: u32, activation: Float, gain: Float, eps: f32 };
pub const Embed = extern struct { width: u32, rows: u32, vocab: u32, bits: u32, group: u32, metadata: Float };
/// h_end, w_end: M-RoPE's interleaved h and w pairs end there (0: every pair turns by plane 0); planes `plane` apart.
pub const Rope = extern struct { width: u32, heads: u32, rows: u32, rotary: u32, activation: Float, theta: f32, h_end: u32 = 0, w_end: u32 = 0, plane: u32 = 0 };
pub const Element = extern struct { width: u32, heads: u32, rows: u32, activation: Float };

pub fn headThreads(a: Head) !u32 {
    if (a.width == 0 or a.width > 16384 or a.width % 32 != 0 or a.stride < a.width or a.rows == 0 or a.eps <= 0) return error.BadHead;
    return @min(1024, @max(32, (a.width + 127) / 128 * 32));
}

pub fn checkEmbed(a: Embed) !void {
    try @import("affine.zig").Spec.check(.{ .bits = std.math.cast(u8, a.bits) orelse return error.BadAffine, .group_size = a.group });
    if (a.width == 0 or a.rows == 0 or a.vocab == 0 or a.width % a.group != 0 or @as(u64, a.width) * a.bits % 32 != 0) return error.BadEmbedding;
}

test "operator parameters match their Metal structs word for word" {
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(Head));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(Embed));
    try std.testing.expectEqual(@as(usize, 36), @sizeOf(Rope));
    try std.testing.expectEqual(@as(usize, 24), @offsetOf(Rope, "h_end"));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(Element));
}

test "native types and packed shapes are explicit" {
    try std.testing.expectEqual(Float.f16, try floatType(.f16));
    try std.testing.expectError(error.UnsupportedFloat, floatType(.u32));
    try checkEmbed(.{ .width = 5120, .rows = 8, .vocab = 4, .bits = 3, .group = 64, .metadata = .f16 });
    try std.testing.expectError(error.BadEmbedding, checkEmbed(.{ .width = 63, .rows = 1, .vocab = 4, .bits = 4, .group = 64, .metadata = .bf16 }));
}

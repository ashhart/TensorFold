//! The element types logits come in: the activation dtype the engine draws from, or fp32.

const std = @import("std");

pub const Kind = enum {
    bf16,
    f16,
    f32,

    pub fn size(k: Kind) usize {
        return if (k == .f32) 4 else 2;
    }

    pub fn descr(k: Kind) []const u8 {
        return switch (k) {
            .bf16 => "<u2",
            .f16 => "<f2",
            .f32 => "<f4",
        };
    }

    /// The value of the element at the start of `bytes`.
    pub fn at(k: Kind, bytes: []const u8) f64 {
        return switch (k) {
            .bf16 => @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, bytes[0..2], .little)) << 16)),
            .f16 => @as(f16, @bitCast(std.mem.readInt(u16, bytes[0..2], .little))),
            .f32 => @as(f32, @bitCast(std.mem.readInt(u32, bytes[0..4], .little))),
        };
    }
};

test "elements decode to their values" {
    try std.testing.expectEqual(@as(f64, 1.0), Kind.bf16.at(&.{ 0x80, 0x3f }));
    try std.testing.expectEqual(@as(f64, 1.0), Kind.f16.at(&.{ 0x00, 0x3c }));
    try std.testing.expectEqual(@as(f64, -2.0), Kind.f32.at(&.{ 0x00, 0x00, 0x00, 0xc0 }));
}

//! Preserve F32 A_log values and widen BF16 scalars once for the Metal backward.
const std = @import("std");
pub const Kind = enum { bf16, f32 };

pub fn values(a: std.mem.Allocator, kind: Kind, raw: []const u8) ![]f32 {
    const stride: usize = if (kind == .bf16) 2 else 4;
    if (raw.len % stride != 0) return error.BadLearningScalar;
    const out = try a.alloc(f32, raw.len / stride);
    for (out, 0..) |*v, i| {
        const bits = if (kind == .bf16) @as(u32, std.mem.readInt(u16, raw[i * 2 ..][0..2], .little)) << 16 else std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
        v.* = @bitCast(bits);
    }
    return out;
}

test "qwen27 learning: F32 decay logs keep their low bits and BF16 gains widen exactly" {
    const a = std.testing.allocator;
    const f = [_]f32{ -0.1234567, 1.234567, 0, -3.1 };
    const full = try values(a, .f32, std.mem.sliceAsBytes(&f));
    defer a.free(full);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&f), std.mem.sliceAsBytes(full));
    const b = [_]u16{ 0xbf80, 0x0000, 0x3f81, 0x8000 };
    const wide = try values(a, .bf16, std.mem.sliceAsBytes(&b));
    defer a.free(wide);
    for (b, wide) |bits, v| try std.testing.expectEqual(@as(u32, bits) << 16, @as(u32, @bitCast(v)));
    try std.testing.expectError(error.BadLearningScalar, values(a, .f32, &.{ 0, 1 }));
}

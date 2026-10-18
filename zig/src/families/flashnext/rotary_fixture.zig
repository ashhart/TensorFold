//! Exact finite normalization fixtures, independent of position and keyed by their original bytes.
const std = @import("std");
const Role = @import("rotary_domain.zig").Role;
pub const Fixture = struct { width: usize, weight: f32, input: [256]u16 = @splat(0) };

pub fn bf(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}
pub fn from(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}
pub fn kinds(role: Role) usize {
    return if (role == .pooled or role == .absolute) 1 else 3;
}
pub fn make(role: Role, kind: usize, phase: usize) Fixture {
    const pool = kinds(role) == 1;
    var result: Fixture = .{ .width = if (pool or kind == 2) 128 else 256, .weight = if (pool or kind == 2) 1.25 else if (kind == 1) 0.75 else 1 };
    for (result.input[0..result.width], 0..) |*h, d| {
        var sum: f32 = 0;
        for (0..if (pool) @as(usize, 4) else 1) |k| sum += @as(f32, @floatFromInt((phase + d + k * 128) % 17)) / 8 - 1;
        h.* = bf(sum * (if (pool) @as(f32, 0.25) else 1));
    }
    return result;
}
pub fn match(role: Role, width: usize, weight: u32, input: []const u16) ?usize {
    for (0..kinds(role)) |kind| for (0..17) |phase| {
        const fixture = make(role, kind, phase);
        if (fixture.width == width and @as(u32, @bitCast(fixture.weight)) == weight and
            std.mem.eql(u16, fixture.input[0..width], input)) return kind * 17 + phase;
    };
    return null;
}

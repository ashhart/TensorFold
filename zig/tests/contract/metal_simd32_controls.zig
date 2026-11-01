//! The 32-row projection fault couples its last two row tiles to its first two, leaving 16-row passes intact.
const std = @import("std");
pub const frozen = @embedFile("fixtures/simd32/main_q4.metal");
pub const frozen_strided = @embedFile("fixtures/simd32/main_strided_q4.metal");
pub fn source(a: std.mem.Allocator, original: []const u8) ![]u8 {
    const before = "xr0[rt] = min(rb + 8 * rt + fn, R - 1);";
    const after = "xr0[rt] = min(rb + 8 * (rt % 2) + fn, R - 1);";
    if (std.mem.indexOf(u8, original, before) == null) return error.ControlSourceChanged;
    return std.mem.replaceOwned(u8, a, original, before, after);
}

test "simd32 control preserves narrow rows and changes the third tile's input" {
    const sources = @import("kernel_sources");
    for ([_][]const u8{ sources.core_simd_qmm, sources.core_simd_qmm_bits }) |text| {
        const bad = try source(std.testing.allocator, text);
        defer std.testing.allocator.free(bad);
        try std.testing.expect(!std.mem.eql(u8, text, bad));
        try std.testing.expect(std.mem.indexOf(u8, bad, "8 * (rt % 2)") != null);
    }
    for (0..2) |tile| try std.testing.expectEqual(tile * 8, (tile % 2) * 8);
    try std.testing.expect(@as(usize, 16) != (2 % 2) * 8);
}

pub fn packRows(out: []u8, bytes: []const u8, rows: usize, n: usize, stride: usize) void {
    for (0..rows) |r| @memcpy(out[r * n * 2 ..][0 .. n * 2], bytes[r * stride * 2 ..][0 .. n * 2]);
}

pub fn paddingIntact(bytes: []const u8, rows: usize, n: usize, stride: usize) bool {
    for (0..rows) |r| for (bytes[(r * stride + n) * 2 .. (r + 1) * stride * 2]) |b| {
        if (b != 0xff) return false;
    };
    return true;
}

test "simd32 strided output compares only members and checks every padding byte" {
    var bytes: [32 * 12 * 2]u8 = @splat(0xff);
    var compact: [32 * 4 * 2]u8 = undefined;
    for (0..32) |r| @memset(bytes[r * 24 ..][0..8], @intCast(r));
    packRows(&compact, &bytes, 32, 4, 12);
    for (0..32) |r| for (compact[r * 8 ..][0..8]) |b| try std.testing.expectEqual(@as(u8, @intCast(r)), b);
    try std.testing.expect(paddingIntact(&bytes, 32, 4, 12));
    bytes[31 * 24 + 8] = 0;
    try std.testing.expect(!paddingIntact(&bytes, 32, 4, 12));
}

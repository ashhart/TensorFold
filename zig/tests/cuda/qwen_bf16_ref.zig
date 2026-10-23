//! Independent copy and FP64 dot models, with a stress row shared by the real kernel adapter.
const std = @import("std");
const core = @import("core");
const c = core.contract;

pub fn stress(x: []u16) void {
    for (x, 0..) |*v, i| v.* = c.spec.toBf16(if (i % 64 < 16) 2048 else if (i % 64 < 48) 1 else -2048);
}
pub fn copy(table: []const u16, ids: []const i32, width: usize, live: usize, out: []u16) void {
    @memset(out, 0xffff);
    for (ids[0..live], 0..) |id, r| @memcpy(out[r * width ..][0..width], table[@as(usize, @intCast(id)) * width ..][0..width]);
}
pub fn dot(x: []const u16, w: []const u16) f64 {
    var sum: f64 = 0;
    for (x, w) |a, b| sum += @as(f64, c.spec.fromBf16(a)) * c.spec.fromBf16(b);
    return sum;
}

test "Qwen BF16 reference: copied words and masked rows retain NaN and signed-zero payloads" {
    const table = [_]u16{ 0x7fc1, 0x8000, 0x7f80, 0xff80, 0x3f80, 0 };
    var out: [6]u16 = undefined;
    copy(&table, &.{ 1, 0, 2 }, 2, 2, &out);
    try std.testing.expectEqualSlices(u16, &.{ 0x7f80, 0xff80, 0x7fc1, 0x8000, 0xffff, 0xffff }, &out);
    out[2] = 0xffff; // a truncated real row is not masked padding
    try std.testing.expect(out[2] != table[0]);
}
test "Qwen BF16 reference: stress proves fp32 exact and both smaller accumulators wrong in every order" {
    var x: [128]u16 = undefined;
    stress(&x);
    const w: [128]u16 = @splat(0x3f80);
    const want = dot(&x, &w);
    try std.testing.expectEqual(@as(f64, 64), want);
    var terms: [128]f64 = undefined;
    for (&terms, x) |*v, b| v.* = c.spec.fromBf16(b);
    for ([_]c.gates.Order{ .sequential, .pairwise, .strided }) |order| {
        try std.testing.expectEqual(want, c.rows.sumIn(.f32, order, &terms));
        try std.testing.expect(@abs(c.rows.sumIn(.f16, order, &terms) - want) >= 16);
        try std.testing.expect(@abs(c.rows.sumIn(.bf16, order, &terms) - want) >= 16);
    }
}

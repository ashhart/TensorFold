//! Independent copy and FP64 dot models, with a stress row shared by the real kernel adapter.
const std = @import("std");
const core = @import("core");
const c = core.contract;

pub fn stress(x: []u16) void {
    const half = x.len / 2;
    var random = std.Random.DefaultPrng.init(0x35bf16);
    for (0..half) |i| {
        const value: f32 = @floatFromInt(1 + random.random().uintLessThan(u32, 64));
        x[i] = c.spec.toBf16(value);
        x[half + i] = c.spec.toBf16(-value + @as(f32, if (i < 8) 1 else 0));
    }
    if (x.len % 2 != 0) x[x.len - 1] = 0;
}
fn noDepth(_: c.spec.Shape) usize {
    return 0;
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
    var x: [17408]u16 = undefined;
    stress(&x);
    const w: [17408]u16 = @splat(0x3f80);
    const want = dot(&x, &w);
    try std.testing.expectEqual(@as(f64, 8), want);
    var terms: [17408]f64 = undefined;
    for (&terms, x) |*v, b| v.* = c.spec.fromBf16(b);
    for ([_]c.gates.Order{ .sequential, .pairwise, .strided }) |order| {
        try std.testing.expectEqual(want, c.rows.sumIn(.f32, order, &terms));
        for ([_]c.spec.Accum{ .f16, .bf16 }) |acc| {
            const got = c.rows.sumIn(acc, order, &terms);
            const miss = if (std.math.isFinite(got)) @abs(got - want) else std.math.inf(f64);
            const spec: c.spec.Spec = .{ .name = "stress", .accum = .f32, .depth = noDepth, .out = .bf16, .served = &.{.{ .rows = 1, .n = 1, .k = x.len }} };
            try std.testing.expect(miss / c.spec.boundExact(spec, want) >= 100);
        }
    }
}

test {
    _ = @import("qwen_scores_ref.zig");
}
